#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bout_save_v3 -- transport-coefficient post-processing for SIBIL / elm_6f.

New in v3 (the staleness problem)
---------------------------------
v1/v2 could silently reuse .npy files written by a PREVIOUS run.  That is how
numpy/ddr_Tetot.npy ended up with nt=7 while the live run had 15 outputs, and
with values bit-identical at t=0,1,5.

v3 refuses to trust a cached array unless ALL of these hold:

  (a) mtime(numpy/x.npy)  >  mtime(newest BOUT.dmp.*.nc)
  (b) mtime(numpy/x.npy)  >  mtime(BOUT.inp)        [optional, --no-check-inp]
  (c) the provenance manifest numpy/.provenance.json records the same
      dump-file fingerprint (name, size, mtime, nt) that is on disk now
  (d) the array's leading dimension matches the expected nt

Any failure -> the file is treated as absent and re-collected.  --audit prints
the table and exits without doing any work; --purge deletes the stale entries.

Also in v3
----------
* diagnose_ddr() now separates toroidal from radial variation.  v2 sliced
  [t,:,j,:] and called the spread "z-std", but that slice spans x AND z, so
  radial structure dominated and the number could not distinguish a
  perturbation-only field from a frozen one.  v3 reports, per field:
      - z-mean / z-std at a FIXED (x,y)  -> genuine toroidal fluctuation level
      - max|f(t) - f(0)|                 -> whether the field evolves at all
* derived products carry their own provenance entry, so editing this script
  (mtime of __file__) also invalidates them.

Conventions (unchanged from v2)
-------------------------------
    Particles:  Gamma_r = -D   dn/dr   =>  D   = -<n Vr>_z   / (d<n>_z/dr)
    Heat:       q_r     = -n chi dT/dr =>  chi = -C <T Vr>_z / (d<T>_z/dr)
        C = 3/2 conductive, 5/2 total enthalpy.  Density cancels in chi.
    Fluxes are CORRELATIONS <f*Vr>_z, formed in 3D before averaging.
"""

import os
import gc
import json
import time as _time
import argparse
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from boutdata.collect import collect
from boutdata.data import BoutData
from boututils.datafile import DataFile

HEAT_COEFF = 1.5      # 1.5 internal energy, 2.5 total enthalpy
REL_EPS = 1.0e-3      # mask |grad| < REL_EPS * max|grad|
MANIFEST = ".provenance.json"


# ===========================================================================
# provenance / staleness
# ===========================================================================

def dump_files(data_dir):
    """Sorted list of BOUT.dmp.*.nc in data_dir."""
    return sorted(Path(data_dir).glob("BOUT.dmp.*.nc"))


def dump_fingerprint(data_dir):
    """Fingerprint of the current output set.

    Deliberately does NOT hash file contents -- dumps are multi-GB.  Name +
    size + mtime + nt is enough to catch a re-run, a restart that appended
    records, or a different number of ranks.
    """
    files = dump_files(data_dir)
    if not files:
        raise FileNotFoundError(f"no BOUT.dmp.*.nc in {data_dir}")

    entries = []
    nts = []
    newest = 0.0
    for f in files:
        st = f.stat()
        with Dataset(f) as ds:
            nt = len(ds.dimensions["t"])
        nts.append(nt)
        newest = max(newest, st.st_mtime)
        entries.append({"name": f.name,
                        "size": st.st_size,
                        "mtime": round(st.st_mtime, 3),
                        "nt": nt})

    return {"nfiles": len(files),
            "newest_mtime": round(newest, 3),
            "nt_min": min(nts),
            "nt_max": max(nts),
            "files": entries}


def load_manifest(numpy_dir):
    p = Path(numpy_dir) / MANIFEST
    if not p.exists():
        return {}
    try:
        with open(p) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        print(f"  ! {p} unreadable -- treating all cached arrays as stale")
        return {}


def save_manifest(numpy_dir, man):
    with open(Path(numpy_dir) / MANIFEST, "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True)


def _same_dumps(a, b):
    """Compare two fingerprints, ignoring key order."""
    if not a or not b:
        return False
    for k in ("nfiles", "nt_min", "nt_max"):
        if a.get(k) != b.get(k):
            return False
    fa = {e["name"]: (e["size"], e["mtime"], e["nt"]) for e in a.get("files", [])}
    fb = {e["name"]: (e["size"], e["mtime"], e["nt"]) for e in b.get("files", [])}
    return fa == fb


class Cache:
    """Decides whether a cached numpy/<name>.npy may be reused."""

    def __init__(self, data_dir, numpy_dir, fp, force=False,
                 check_inp=True, script_mtime=None):
        self.data_dir = Path(data_dir)
        self.numpy_dir = Path(numpy_dir)
        self.fp = fp
        self.force = force
        self.check_inp = check_inp
        self.script_mtime = script_mtime or 0.0
        self.man = load_manifest(numpy_dir)

        inp = self.data_dir / "BOUT.inp"
        self.inp_mtime = inp.stat().st_mtime if inp.exists() else 0.0

        # manifest-wide check: do the dumps on disk match what the manifest
        # was built against?  If not, nothing cached is trustworthy.
        self.dumps_match = _same_dumps(self.man.get("dumps"), fp)
        if self.man and not self.dumps_match:
            print("  ! output files have changed since numpy/ was written "
                  "-- ALL cached arrays are stale")

    # -- decision ----------------------------------------------------------
    def status(self, name, nt_expect=None, derived=False):
        """Return (ok, reason). ok=True means the cached file may be reused."""
        p = self.numpy_dir / f"{name}.npy"

        if self.force:
            return False, "--force"
        if not p.exists():
            return False, "absent"

        mt = p.stat().st_mtime
        if mt <= self.fp["newest_mtime"]:
            return False, (f"older than newest dump by "
                           f"{_age(self.fp['newest_mtime'] - mt)}")
        if self.check_inp and mt <= self.inp_mtime:
            return False, "older than BOUT.inp"
        if derived and mt <= self.script_mtime:
            return False, "older than this script"
        if not self.dumps_match:
            return False, "dump fingerprint changed"

        try:
            shp = np.load(p, mmap_mode="r").shape
        except (ValueError, OSError) as e:
            return False, f"unreadable ({e.__class__.__name__})"

        if nt_expect is not None and len(shp) == 4 and shp[0] != nt_expect:
            return False, f"nt={shp[0]}, expected {nt_expect}"

        rec = self.man.get("arrays", {}).get(name)
        if rec is None:
            return False, "no manifest entry"
        if tuple(rec.get("shape", ())) != tuple(shp):
            return False, f"shape {shp} != manifest {tuple(rec['shape'])}"

        return True, "ok"

    def record(self, name, arr, derived=False):
        self.man.setdefault("arrays", {})[name] = {
            "shape": list(np.shape(arr)),
            "dtype": str(np.asarray(arr).dtype),
            "written": round(_time.time(), 3),
            "derived": bool(derived),
        }

    def commit(self):
        self.man["dumps"] = self.fp
        self.man["data_dir"] = str(self.data_dir.resolve())
        self.man["script_mtime"] = round(self.script_mtime, 3)
        self.man["updated"] = round(_time.time(), 3)
        save_manifest(self.numpy_dir, self.man)


def _age(sec):
    sec = abs(float(sec))
    if sec < 90:
        return f"{sec:.0f}s"
    if sec < 5400:
        return f"{sec / 60:.0f}m"
    if sec < 172800:
        return f"{sec / 3600:.1f}h"
    return f"{sec / 86400:.1f}d"


def audit(data_dir, numpy_dir, check_inp=True):
    """Print a staleness table for everything in numpy/ and return it."""
    numpy_dir = Path(numpy_dir)
    fp = dump_fingerprint(data_dir)
    cache = Cache(data_dir, numpy_dir, fp, check_inp=check_inp,
                  script_mtime=_script_mtime())

    ragged = fp["nt_min"] != fp["nt_max"]
    print(f"data_dir        : {Path(data_dir).resolve()}")
    print(f"dump files      : {fp['nfiles']}  nt={fp['nt_min']}"
          + (f"..{fp['nt_max']}  (RAGGED)" if ragged else ""))
    print(f"newest dump     : {_time.ctime(fp['newest_mtime'])}")
    if cache.inp_mtime:
        print(f"BOUT.inp        : {_time.ctime(cache.inp_mtime)}")
    print(f"manifest        : {'present' if cache.man else 'MISSING'}"
          + ("" if cache.dumps_match or not cache.man else "  (dumps changed)"))
    print()

    files = sorted(numpy_dir.glob("*.npy"))
    if not files:
        print("numpy/ is empty")
        return []

    rows = []
    w = max(max(len(f.stem) for f in files), 12)
    print(f"{'array':<{w}}  {'shape':>18}  {'vs newest dump':>15}  status")
    print("-" * (w + 56))
    for f in files:
        name = f.stem
        try:
            shp = np.load(f, mmap_mode="r").shape
        except Exception:
            shp = ("?",)
        ok, why = cache.status(name, nt_expect=fp["nt_min"])
        delta = f.stat().st_mtime - fp["newest_mtime"]
        tag = ("+" if delta > 0 else "-") + _age(delta)
        print(f"{name:<{w}}  {str(shp):>18}  {tag:>15}  "
              f"{'OK' if ok else 'STALE: ' + why}")
        rows.append((name, shp, ok, why))

    nstale = sum(1 for r in rows if not r[2])
    print(f"\n{nstale}/{len(rows)} arrays stale")
    if nstale:
        print("re-run without --audit (stale entries refresh automatically), "
              "or add --purge to delete them first")
    return rows


def purge(numpy_dir, rows):
    """Delete every array flagged stale by audit()."""
    n = 0
    for name, _shp, ok, _why in rows:
        if not ok:
            p = Path(numpy_dir) / f"{name}.npy"
            if p.exists():
                p.unlink()
                n += 1
    print(f"purged {n} stale arrays")


def _script_mtime():
    try:
        return Path(__file__).stat().st_mtime
    except (NameError, OSError):
        return 0.0


# ===========================================================================
# numerics
# ===========================================================================

def safe_div(num, den, rel_eps=REL_EPS):
    """num/den with a RELATIVE floor on the denominator -> NaN where unreliable."""
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    scale = np.nanmax(np.abs(den))
    if not np.isfinite(scale) or scale == 0.0:
        return np.full(num.shape, np.nan)
    bad = ~np.isfinite(den) | (np.abs(den) < rel_eps * scale)
    out = np.divide(num, np.where(bad, 1.0, den))
    out[bad] = np.nan
    return out


def load_var(numpy_dir, name):
    return np.load(Path(numpy_dir) / f"{name}.npy", mmap_mode="r")


def radial_arclength(Rxy, Zxy):
    """Arc length [m] between adjacent radial grid points, shape (nx, ny)."""
    Rxy = np.asarray(Rxy, dtype=float)
    Zxy = np.asarray(Zxy, dtype=float)
    dr = np.empty_like(Rxy)
    dr[1:-1, :] = 0.5 * np.hypot(Rxy[2:, :] - Rxy[:-2, :],
                                 Zxy[2:, :] - Zxy[:-2, :])
    dr[0, :] = np.hypot(Rxy[1, :] - Rxy[0, :], Zxy[1, :] - Zxy[0, :])
    dr[-1, :] = np.hypot(Rxy[-1, :] - Rxy[-2, :], Zxy[-1, :] - Zxy[-2, :])
    dr[dr == 0.0] = np.nan
    return dr


def _ddx_index(f):
    """Centred difference along axis -2 (x) in INDEX space, one-sided at ends.

    Identical stencil to BOUT++ DDX with ddx:first = C2, which is what
    BOUT.inp specifies.
    """
    f = np.asarray(f, dtype=float)
    out = np.empty_like(f)
    out[..., 1:-1, :] = 0.5 * (f[..., 2:, :] - f[..., :-2, :])
    out[..., 0, :] = f[..., 1, :] - f[..., 0, :]
    out[..., -1, :] = f[..., -1, :] - f[..., -2, :]
    return out


def ddr(f, dr):
    """d/dr via arc length in the (R,Z) plane.  Kept as a cross-check only.

    Exact only on an ORTHOGONAL grid: it measures distance along the radial
    grid line, which equals distance along grad(psi) only if the two are
    parallel.  SOLPS-ITER grids are often non-orthogonal near the X-point,
    where this OVERESTIMATES the distance and so UNDERESTIMATES the gradient.
    Use ddr_metric() for anything quantitative.
    """
    return _ddx_index(f) / dr


def ddr_metric(f, Rxy, Bpxy, dx):
    """d/dr exactly as the .cxx model computes it:  sqrt(g11) * DDX(f).

        df/dr = (grad(psi) . grad f) / |grad psi|
              = sqrt(g11) * df/dx,        g11 = |grad psi|^2 = (R Bp)^2
              = R*Bp * df/dpsi

    This is metric-exact on any grid, orthogonal or not, and it reproduces
    the code-side ddr_* diagnostics bit for bit (given g13 = 0, which is the
    case for a shifted-metric grid -- check with grid["g13"]).

    Parameters
    ----------
    f    : (x,y) or (t,x,y)
    Rxy  : (nx,ny) major radius [m]
    Bpxy : (nx,ny) poloidal field [T]
    dx   : (nx,ny) grid-file dx, i.e. dpsi per cell [Wb/rad]

    Returns df/dr in units of [f] per metre.
    """
    Rxy = np.asarray(Rxy, dtype=float)
    Bpxy = np.asarray(Bpxy, dtype=float)
    dx = np.asarray(dx, dtype=float)
    if dx.ndim == 1:                      # some grids store dx as (nx,)
        dx = dx[:, None] * np.ones_like(Rxy)
    dxs = np.where(dx == 0.0, np.nan, dx)
    return _ddx_index(f) / dxs * (Rxy * Bpxy)


def zavg(arr4):
    return np.asarray(arr4).mean(axis=3)


# ===========================================================================
# diagnostics on the dumped ddr_* fields
# ===========================================================================

def diagnose_ddr(numpy_dir, name, jomp=48, ix=None, ntimes=6):
    """Is a dumped gradient diagnostic usable?

    v2's version sliced [t,:,j,:] and reported the spread as "z-std", but that
    slice spans x AND z, so radial structure dominated and the number said
    nothing about toroidal fluctuation.  v3 separates them:

        z-mean / z-std at a FIXED (x,y)  -> toroidal fluctuation level
        max|f(t) - f(0)|                 -> does the field evolve at all

    FROZEN  -> stale file, or diagnostic written only at initialisation.
    PERTURBATION-ONLY -> z-mean negligible against z-std.
    """
    p = Path(numpy_dir) / f"{name}.npy"
    if not p.exists():
        print(f"  {name}: not collected")
        return
    g = np.load(p, mmap_mode="r")
    if g.ndim != 4:
        print(f"  {name}: ndim={g.ndim} (expected 4) shape={g.shape}")
        return

    nt, nx, ny, nz = g.shape
    j = min(jomp, ny - 1)
    if ix is None:
        prof = np.abs(np.asarray(g[0, :, j, :], dtype=float).mean(axis=1))
        ix = int(np.nanargmax(prof)) if np.any(np.isfinite(prof)) else nx // 2

    print(f"  {name}  shape={g.shape}  probing (ix={ix}, iy={j})")
    ts = sorted(set(np.linspace(0, nt - 1, min(ntimes, nt)).astype(int)))
    f0 = np.asarray(g[0, :, j, :], dtype=float)
    for t in ts:
        line = np.asarray(g[t, ix, j, :], dtype=float)      # pure z line
        m, s = np.nanmean(line), np.nanstd(line)
        evo = np.nanmax(np.abs(np.asarray(g[t, :, j, :], dtype=float) - f0))
        print(f"    t={t:3d}  z-mean={m: .4e}  z-std={s: .4e}  "
              f"max|f(t)-f(0)|={evo: .4e}")

    f_last = np.asarray(g[-1, :, j, :], dtype=float)
    if np.array_equal(f0, f_last):
        print("    -> FROZEN: bit-identical at first and last record. "
              "Stale file, or written only at initialisation.")
    else:
        zstd = np.nanstd(f_last, axis=1)
        zmen = np.abs(np.nanmean(f_last, axis=1))
        if np.nanmedian(zmen) < 1e-3 * np.nanmedian(zstd):
            print("    -> PERTURBATION-ONLY: z-mean negligible vs z-std.")
        else:
            print("    -> looks like a total (equilibrium+perturbed) field.")


def verify_gradient(numpy_dir, grid_fname, jomp=48, t=0):
    """Does the Python ddr_metric reproduce the model's sqrt(g11)*DDX?

    Compares the z-average of the dumped ddr_nitot against ddr_metric applied
    to <Ni>_z + N0.  They must agree to a few percent at t=0; the only
    legitimate sources of difference are guard-cell handling and the
    normalisation factor Nbar/Lbar.

    A LARGE discrepancy means the dx convention or the normalisation is wrong.
    A discrepancy that is a CONSTANT FACTOR is the normalisation, and the
    printed ratio is the factor you need.
    """
    nd = Path(numpy_dir)
    need = ["ddr_nitot.npy", "N0.npy", "Ni.npy"]
    if not all((nd / f).exists() for f in need):
        print("verify_gradient: need " + ", ".join(need))
        return

    with DataFile(grid_fname) as g:
        Rxy = np.asarray(g["Rxy"], dtype=float)
        Bpxy = np.asarray(g["Bpxy"], dtype=float)
        dxg = np.asarray(g["dx"], dtype=float)

    code = np.load(nd / "ddr_nitot.npy", mmap_mode="r")
    N0 = np.asarray(np.load(nd / "N0.npy"), dtype=float)
    Ni = np.load(nd / "Ni.npy", mmap_mode="r")

    t = min(t, code.shape[0] - 1, Ni.shape[0] - 1)
    j = min(jomp, Rxy.shape[1] - 1)

    code_z = np.asarray(code[t], dtype=float).mean(axis=2)
    ntot = np.asarray(Ni[t], dtype=float).mean(axis=2) + N0
    mine = ddr_metric(ntot[None, :, :], Rxy, Bpxy, dxg)[0]

    a = code_z[2:-2, j]
    b = mine[2:-2, j]
    good = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 0.05 * np.nanmax(np.abs(a)))
    if good.sum() < 5:
        print("verify_gradient: too few usable points")
        return
    ratio = np.nanmedian(b[good] / a[good])
    rel = np.nanmax(np.abs(b[good] - a[good])) / (np.nanmax(np.abs(a[good])) + 1e-300)

    print(f"verify_gradient  (t={t}, j={j}, interior only)")
    print(f"  code  sqrt(g11)*DDX(Ni+N0), z-avg : peak {np.nanmax(np.abs(a)):.4e}")
    print(f"  python ddr_metric                 : peak {np.nanmax(np.abs(b)):.4e}")
    print(f"  median ratio python/code = {ratio:.6f}")
    print(f"  max relative difference  = {rel:.3%}")
    if abs(ratio - 1.0) < 0.05 and rel < 0.10:
        print("  -> AGREE. ddr_metric reproduces the model's radial derivative.")
    elif abs(rel - abs(ratio - 1.0)) < 0.05:
        print(f"  -> CONSTANT FACTOR {ratio:.6g}. Almost certainly the dx "
              "convention or a missing Nbar/Lbar normalisation.")
    else:
        print("  -> DISAGREE in shape, not just scale. Check that dx in the "
              "grid file is dpsi per cell and not something else.")
    return ratio


# ===========================================================================
# collection
# ===========================================================================

def save_variable(data_dir, numpy_dir, key, time, cache, nt_expect=None):
    ok, why = cache.status(key, nt_expect=nt_expect)
    if ok:
        print(f"  {key:<12} cached")
        cache.record(key, np.load(Path(numpy_dir) / f"{key}.npy", mmap_mode="r"))
        return

    print(f"  {key:<12} collecting   ({why})")
    nt_min, nt_max = cache.fp["nt_min"], cache.fp["nt_max"]
    if nt_min != nt_max:
        print(f"    ! ragged dumps nt={nt_min}..{nt_max}; truncating to {nt_min}")
    arr = collect(key, path=data_dir, info=False, tind=[0, nt_min - 1])

    if time is not None and np.ndim(arr) == 4:
        arr = arr[time, ...]

    np.save(Path(numpy_dir) / f"{key}.npy", arr)
    cache.record(key, arr)
    cache.commit()
    del arr
    gc.collect()


# ===========================================================================
# main
# ===========================================================================

def bout_save_data(data_dir, grid_fname, var_names, time, debug,
                   force=False, jomp=48, heat_coeff=HEAT_COEFF,
                   check_inp=True):

    data_dir = str(data_dir)
    numpy_dir = Path(data_dir) / "numpy"
    numpy_dir.mkdir(exist_ok=True)

    fp = dump_fingerprint(data_dir)
    cache = Cache(data_dir, numpy_dir, fp, force=force,
                  check_inp=check_inp, script_mtime=_script_mtime())

    nt_min = fp["nt_min"]
    nt_expect = nt_min if time is None else len(time)
    ragged = fp["nt_min"] != fp["nt_max"]
    print(f"dumps: {fp['nfiles']} files, nt={nt_min}"
          f"{'..' + str(fp['nt_max']) + ' (RAGGED)' if ragged else ''}"
          f"  -> expecting nt={nt_expect}\n")

    d = BoutData(path=data_dir)
    output_keys = list(d["outputs"].keys())

    def put(name, arr, derived=True):
        np.save(numpy_dir / f"{name}.npy", arr)
        cache.record(name, arr, derived=derived)

    def save_debug(name, arr):
        if debug:
            put(name, arr)

    # -- 1. raw variables ---------------------------------------------------
    print("Raw BOUT++ outputs")
    keys_to_save = output_keys if var_names[0] == "All" else var_names
    for key in keys_to_save:
        if key not in output_keys:
            print(f"  {key:<12} NOT IN DUMPS, skipping")
            continue
        save_variable(data_dir, numpy_dir, key, time, cache,
                      nt_expect=nt_expect)
    cache.commit()

    # -- 2. grid ------------------------------------------------------------
    with DataFile(grid_fname) as g:
        gkeys = list(g.keys())
        Rxy = np.asarray(g["Rxy"], dtype=float)
        Zxy = np.asarray(g["Zxy"], dtype=float)
        Bpxy = np.asarray(g["Bpxy"], dtype=float)
        dxg = np.asarray(g["dx"], dtype=float)
        g13 = np.asarray(g["g13"], dtype=float) if "g13" in gkeys else None

    dr = radial_arclength(Rxy, Zxy)
    put("dr_m", dr)

    # sqrt(g11) = R*Bp.  Report the non-orthogonality of the grid: the ratio
    # of the metric spacing dpsi/|grad psi| to the (R,Z) arc length is 1 on an
    # orthogonal grid and < 1 where the radial grid line leans off grad(psi).
    sqrt_g11 = Rxy * Bpxy
    dx2 = dxg[:, None] * np.ones_like(Rxy) if dxg.ndim == 1 else dxg
    with np.errstate(invalid="ignore", divide="ignore"):
        ortho = (dx2 / np.where(sqrt_g11 == 0, np.nan, sqrt_g11)) / dr
    put("sqrt_g11", sqrt_g11)
    put("orthogonality", ortho)

    j_ = min(jomp, Rxy.shape[1] - 1)
    print(f"\nGrid metric")
    if g13 is None:
        print("  g13 absent -> shifted metric, sqrt(g11)*DDX is the exact d/dr")
    else:
        mx = float(np.nanmax(np.abs(g13)))
        print(f"  max|g13| = {mx:.3e} -> "
              + ("shifted metric; no twist-shift correction needed"
                 if mx == 0.0 else
                 "NONZERO: a -I*d/dz term is missing from any sqrt(g11)*DDX"))
    print(f"  orthogonality (metric dr / arc-length dr), 1.0 = orthogonal:")
    print(f"    at j={j_}: median {np.nanmedian(ortho[:, j_]):.4f}   "
          f"min {np.nanmin(ortho[:, j_]):.4f}")
    print(f"    whole grid: median {np.nanmedian(ortho):.4f}   "
          f"min {np.nanmin(ortho):.4f}  "
          f"(at j={int(np.unravel_index(np.nanargmin(ortho), ortho.shape)[1])})")

    # -- 3. are the dumped gradients usable? --------------------------------
    print("\nDumped ddr_* diagnostics")
    for nm in ("ddr_nitot", "ddr_Tetot", "ddr_Titot", "ddr_ni"):
        diagnose_ddr(numpy_dir, nm, jomp=jomp)
    print()
    try:
        verify_gradient(numpy_dir, grid_fname, jomp=jomp, t=0)
    except Exception as exc:
        print(f"  verify_gradient failed: {exc}")
    print("  (gradients below are the DUMPED ddr_*tot fields, z-averaged)\n")

    # -- 4. total z-averaged profiles ---------------------------------------
    print("Derived quantities")

    def total_profile(base_key, pert_key, out_name):
        if base_key not in output_keys or pert_key not in output_keys:
            return None
        base = np.asarray(load_var(numpy_dir, base_key), dtype=float)
        pert = load_var(numpy_dir, pert_key)
        txy = zavg(pert) + base[None, :, :]
        put(f"{out_name}_txy", txy)
        put(f"{out_name}_xy", txy.mean(axis=0))
        del base, pert
        gc.collect()
        return txy

    ni_txy = total_profile("N0", "Ni", "ni")
    Ti_txy = total_profile("Ti0", "Ti", "Ti")
    Te_txy = total_profile("Te0", "Te", "Te")

    if ni_txy is None or Ti_txy is None or Te_txy is None:
        print("Missing N0/Ni, Ti0/Ti or Te0/Te -- cannot continue")
        cache.commit()
        return

    # ---- gradients: the model's own sqrt(g11)*DDX(f), dumped as ddr_*tot,
    #      z-averaged to (t,x,y)
    gkeys_ = ("ddr_nitot", "ddr_Tetot", "ddr_Titot")
    missing = [k for k in gkeys_ if k not in output_keys or k not in keys_to_save]
    if missing:
        print(f"Missing {', '.join(missing)} -- cannot build coefficients")
        cache.commit()
        return
    grad_ni = zavg(load_var(numpy_dir, "ddr_nitot"))
    grad_Te = zavg(load_var(numpy_dir, "ddr_Tetot"))
    grad_Ti = zavg(load_var(numpy_dir, "ddr_Titot"))

    # Written under BOTH naming conventions.  The *tot* names are what the
    # v1 pipeline and the interactive plotter load; without these aliases the
    # notebook silently keeps reading whatever stale v1 file is on disk.
    for base, arr in (("ni", grad_ni), ("Te", grad_Te), ("Ti", grad_Ti)):
        put(f"ddr_{base}_txy", arr)
        put(f"ddr_{base}_xy", np.nanmean(arr, axis=0))
    put("ddr_nitot_txy", grad_ni)
    put("ddr_Tetot_txy", grad_Te)
    put("ddr_Titot_txy", grad_Ti)
    put("ddr_nitot_xy", np.nanmean(grad_ni, axis=0))
    put("ddr_Tetot_xy", np.nanmean(grad_Te, axis=0))
    put("ddr_Titot_xy", np.nanmean(grad_Ti, axis=0))

    # arc-length version, for the orthogonality cross-check only
    save_debug("ddr_ni_arclen_txy", ddr(ni_txy, dr))

    # the gradients MUST evolve; if they do not, nothing downstream is real
    print("  gradient evolution check")
    for nm, gg in (("ddr_ni", grad_ni), ("ddr_Te", grad_Te), ("ddr_Ti", grad_Ti)):
        j = min(jomp, gg.shape[2] - 1)
        evo = np.nanmax(np.abs(gg[-1, :, j] - gg[0, :, j]))
        ref = np.nanmax(np.abs(gg[0, :, j]))
        print(f"    {nm:<8} t=0 peak {ref:.4e}   "
              f"max change over run {evo:.4e} ({evo / (ref + 1e-300):.2%})")
        if evo == 0.0:
            print("             ^ FROZEN -- check that Ni/Te/Ti themselves evolve")

    # -- 5. fluxes: correlations, not products of averages ------------------
    need = ("Vexbx", "Vbtildx", "Va")
    missing = [k for k in need if k not in output_keys]
    if missing:
        print(f"Missing {', '.join(missing)} -- cannot build fluxes")
        cache.commit()
        return

    Va = float(np.asarray(load_var(numpy_dir, "Va")).ravel()[0])
    print(f"\nVa = {Va:.6g} m/s")

    Vexbx = load_var(numpy_dir, "Vexbx")
    Vbtildx = load_var(numpy_dir, "Vbtildx")
    N0 = np.asarray(load_var(numpy_dir, "N0"), dtype=float)
    Te0 = np.asarray(load_var(numpy_dir, "Te0"), dtype=float)
    Ti0 = np.asarray(load_var(numpy_dir, "Ti0"), dtype=float)
    Ni = load_var(numpy_dir, "Ni")
    Te = load_var(numpy_dir, "Te")
    Ti = load_var(numpy_dir, "Ti")

    nt = Ni.shape[0]
    Gamma = np.zeros(Ni.shape[:3])
    qe = np.zeros(Ni.shape[:3])
    qi = np.zeros(Ni.shape[:3])
    Vr_mean = np.zeros(Ni.shape[:3])

    # Vexb.covariant = false in the model, so Vexbx is the CONTRAVARIANT
    # component V^x = V.grad(psi).  The physical radial velocity is
    # V^x / |grad psi| = V^x / sqrt(g11).  Without this the ratio
    # flux/gradient is wrong by a factor g11 = (R Bp)^2, which varies
    # fastest exactly at the separatrix.
    inv_sg11 = 1.0 / np.where(sqrt_g11 == 0, np.nan, sqrt_g11)
    print("  applying 1/sqrt(g11) to Vexbx, Vbtildx (contravariant -> physical)")

    Vexbx_txy = np.zeros(Ni.shape[:3])
    Vbtildx_txy = np.zeros(Ni.shape[:3])

    print("Computing <f*Vr>_z one timestep at a time")
    for t in range(nt):
        ve = np.asarray(Vexbx[t], dtype=float) * inv_sg11[:, :, None] * Va
        vb = np.asarray(Vbtildx[t], dtype=float) * inv_sg11[:, :, None] * Va
        vr = ve + vb
        Vexbx_txy[t] = ve.mean(axis=2)
        Vbtildx_txy[t] = vb.mean(axis=2)
        del ve, vb
        n_t = np.asarray(Ni[t], dtype=float) + N0[:, :, None]
        te_t = np.asarray(Te[t], dtype=float) + Te0[:, :, None]
        ti_t = np.asarray(Ti[t], dtype=float) + Ti0[:, :, None]

        Gamma[t] = (n_t * vr).mean(axis=2)
        qe[t] = heat_coeff * (te_t * vr).mean(axis=2)
        qi[t] = heat_coeff * (ti_t * vr).mean(axis=2)
        Vr_mean[t] = vr.mean(axis=2)

        del vr, n_t, te_t, ti_t
    gc.collect()

    put("Gamma_txy", Gamma)
    put("qe_txy", qe)
    put("qi_txy", qi)

    # names the interactive plotter loads
    put("Vexbx_txy", Vexbx_txy)
    put("Vbtildx_txy", Vbtildx_txy)
    put("Vexbx_xy", Vexbx_txy.mean(axis=0))
    put("Vbtildx_xy", Vbtildx_txy.mean(axis=0))
    put("Vr_t", Vr_mean)
    put("Vr", Vr_mean.mean(axis=0))

    # <V_ExB>_z is zero by symmetry: V^x_ExB ~ d(phi)/dz, and the z-average of
    # a z-derivative vanishes identically on a periodic domain.  A value at
    # roundoff is CORRECT, not a broken Laplacian solve.  Report the ratio so
    # the magnitude is never mistaken for signal again.
    jj = min(jomp, Vr_mean.shape[2] - 1)
    zonal = np.nanmax(np.abs(Vr_mean[:, :, jj]))
    full = np.nanmax(np.abs(Gamma[:, :, jj])) / (
        np.nanmax(np.abs(ni_txy[:, :, jj])) + 1e-300)
    print(f"\n  |<Vr>_z|max       = {zonal:.3e} m/s   (zero by symmetry)")
    print(f"  |<n Vr>_z / <n>|  = {full:.3e} m/s   (the real transport)")
    if full > 0 and zonal / full < 1e-6:
        print("  -> zonal Vr is pure roundoff; v1 built every coefficient "
              "from this. All v1 D/chi values were amplified noise.")

    Gamma_meanonly = ni_txy * Vr_mean
    with np.errstate(invalid="ignore", divide="ignore"):
        frac = (np.nanmax(np.abs(Gamma_meanonly))
                / (np.nanmax(np.abs(Gamma)) + 1e-300))
    print(f"\n  <n><Vr> / <n Vr>  peak ratio = {frac:.3e}")
    print("  (v1 kept only the numerator; << 1 means v1 discarded "
          "essentially all transport)")
    save_debug("Vr_mean_txy", Vr_mean)
    save_debug("Gamma_meanonly_txy", Gamma_meanonly)

    # -- 6. transport coefficients -----------------------------------------
    print("\nTransport coefficients")
    Da_t = -safe_div(Gamma, grad_ni)
    Xe_t = -safe_div(qe, grad_Te)
    Xi_t = -safe_div(qi, grad_Ti)
    put("Da_t", Da_t)
    put("Xe_t", Xe_t)
    put("Xi_t", Xi_t)

    Da_eff = -safe_div(np.nanmean(Gamma, axis=0), np.nanmean(grad_ni, axis=0))
    Xe_eff = -safe_div(np.nanmean(qe, axis=0), np.nanmean(grad_Te, axis=0))
    Xi_eff = -safe_div(np.nanmean(qi, axis=0), np.nanmean(grad_Ti, axis=0))
    for nm, a in (("Da_eff", Da_eff), ("Xe_eff", Xe_eff), ("Xi_eff", Xi_eff)):
        put(nm, a)
    for nm, a in (("Da", Da_eff), ("Xe", Xe_eff), ("Xi", Xi_eff)):
        put(nm, a)   # backwards-compatible names

    for nm, a in (("D", Da_t), ("chi_e", Xe_t), ("chi_i", Xi_t)):
        print(f"  {nm:6s} finite {np.isfinite(a).sum() / a.size:6.1%}   "
              f"median |.| = {np.nanmedian(np.abs(a)):.3e} m^2/s   "
              f"max = {np.nanmax(np.abs(a)):.3e}")

    cache.commit()
    print(f"\nmanifest written: {numpy_dir / MANIFEST}")
    print("Done")


# ===========================================================================

def find_grid(data_dir):
    inp = Path(data_dir) / "BOUT.inp"
    if not inp.exists():
        inp = Path("BOUT.inp")
    with open(inp) as f:
        for line in f:
            s = line.split("#")[0]
            if "grid" in s and "=" in s and '"' in s:
                return Path(os.path.expandvars(s.split('"')[1]))
    raise RuntimeError(f'could not find grid = "..." in {inp}')


if __name__ == "__main__":

    default_vars = ["N0", "Ni", "Ti", "Ti0", "Te", "Te0", "Va",
                    "U", "Phi", "Vexbx", "Vbtildx",
                    "ddr_nitot", "ddr_Titot", "ddr_Tetot",
                    "ddr_ni", "ddr_Ti", "ddr_Te",
                    "T_R", "T_M", "T_C", "T_G"]

    def flag(v):
        return str(v).lower() not in ("false", "0", "no", "n", "")

    ap = argparse.ArgumentParser(prog="bout_save_v3")
    ap.add_argument("path", nargs="?", type=Path, default=Path("."))
    ap.add_argument("-v", "--vars", nargs="+", default=default_vars)
    ap.add_argument("-t", "--time", nargs="+", default=["All"],
                    help="'All' or integer time indices")
    ap.add_argument("-d", "--debug", type=flag, default=True)
    ap.add_argument("-c", "--clean", type=flag, default=False)
    ap.add_argument("-f", "--force", type=flag, default=False,
                    help="re-collect everything regardless of provenance")
    ap.add_argument("-j", "--jomp", type=int, default=48)
    ap.add_argument("--heat-coeff", type=float, default=HEAT_COEFF)
    ap.add_argument("--audit", action="store_true",
                    help="report staleness of numpy/*.npy and exit")
    ap.add_argument("--purge", action="store_true",
                    help="delete stale arrays (runs an audit first)")
    ap.add_argument("--no-check-inp", action="store_true",
                    help="do not invalidate caches when BOUT.inp is newer")
    args = ap.parse_args()

    check_inp = not args.no_check_inp
    numpy_dir = Path(args.path) / "numpy"

    if args.audit or args.purge:
        numpy_dir.mkdir(exist_ok=True)
        rows = audit(args.path, numpy_dir, check_inp=check_inp)
        if args.purge:
            purge(numpy_dir, rows)
        if args.audit:
            raise SystemExit(0)

    time = None if args.time[0] == "All" else [int(s) for s in args.time]

    bout_save_data(args.path, str(find_grid(args.path)), args.vars, time,
                   args.debug, force=args.force, jomp=args.jomp,
                   heat_coeff=args.heat_coeff, check_inp=check_inp)

    if args.clean:
        keep = {"Da.npy", "Da_t.npy", "Da_eff.npy",
                "Xe.npy", "Xe_t.npy", "Xe_eff.npy",
                "Xi.npy", "Xi_t.npy", "Xi_eff.npy",
                "Gamma_txy.npy", "qe_txy.npy", "qi_txy.npy",
                "ddr_ni_txy.npy", "ddr_Te_txy.npy", "ddr_Ti_txy.npy",
                "ddr_nitot_txy.npy", "ddr_Tetot_txy.npy", "ddr_Titot_txy.npy",
                "Vexbx_txy.npy", "Vbtildx_txy.npy", "Vr_t.npy", "Vr.npy",
                "sqrt_g11.npy", "orthogonality.npy",
                "ni_txy.npy", "Te_txy.npy", "Ti_txy.npy",
                "dr_m.npy", "Ti0.npy", "Te0.npy", "N0.npy",
                "U.npy", "Phi.npy",
                "Ni.npy", "Te.npy", "Ti.npy",
                "T_R.npy", "T_M.npy", "T_C.npy", "T_G.npy"}
        for fn in os.listdir(numpy_dir):
            if fn.endswith(".npy") and fn not in keep:
                print(f"Cleaning up: {fn}")
                os.remove(numpy_dir / fn)
