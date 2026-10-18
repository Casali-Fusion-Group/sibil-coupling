#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_fch_conventions.py

Decide, once and for all, whether the small SOLPS parallel current you get out
of ``b2fstate.fch`` is a *convention bug* in the extraction or *real physics*.

The test is charge conservation.  In steady state SOLPS solves

    div(j) = 0

cell by cell, so for the face-integrated currents stored in ``fch`` the net
current out of every interior cell must vanish to round-off:

    [ fchx(ix+1,iy) - fchx(ix,iy) ] + [ fchy(ix,iy+1) - fchy(ix,iy) ] = 0

This identity holds **only** if all three of the following are right:

  1. which component of ``fch[..., k]`` is poloidal and which is radial,
  2. which face (left/lower vs right/upper) the stored value belongs to,
  3. the sign convention (directed along +index vs outward normal).

So we simply try the plausible variants and see which one conserves charge.
Whichever variant wins is the one ``solps_jpar_to_bout.py`` must use.

Usage
-----
    python check_fch_conventions.py SIBIL_config.yaml

or explicitly

    python check_fch_conventions.py \
        --b2fstate /run/b2fstate --b2fgmtry /run/b2fgmtry \
        [--b2mn /run/b2mn.dat] [--geofile /run/solps.geo]

What to do with the answer
--------------------------
* ``normal`` wins, residual <~ 1e-6  ->  the extraction is correct.  The 7x
  gap between SOLPS J_par and the g-file equilibrium current is genuine: SOLPS
  has no Ohmic drive, no bootstrap current and no Ampere constraint.  Use the
  *additive* scheme (``solps_jpar_to_bout_v2.py --mode add``).

* a ``swapped`` or ``offset`` variant wins  ->  real bug.  The core current may
  be 10-100x larger than plotted.  Report the winning variant and the stencil
  gets fixed.

* nothing conserves charge  ->  either the run is not converged, or ``fch``
  is only one piece of the total current (e.g. the diamagnetic / Pfirsch-
  Schlueter part is carried in a separate array, or the diamagnetic drift
  terms were switched off in ``b2mn.dat``).  The b2mn scan at the end of the
  output is there for exactly that case.

@author: ttaczak
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

QE = 1.602176634e-19
MP = 1.67262192e-27


# ===========================================================================
# flexible imports -- works inside the SIBIL package or as a loose script
# ===========================================================================
def _import_readers():
    """Return (B2fstate, B2fgmtry) however they happen to be installed."""
    patterns = (
        "sibil.readers.{}_Reader",
        "readers.{}_Reader",
        "{}_Reader",
        "{}",
    )
    out = {}
    for stem in ("B2fstate", "B2fgmtry"):
        mod = None
        for pat in patterns:
            try:
                mod = __import__(pat.format(stem), fromlist=[stem])
                break
            except ImportError:
                continue
        if mod is None:
            raise ImportError(
                "cannot import {}_Reader -- put it on PYTHONPATH or run this "
                "script from the directory containing the readers".format(stem))
        out[stem] = getattr(mod, stem)
    return out["B2fstate"], out["B2fgmtry"]


# ===========================================================================
# config handling
# ===========================================================================
def _flatten(d, out=None):
    """Flatten a nested dict so we can look up keys regardless of nesting."""
    if out is None:
        out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                _flatten(v, out)
            else:
                out[str(k).lower()] = v
    elif isinstance(d, list):
        for v in d:
            _flatten(v, out)
    return out


def _load_config(path):
    """Read a SIBIL-style YAML config and pull out the paths we need."""
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is needed to read the config file; "
                          "pass --b2fstate/--b2fgmtry explicitly instead")
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    flat = _flatten(raw)

    def pick(*names):
        for n in names:
            if n in flat and flat[n]:
                return str(flat[n])
        return None

    cfg = {
        "b2fstate": pick("b2fstate", "statefile", "b2fstate_file"),
        "b2fgmtry": pick("b2fgmtry", "gmtryfile", "b2fgmtry_file"),
        "geofile":  pick("geofile", "geo", "solps_geo"),
        "b2mn":     pick("b2mn", "b2mn.dat", "b2mndat"),
        "rundir":   pick("rundir", "run_dir", "solps_dir", "solpsdir", "path"),
    }

    # fill in from a run directory if individual files were not named
    rd = cfg["rundir"]
    if rd and os.path.isdir(rd):
        for key, fname in (("b2fstate", "b2fstate"),
                           ("b2fgmtry", "b2fgmtry"),
                           ("b2mn", "b2mn.dat")):
            if not cfg[key]:
                cand = os.path.join(rd, fname)
                if os.path.isfile(cand):
                    cfg[key] = cand
    return cfg


# ===========================================================================
# small helpers
# ===========================================================================
def _get(obj, *names):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    return None


def _hdr(title):
    print("")
    print("=" * 74)
    print(title)
    print("=" * 74)


def _fmt(x):
    if x is None:
        return "   n/a"
    if not np.isfinite(x):
        return "   nan"
    return "{:10.3e}".format(x)


# ===========================================================================
# the divergence test
# ===========================================================================
def divergence_variants(fch):
    """Net current out of every interior cell, for each candidate convention.

    ``fch`` is the native SOLPS array, shape (npol, nrad, 2).

    The four variants are

      normal        : fch[...,0] poloidal, stored on the *lower-ix* face
      swapped       : fch[...,0] radial,   stored on the *lower-index* face
      normal_offset : fch[...,0] poloidal, stored on the *upper-ix* face
      swapped_offset: fch[...,0] radial,   stored on the *upper-index* face

    Returns
    -------
    dict
        {name: (residual_array, scale_array)} on the interior cells only.
    """
    f0 = np.asarray(fch[..., 0], dtype=float)
    f1 = np.asarray(fch[..., 1], dtype=float)

    def net(fpol, frad, offset):
        """Net outflow on interior cells.

        offset=0 : value at index i is the flux through the LOWER face of i
                   -> out(i) = f(i+1) - f(i)
        offset=1 : value at index i is the flux through the UPPER face of i
                   -> out(i) = f(i) - f(i-1)
        """
        if offset == 0:
            dpol = fpol[2:, 1:-1] - fpol[1:-1, 1:-1]
            drad = frad[1:-1, 2:] - frad[1:-1, 1:-1]
            spol = np.abs(fpol[2:, 1:-1]) + np.abs(fpol[1:-1, 1:-1])
            srad = np.abs(frad[1:-1, 2:]) + np.abs(frad[1:-1, 1:-1])
        else:
            dpol = fpol[1:-1, 1:-1] - fpol[:-2, 1:-1]
            drad = frad[1:-1, 1:-1] - frad[1:-1, :-2]
            spol = np.abs(fpol[1:-1, 1:-1]) + np.abs(fpol[:-2, 1:-1])
            srad = np.abs(frad[1:-1, 1:-1]) + np.abs(frad[1:-1, :-2])
        return dpol + drad, spol + srad

    out = {}
    out["normal"] = net(f0, f1, 0)
    out["swapped"] = net(f1, f0, 0)
    out["normal_offset"] = net(f0, f1, 1)
    out["swapped_offset"] = net(f1, f0, 1)
    return out


def score(residual, scale):
    """Relative charge-conservation error: median and worst case."""
    s = np.where(scale > 0, scale, np.nan)
    rel = np.abs(residual) / s
    rel = rel[np.isfinite(rel)]
    if rel.size == 0:
        return np.nan, np.nan, np.nan
    return (np.nanmedian(rel),
            np.nanmean(rel),
            np.nanpercentile(rel, 99.0))


# ===========================================================================
# main
# ===========================================================================
def main(argv=None):
    ap = argparse.ArgumentParser(
        description="test SOLPS fch component/face/sign conventions against "
                    "steady-state charge conservation")
    ap.add_argument("config", nargs="?", default=None,
                    help="SIBIL YAML config (optional if paths are given)")
    ap.add_argument("--b2fstate", default=None)
    ap.add_argument("--b2fgmtry", default=None)
    ap.add_argument("--b2mn", default=None, help="b2mn.dat, for the flag scan")
    ap.add_argument("--geofile", default=None,
                    help="solps .geo file (used only for R, Bpol, Btot)")
    ap.add_argument("--ixsep", type=int, default=None,
                    help="radial index of the separatrix in SOLPS indexing "
                         "(only used to label the midplane report)")
    ap.add_argument("--amu", type=float, default=2.0,
                    help="main-ion mass in amu, for j_sat (default 2 = D)")
    args = ap.parse_args(argv)

    cfg = {"b2fstate": args.b2fstate, "b2fgmtry": args.b2fgmtry,
           "b2mn": args.b2mn, "geofile": args.geofile, "rundir": None}
    if args.config:
        fromfile = _load_config(args.config)
        for k, v in fromfile.items():
            if not cfg.get(k):
                cfg[k] = v

    if not cfg["b2fstate"] or not cfg["b2fgmtry"]:
        ap.error("need both b2fstate and b2fgmtry (via config or flags)")

    B2fstate, B2fgmtry = _import_readers()

    _hdr("input")
    print("  b2fstate : {}".format(cfg["b2fstate"]))
    print("  b2fgmtry : {}".format(cfg["b2fgmtry"]))
    if cfg["b2mn"]:
        print("  b2mn.dat : {}".format(cfg["b2mn"]))

    gmtry = B2fgmtry(cfg["b2fgmtry"])
    state = B2fstate(cfg["b2fstate"])

    fch = _get(state, "fch")
    if fch is None:
        print("\n!! b2fstate has no 'fch' array -- nothing to test")
        return 2
    fch = np.asarray(fch, dtype=float)
    npol, nrad = fch.shape[0], fch.shape[1]
    print("  fch shape: {}   (npol={}, nrad={}, ncomp={})"
          .format(fch.shape, npol, nrad, fch.shape[-1]))

    # ---------------------------------------------------------------- 1
    _hdr("1. raw fch content")
    for k in range(fch.shape[-1]):
        comp = fch[..., k]
        amax = np.nanmax(np.abs(comp))
        idx = np.unravel_index(np.nanargmax(np.abs(comp)), comp.shape)
        print("  fch[...,{}] : max|.| = {}  A   at (ix={}, iy={})"
              .format(k, _fmt(amax), idx[0], idx[1]))
        print("              nonzero fraction = {:.4f}"
              .format(float(np.mean(np.abs(comp) > 1e-12))))
    print("")
    print("  NOTE: if argmax of the *poloidal* component sits near a divertor")
    print("        target (ix near 0 or near {}), that is the expected")
    print("        signature of sheath/thermoelectric current closure and is")
    print("        consistent with a small midplane value."
          .format(npol - 1))

    # ---------------------------------------------------------------- 2
    _hdr("2. charge-conservation test  (the decisive one)")
    variants = divergence_variants(fch)
    rows = []
    for name, (res, scl) in variants.items():
        med, mean, p99 = score(res, scl)
        rows.append((name, med, mean, p99))
    rows.sort(key=lambda r: (np.inf if not np.isfinite(r[1]) else r[1]))

    print("  relative |div j| / |j_faces|   (smaller = better)")
    print("")
    print("    {:<16} {:>12} {:>12} {:>12}"
          .format("variant", "median", "mean", "99th pct"))
    print("    " + "-" * 54)
    for name, med, mean, p99 in rows:
        print("    {:<16} {:>12} {:>12} {:>12}"
              .format(name, _fmt(med), _fmt(mean), _fmt(p99)))

    best, best_med = rows[0][0], rows[0][1]
    runner_med = rows[1][1] if len(rows) > 1 else np.nan

    print("")
    if not np.isfinite(best_med):
        verdict = "inconclusive"
    elif best_med < 1e-6:
        verdict = "clean"
    elif best_med < 1e-3:
        verdict = "acceptable"
    else:
        verdict = "poor"

    print("  best variant : {}   (median rel. error {})"
          .format(best, _fmt(best_med)))
    if np.isfinite(runner_med) and runner_med > 0:
        print("  separation from runner-up: {:.1f}x".format(runner_med / max(best_med, 1e-300)))

    print("")
    if verdict == "clean" and best == "normal":
        print("  => VERDICT: the 'normal' convention conserves charge to")
        print("     round-off.  The existing extraction in")
        print("     solps_jpar_to_bout.py is CORRECT.  The small SOLPS")
        print("     parallel current is real physics, not a bug.  Use the")
        print("     additive scheme (solps_jpar_to_bout_v2.py --mode add).")
    elif verdict in ("clean", "acceptable") and best != "normal":
        print("  => VERDICT: the winning convention is '{}', NOT 'normal'."
              .format(best))
        print("     The extraction stencil is WRONG.  Re-run the mapper with")
        print("     --fch-convention {}".format(best))
        if "swapped" in best:
            print("     (component order is swapped: fch[...,0] is the RADIAL")
            print("      flux in this run, fch[...,1] the poloidal one)")
        if "offset" in best:
            print("     (stored values belong to the UPPER face, so the cell-")
            print("      centred average is 0.5*(f[i-1]+f[i]), not 0.5*(f[i]+f[i+1]))")
    else:
        print("  => VERDICT: no variant conserves charge well (best median")
        print("     {}).  Likely causes:".format(_fmt(best_med)))
        print("       * the run is not converged (check the residuals in b2fr*)")
        print("       * fch holds only part of the current -- the diamagnetic /")
        print("         Pfirsch-Schlueter piece may be in a separate array or")
        print("         switched off entirely (see the b2mn scan below)")
        print("       * there are volumetric charge sources in this run")

    # ---------------------------------------------------------------- 3
    _hdr("3. is there a dedicated parallel-current array?")
    found_any = False
    for name in ("fch_p", "fchp", "fchparallel", "jpar", "j_par",
                 "fch_dia", "fchdia", "fch_32", "fna_32"):
        v = _get(state, name)
        if v is not None:
            v = np.asarray(v, dtype=float)
            print("  {:<12} present, shape {}, max|.| = {}"
                  .format(name, v.shape, _fmt(np.nanmax(np.abs(v)))))
            found_any = True
    if not found_any:
        print("  none of fch_p / fchp / jpar / fch_dia found in b2fstate.")
        print("  -> the |B|/|B_pol| projection of the poloidal flux is the")
        print("     only route available, which is what the mapper does.")

    # ---------------------------------------------------------------- 4
    _hdr("4. cell-volume / face-area consistency")
    hx = _get(gmtry, "hx")
    hy = _get(gmtry, "hy")
    hz = _get(gmtry, "hz")
    vol = _get(gmtry, "vol")
    crx = _get(gmtry, "crx")

    if hx is None or hy is None:
        print("  hx/hy missing from b2fgmtry -- cannot check")
    else:
        hx = np.asarray(hx, dtype=float)
        hy = np.asarray(hy, dtype=float)
        print("  hx  : min {}  median {}  max {}"
              .format(_fmt(np.nanmin(hx)), _fmt(np.nanmedian(hx)),
                      _fmt(np.nanmax(hx))))
        print("  hy  : min {}  median {}  max {}"
              .format(_fmt(np.nanmin(hy)), _fmt(np.nanmedian(hy)),
                      _fmt(np.nanmax(hy))))
        if hz is not None:
            hz = np.asarray(hz, dtype=float)
            print("  hz  : min {}  median {}  max {}"
                  .format(_fmt(np.nanmin(hz)), _fmt(np.nanmedian(hz)),
                          _fmt(np.nanmax(hz))))

        # R at cell centres, for the 2*pi*R comparison
        Rc = None
        if crx is not None:
            crx = np.asarray(crx, dtype=float)
            if crx.ndim == 3 and crx.shape[-1] >= 4:
                Rc = crx[..., :4].mean(axis=-1)
            elif crx.ndim == 3 and crx.shape[0] >= 4:
                Rc = crx[:4, ...].mean(axis=0)

        if vol is not None:
            vol = np.asarray(vol, dtype=float)
            print("")
            print("  vol : min {}  median {}  max {}"
                  .format(_fmt(np.nanmin(vol)), _fmt(np.nanmedian(vol)),
                          _fmt(np.nanmax(vol))))
            if hz is not None:
                r1 = np.nanmedian(vol / (hx * hy * hz))
                print("    median vol / (hx*hy*hz)      = {:8.4f}".format(r1))
            if Rc is not None and Rc.shape == hx.shape:
                r2 = np.nanmedian(vol / (hx * hy * 2.0 * np.pi * Rc))
                print("    median vol / (hx*hy*2*pi*R)  = {:8.4f}".format(r2))
                print("")
                print("    both ratios should be ~1.  A value near 1/(2*pi)")
                print("    = 0.159 means vol is stored PER RADIAN and the")
                print("    poloidal face area is wrong by 2*pi (-> J_par too")
                print("    large by 6.28, i.e. the wrong direction to explain")
                print("    a small current).")

    # ---------------------------------------------------------------- 5
    _hdr("5. physics scale check:  j_par / j_sat")
    ne = _get(state, "ne")
    te = _get(state, "te")
    if ne is None or te is None:
        print("  ne/te unavailable -- skipping")
    else:
        ne = np.asarray(ne, dtype=float)
        te = np.asarray(te, dtype=float)
        # SOLPS stores T in Joules
        te_eV = te / QE if np.nanmax(te) < 1e-6 else te
        cs = np.sqrt(np.clip(te_eV, 0.0, None) * QE / (args.amu * MP))
        jsat = QE * np.clip(ne, 0.0, None) * cs

        # poloidal component according to the winning variant
        ipol = 1 if "swapped" in best else 0
        fpol = fch[..., ipol]
        off = 1 if "offset" in best else 0
        Ipol = np.full_like(fpol, np.nan)
        if off == 0:
            Ipol[:-1, :] = 0.5 * (fpol[:-1, :] + fpol[1:, :])
        else:
            Ipol[1:, :] = 0.5 * (fpol[:-1, :] + fpol[1:, :])

        # poloidal face area
        Apol = None
        if vol is not None and hx is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                Apol = vol / hx
        elif hy is not None and hz is not None:
            Apol = hy * hz
        if Apol is None:
            print("  cannot form the poloidal face area -- skipping")
        else:
            with np.errstate(divide="ignore", invalid="ignore"):
                Jpol = Ipol / Apol

            # |B| / |B_pol|
            fac = None
            bb = _get(gmtry, "bb")
            if bb is not None:
                bb = np.asarray(bb, dtype=float)
                if bb.ndim == 3 and bb.shape[-1] >= 4:
                    mags = [np.nanmedian(np.abs(bb[..., k]))
                            for k in range(bb.shape[-1])]
                    itot = int(np.nanargmax(mags))
                    ipolb = int(np.nanargmin([m if k != itot else np.inf
                                              for k, m in enumerate(mags)]))
                    Btot = np.abs(bb[..., itot])
                    Bpol = np.abs(bb[..., ipolb])
                    with np.errstate(divide="ignore", invalid="ignore"):
                        fac = Btot / Bpol
                    print("  bb components: using index {} as |B| and {} as "
                          "|B_pol|".format(itot, ipolb))
            if fac is None:
                print("  bb unusable; assuming |B|/|B_pol| = 10 for this estimate")
                fac = np.full_like(Jpol, 10.0)

            Jpar = Jpol * fac
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.abs(Jpar) / jsat

            print("")
            print("  max |J_par|            = {} A/m^2".format(_fmt(np.nanmax(np.abs(Jpar)))))
            print("  max  j_sat             = {} A/m^2".format(_fmt(np.nanmax(jsat))))
            print("  max  |J_par| / j_sat   = {}".format(_fmt(np.nanmax(ratio))))
            print("  median |J_par| / j_sat = {}".format(_fmt(np.nanmedian(ratio))))

            # near-target rows
            for label, ix in (("inner target region", 1),
                              ("outer target region", npol - 2)):
                r = ratio[ix, :]
                print("  {:<22}: max |J_par|/j_sat = {}"
                      .format(label, _fmt(np.nanmax(r))))

            print("")
            print("  Expected for edge thermoelectric / sheath currents:")
            print("    |j_par| / j_sat  ~  0.1 - 0.5  near the targets.")
            print("  If you see that, the extraction is sound and the SOLPS")
            print("  current really is ~0.05 MA/m^2, i.e. ~10x below the")
            print("  0.4 MA/m^2 equilibrium current from the g-file.")
            print("  If |j_par|/j_sat << 0.01 everywhere, something IS being")
            print("  lost -- most likely face cancellation (see section 2) or")
            print("  a missing diamagnetic contribution (section 6).")

    # ---------------------------------------------------------------- 6
    _hdr("6. b2mn.dat scan: are the current / drift terms switched on?")
    b2mn = cfg.get("b2mn")
    if not b2mn or not os.path.isfile(b2mn):
        print("  b2mn.dat not supplied (--b2mn) -- skipping.")
        print("  Worth checking by hand for the current and drift switches:")
        print("    b2mndr_*  , b2tqna_* , b2sigp_* , b2cd* , *dia* , *pot*")
    else:
        keys = ("pot", "curr", "cur", "drift", "dia", "b2cd", "sigp",
                "jterm", "fch", "b2tqna", "b2mndr", "b2stbc", "ebfield",
                "bcpot", "ezhg", "ehxy")
        pat = re.compile("|".join(keys), re.IGNORECASE)
        hits = []
        with open(b2mn, "r", errors="replace") as fh:
            for ln in fh:
                s = ln.strip()
                if not s or s.startswith("#"):
                    continue
                if pat.search(s):
                    hits.append(s)
        if hits:
            print("  lines mentioning current / potential / drift terms:")
            print("")
            for s in hits:
                print("    " + s)
            print("")
            print("  Check in particular:")
            print("    * the potential equation must be SOLVED (not held fixed)")
            print("    * diamagnetic drift terms ON, otherwise the Pfirsch-")
            print("      Schlueter current is absent -- and PS current IS")
            print("      comparable to the equilibrium current near the")
            print("      separatrix, which would be a third explanation for")
            print("      the small values you are seeing.")
        else:
            print("  no current/potential/drift switches found in b2mn.dat.")
            print("  SOLPS defaults then apply -- which in many builds means")
            print("  drifts and currents are OFF.  Given that fch is 98%")
            print("  nonzero here, the potential equation evidently IS being")
            print("  solved, but the diamagnetic terms may still be off.")

    _hdr("summary")
    print("  winning fch convention : {}".format(best))
    print("  conservation quality   : {}  (median rel. error {})"
          .format(verdict, _fmt(best_med)))
    print("")
    print("  next step:")
    if best == "normal" and verdict in ("clean", "acceptable"):
        print("    python solps_jpar_to_bout_v2.py <config> --mode add --plot")
    else:
        print("    python solps_jpar_to_bout_v2.py <config> --mode add "
              "--fch-convention {} --plot".format(best))
    print("")
    return 0


if __name__ == "__main__":
    sys.exit(main())
