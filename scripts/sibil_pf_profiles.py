#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sibil_pf_profiles.py

Replace the private-flux (PF) part of the equilibrium profiles in a BOUT++
elm-6f grid, and/or check that the grid and the SOLPS-ITER mesh agree on
where the separatrix is.

Why
---
Niexp / Teexp / Tiexp are flux functions mapped from the outer midplane, so
the PF cells (ix < ixseps1 in the divertor-leg columns) carry the full core
pedestal gradient (psi_n 0.95-0.99) directly under the X-point.  Physically
the PF there is cold and flat.  This script swaps the PF part for either

  flat   the separatrix value (times --pf-frac),
  flux   the outer-midplane profile at the same psi_n (a true flux function;
         the original PF values are the OMP values at the same ix), or
  solps  the SOLPS-ITER PF profile, binned in psi_n and scaled so the SOLPS
         outer-midplane separatrix value matches the grid's (units cancel),

blended from the separatrix value with a quintic smoothstep over --width in
psi_n inside the separatrix, so the PF profile is C2 and still equals the original at
psi_n = 1 (continuous with the SOL across the separatrix).  Core and SOL
cells are not touched.  Pi0 = N0*Ti0 and Pe0 = Ne0*Te0 are built from these
arrays by the model; 'pressure'/'pressure_s' are rescaled to match in the PF.

Originals are kept as <name>_orig the first time, and always read back from
there, so re-running is idempotent.

Usage
-----
  # quick test of the hypothesis, current grid
  python sibil_pf_profiles.py --grid g.nc --gfile g184016.03200 --out g_pfflat.nc

  # SOLPS PF profile
  python sibil_pf_profiles.py --grid g.nc --gfile gfile --solps <SOLPS run dir> \
      --mode solps --out g_pfsolps.nc --plot pf.png

  # separatrix consistency only (writes nothing)
  python sibil_pf_profiles.py --grid g.nc --gfile gfile --solps <dir> --check-sep

Needs the same helpers as sibil_solps_jpar_to_bout.py (it imports them):
SOLPS readers, gqdsk_compat.py, boutgrid_compat.py, boututils.

@author: ttaczak
"""

import argparse
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sibil_solps_jpar_to_bout import (BoutGrid, _find, _get, _hdr,   # noqa: E402
                                      _import_datafile, _import_eq_resolver,
                                      _import_readers, smoothstep5)

FIELDS = ("Niexp", "Neexp", "Teexp", "Tiexp")
SOLPS_KEY = {"Niexp": "ne", "Neexp": "ne", "Teexp": "te", "Tiexp": "ti"}


# ===========================================================================
# geometry helpers
# ===========================================================================
def omp_column(grid):
    core_j = np.flatnonzero(~grid.leg_columns())
    R = np.asarray(grid.data["Rxy"], float)
    return int(core_j[np.argmax(R[grid.ixseps1, core_j])])


def xpoint_Z(grid):
    Z = np.asarray(grid.data["Zxy"], float)
    i, j = grid.ixseps1, grid.jyseps1_1
    return float(Z[i - 1:i + 1, j:j + 2].mean())


def sep_value(grid, f, jomp):
    """f at psi_n = 1 on the outer-midplane column (core + SOL, monotone)."""
    p = grid.psinxy[:, jomp]
    o = np.argsort(p)
    return float(np.interp(1.0, p[o], f[o, jomp]))


# ===========================================================================
# SOLPS side
# ===========================================================================
class Solps(object):
    def __init__(self, statefile, gmtryfile, eq):
        B2fstate, B2fgmtry, _ = _import_readers()
        self.gm = B2fgmtry(gmtryfile)
        self.st = B2fstate(statefile) if statefile else None
        crx = np.asarray(_get(self.gm, "crx"), float)
        cry = np.asarray(_get(self.gm, "cry"), float)
        te = _get(self.st, "te") if self.st is not None else None
        if te is not None:
            npol, nrad = np.asarray(te).shape[:2]
        else:
            npol, nrad = np.asarray(_get(self.gm, "vol")).shape[:2]
        self.crx = crx.reshape(npol, nrad, -1)
        self.cry = cry.reshape(npol, nrad, -1)
        self.Rc, self.Zc = self.crx.mean(2), self.cry.mean(2)
        self.psin = np.asarray(eq["psin_of_RZ"](self.Rc, self.Zc), float)
        self.psin_c = np.asarray(eq["psin_of_RZ"](self.crx, self.cry), float)
        zax = float(eq.get("zmaxis", 0.0))
        R, Z = self.Rc[:, 1], self.Zc[:, 1]
        ok = np.isfinite(R) & (np.abs(Z - zax) < 0.5)
        self.jomp = int(np.flatnonzero(ok)[np.argmax(R[ok])])

    def field(self, key):
        v = _get(self.st, key, key.capitalize())
        if v is None:
            raise KeyError("b2fstate has no '{}'".format(key))
        return np.asarray(v, float)

    def omp_sep(self, f):
        p, v = self.psin[self.jomp, 1:-1], f[self.jomp, 1:-1]
        o = np.argsort(p)
        return float(np.interp(1.0, p[o], v[o]))

    def pf_profile(self, f, zx, nbins=24):
        """Median of f over SOLPS PF cells, binned in psi_n."""
        p, z, v = self.psin[1:-1, 1:-1], self.Zc[1:-1, 1:-1], f[1:-1, 1:-1]
        m = (p < 1.0) & (z < zx) & np.isfinite(v)
        p, v = p[m], v[m]
        edges = np.linspace(p.min(), 1.0, nbins + 1)
        k = np.clip(np.digitize(p, edges) - 1, 0, nbins - 1)
        pc, vc = [], []
        for b in range(nbins):
            s = k == b
            if s.sum() >= 2:
                pc.append(p[s].mean())
                vc.append(np.median(v[s]))
        return np.array(pc), np.array(vc), (p, v)


# ===========================================================================
# PF replacement
# ===========================================================================
def new_pf(f, grid, target, width, fsep):
    """(1-S) fsep + S target in PF cells, S = smoothstep5((1 - psi_n)/width).

    Blends from the separatrix value, not from the original PF profile: the
    original PF values are index-mapped (the OMP value at the same ix), not a
    flux function, so mixing them in puts humps in the PF profile.
    """
    _, _, pf = grid.regions()
    S = smoothstep5((1.0 - grid.psinxy) / width)
    out = f.copy()
    out[pf] = ((1.0 - S) * fsep + S * target)[pf]
    return out


def build(grid, orig, args, solps=None):
    jomp = omp_column(grid)
    zx = xpoint_Z(grid) if solps is not None else None
    new, info = {}, {}
    for name, f in orig.items():
        fsep = sep_value(grid, f, jomp)
        if args.mode == "flat":
            target = np.full_like(f, args.pf_frac * fsep)
            info[name] = (fsep, None)
        elif args.mode == "flux":
            # true flux function: PF value = OMP value at the same psi_n
            p = grid.psinxy[:, jomp]
            o = np.argsort(p)
            new[name] = orig[name].copy()
            _, _, pf = grid.regions()
            new[name][pf] = np.interp(grid.psinxy, p[o], f[o, jomp])[pf]
            info[name] = (fsep, None)
            continue
        else:
            sv = solps.field(SOLPS_KEY[name])
            scale = fsep / solps.omp_sep(sv)
            pc, vc, raw = solps.pf_profile(sv, zx)
            target = np.interp(grid.psinxy, pc, vc * scale)
            info[name] = (fsep, (scale, pc, vc * scale, raw[0], raw[1] * scale))
        new[name] = np.maximum(new_pf(f, grid, target, args.width, fsep), 1e-3 * fsep)
    return new, info


def report(grid, orig, new, info, args):
    _hdr("PF profiles  (mode = {}, width = {:g} psi_n{})".format(
        args.mode, args.width,
        ", pf_frac = {:g}".format(args.pf_frac) if args.mode == "flat" else ""))
    _, _, pf = grid.regions()
    for name in orig:
        fsep, sol = info[name]
        o, n = orig[name][pf], new[name][pf]
        print("  {:<6} sep {:10.4g}   PF old {:10.4g} .. {:10.4g}   PF new {:10.4g} .. {:10.4g}"
              .format(name, fsep, o.min(), o.max(), n.min(), n.max()))
        if sol is not None:
            print("         SOLPS scale (grid/SOLPS at OMP sep) = {:.4g}".format(sol[0]))
    # PF gradient before/after, in cells per gradient length
    j = grid.jyseps1_1
    for name in ("Teexp", "Niexp"):
        if name in orig:
            for lab, F in (("old", orig[name]), ("new", new[name])):
                c = F[:grid.ixseps1, j]
                L = np.abs(c[1:-1] / (0.5 * (c[2:] - c[:-2]) + 1e-300))
                print("  {:<6} {} min L/dx in PF column j={}: {} cells{}".format(
                    name, lab, j, "{:.3g}".format(L.min()) if L.min() < 1e6 else "flat", "   <-- steep: increase --width" if lab == "new" and L.min() < 10 else ""))


def plot(grid, orig, new, info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jomp, jpf = omp_column(grid), grid.jyseps1_1
    isep = grid.ixseps1
    fig, axs = plt.subplots(1, len(orig), figsize=(4.2 * len(orig), 3.6))
    for ax, name in zip(np.atleast_1d(axs), orig):
        p = grid.psinxy
        ax.plot(p[:, jomp], orig[name][:, jomp], "k-", lw=1, label="OMP column (core+SOL)")
        ax.plot(p[:isep, jpf], orig[name][:isep, jpf], "C3--", label="PF old (j={})".format(jpf))
        ax.plot(p[:isep, jpf], new[name][:isep, jpf], "C0-", lw=2, label="PF new")
        sol = info[name][1]
        if sol is not None:
            ax.plot(sol[3], sol[4], ".", ms=2, color="0.6", label="SOLPS PF cells (scaled)")
            ax.plot(sol[1], sol[2], "o", ms=3, color="C2", label="SOLPS binned")
        ax.axvline(1.0, color="0.5", lw=0.5)
        ax.set_xlabel(r"$\psi_n$")
        ax.set_title(name)
    np.atleast_1d(axs)[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print("  plot written: {}".format(path))


# ===========================================================================
# separatrix consistency
# ===========================================================================
def sep_check(grid, eq, solps=None):
    _hdr("separatrix consistency  (everything measured against the g-file psi_n)")
    psin_g = eq["psin_of_RZ"]
    R = np.asarray(grid.data["Rxy"], float)
    Z = np.asarray(grid.data["Zxy"], float)
    i = grid.ixseps1
    core_j = np.flatnonzero(~grid.leg_columns())
    jomp = omp_column(grid)

    pg = np.asarray(psin_g(R, Z), float)
    print("  grid psi_n vs g-file psi_n at Rxy,Zxy: max |diff| {:.2e}  (core+SOL cells)"
          .format(np.nanmax(np.abs(grid.psinxy - pg)[:, core_j])))

    # dpsi_n/dR at the OMP separatrix, for converting to mm
    dpdr = (pg[i, jomp] - pg[i - 1, jomp]) / (R[i, jomp] - R[i - 1, jomp])
    dr_cell = abs(R[i, jomp] - R[i - 1, jomp]) * 1e3

    # BOUT: hypnotoad's separatrix lies half way between ix = isep-1 and isep
    Rm = 0.5 * (R[i - 1] + R[i])
    Zm = 0.5 * (Z[i - 1] + Z[i])
    pb = np.asarray(psin_g(Rm, Zm), float)[core_j]
    k = np.argmax(np.abs(pb - 1))
    print("  BOUT  separatrix (mid ix {}/{}):  psi_n at OMP {:.6f} ({:+.3f} mm, {:+.2f} cells);"
          "  worst core column j={} {:.6f}"
          .format(i - 1, i, pb[list(core_j).index(jomp)],
                  (pb[list(core_j).index(jomp)] - 1) / dpdr * 1e3,
                  (pb[list(core_j).index(jomp)] - 1) / dpdr * 1e3 / dr_cell,
                  core_j[k], pb[k]))
    print("        OMP radial cell at the separatrix: {:.3f} mm".format(dr_cell))

    if solps is None:
        return
    # SOLPS: Carre puts cell corners exactly on its separatrix -> the corner
    # closest to psi_n = 1 in each poloidal column is the SOLPS separatrix
    zx = xpoint_Z(grid)
    dev = solps.psin_c - 1.0
    best = np.nanmin(np.abs(dev), axis=(1, 2))
    above = solps.Zc[:, 1] > zx
    jo = solps.jomp
    r, c = np.unravel_index(np.nanargmin(np.abs(dev[jo])), dev[jo].shape)
    p_omp = solps.psin_c[jo, r, c]
    print("  SOLPS separatrix (nearest cell corner): psi_n at OMP {:.6f} ({:+.3f} mm, {:+.2f} BOUT cells);"
          "  worst core column {:.2e}"
          .format(p_omp, (p_omp - 1) / dpdr * 1e3, (p_omp - 1) / dpdr * 1e3 / dr_cell,
                  np.nanmax(best[above])))
    print("  offsets of more than ~1 BOUT cell mean the two codes disagree on the separatrix"
          "\n  (different g-file, or Carre/hypnotoad X-point psi) -- fix before mapping profiles")


# ===========================================================================
# CLI
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--grid", required=True, help="BOUT++ grid to read")
    p.add_argument("--gfile", required=True, help="EFIT g-file used for the grid and SOLPS")
    p.add_argument("--out", help="write here (copy of --grid); omit with --check-sep/--dry-run")
    p.add_argument("--mode", choices=("flat", "flux", "solps"), default="flat")
    p.add_argument("--width", type=float, default=0.005,
                   help="psi_n width inside the separatrix over which PF goes from original to target")
    p.add_argument("--pf-frac", type=float, default=1.0,
                   help="flat mode: PF target = pf_frac * separatrix value")
    p.add_argument("--fields", nargs="+", default=list(FIELDS))
    p.add_argument("--solps", help="SOLPS-ITER run dir (b2fstate; b2fgmtry here or in baserun/)")
    p.add_argument("--b2fstate")
    p.add_argument("--b2fgmtry")
    p.add_argument("--check-sep", action="store_true", help="separatrix check only, write nothing")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--plot", help="png of old/new PF profiles")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    if a.solps:
        d = os.path.abspath(a.solps)
        dirs = [d, os.path.join(d, "baserun"), os.path.join(os.path.dirname(d), "baserun")]
        a.b2fstate = a.b2fstate or _find(("b2fstate", "b2fstati"), dirs)
        a.b2fgmtry = a.b2fgmtry or _find(("b2fgmtry",), dirs)
    if a.mode == "solps" and not (a.b2fstate and a.b2fgmtry):
        raise SystemExit("--mode solps needs b2fstate and b2fgmtry (--solps DIR)")

    _, _, GQDSK = _import_readers()
    eq = _import_eq_resolver()(GQDSK(a.gfile), verbose=False)
    grid = BoutGrid(a.grid, geq=eq, verbose=False)
    solps = Solps(a.b2fstate, a.b2fgmtry, eq) if a.b2fgmtry else None

    sep_check(grid, eq, solps)
    if a.check_sep:
        return 0

    orig = {}
    for n in a.fields:
        src = n + "_orig" if n + "_orig" in grid.data else n
        if src in grid.data:
            orig[n] = np.asarray(grid.data[src], float)
    new, info = build(grid, orig, a, solps)
    report(grid, orig, new, info, a)
    if a.plot:
        plot(grid, orig, new, info, a.plot)
    if a.dry_run or not a.out:
        print("\n  nothing written ({})".format("--dry-run" if a.dry_run else "no --out"))
        return 0

    shutil.copy2(a.grid, a.out)
    DataFile = _import_datafile()
    with DataFile(a.out, write=True) as f:
        for n in new:
            if n + "_orig" not in grid.data:
                f.write(n + "_orig", orig[n])
            f.write(n, new[n])
        # keep P0 consistent with N(Te+Ti) in the PF
        if all(k in new for k in ("Niexp", "Teexp", "Tiexp")):
            ratio = (new["Niexp"] * (new["Teexp"] + new["Tiexp"])) / \
                    (orig["Niexp"] * (orig["Teexp"] + orig["Tiexp"]))
            for pk in ("pressure", "pressure_s"):
                src = pk + "_orig" if pk + "_orig" in grid.data else pk
                if src in grid.data:
                    p0 = np.asarray(grid.data[src], float)
                    if pk + "_orig" not in grid.data:
                        f.write(pk + "_orig", p0)
                    f.write(pk, p0 * ratio)
    print("\n  wrote {} ({})".format(a.out, ", ".join(new)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
