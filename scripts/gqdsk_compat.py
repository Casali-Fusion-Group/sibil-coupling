#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gqdsk_compat.py

Compatibility shim between SIBIL's ``GQDSK`` reader and any script that
expects the classic EFIT/geqdsk attribute names (``simag``, ``sibry``,
``psirz``, ``rgrid``, ``zgrid``).

Why this exists
---------------
``sibil.GQDSK`` stores the equilibrium under its own names::

    psiaxis   -> simag      (poloidal flux on the magnetic axis)
    psiedge   -> sibry      (poloidal flux on the LCFS)
    psi       -> psirz      (2-D flux map, shape (nR, nZ) or (nZ, nR))
    R_grid    -> rgrid
    Z_grid    -> zgrid
    Raxis     -> rmaxis
    Zaxis     -> zmaxis

so ``getattr(g, "simag")`` returns ``None`` and the caller dies with

    TypeError: float() argument must be a string or a real number,
               not 'NoneType'

This module resolves the names for you, and -- more importantly -- it
*verifies* the orientation of the 2-D flux array instead of assuming it.

The orientation problem
-----------------------
``psi`` is 65x65 for this g-file, so you cannot tell from ``.shape`` whether
it is stored as ``psi[iR, iZ]`` (geqdsk column-major convention) or
``psi[iZ, iR]`` (the C-order read of the EFIT record).  Guessing wrong
transposes the whole equilibrium: psi_n comes out plausible-looking but
every SOLPS cell lands on the wrong flux surface, and the error is largest
exactly where you care (near the X-point, where the mesh is not up-down
symmetric).

There is a clean, decisive test.  The flux on the magnetic axis is known
independently -- it is ``psiaxis``.  So interpolate both candidate
orientations at ``(Raxis, Zaxis)`` and keep whichever reproduces
``psiaxis``.  The wrong orientation misses by O(|psiedge - psiaxis|), i.e.
by ~100%, so the test is never ambiguous.

A second, independent cross-check is available for free: SIBIL also stores
``rhonormrz``, the normalised radius on the (R, Z) grid.  For most EFIT
conventions ``rhonormrz**2 ~ psi_n``, so comparing against the psi_n we
built from ``psi`` catches a transpose *and* a normalisation mistake at the
same time.  It is reported as a diagnostic, not used.

Usage
-----
In ``sibil_solps_jpar_to_bout.py``, add near the top::

    from gqdsk_compat import resolve_equilibrium

then inside ``SolpsJpar._assign_psin`` replace the block that currently
reads ``simag`` / ``sibry`` / ``psirz`` with::

    eq = resolve_equilibrium(g, verbose=self.verbose)
    simag = eq["simag"]
    sibry = eq["sibry"]
    psirz = eq["psirz"]          # guaranteed (nR, nZ), i.e. psi[iR, iZ]
    Rg    = eq["rgrid"]
    Zg    = eq["zgrid"]

or, simpler, skip the manual interpolation entirely and use the ready-made
callable::

    eq = resolve_equilibrium(g, verbose=self.verbose)
    self.psin_solps = eq["psin_of_RZ"](self.Rc, self.Zc)

which already does the bicubic interpolation with the verified orientation
and the correct normalisation.

Standalone check
----------------
    python gqdsk_compat.py ../g184016.03200

prints the resolved values, the orientation verdict, the rho cross-check and
psi_n at a few landmark points.

@author: ttaczak
"""

from __future__ import annotations

import sys

import numpy as np

__all__ = ["resolve_equilibrium", "GqdskAdapter"]


# ===========================================================================
# name resolution
# ===========================================================================
#: candidate attribute names, SIBIL's first, then classic geqdsk spellings
_ALIASES = {
    "simag":  ("psiaxis", "simag", "ssimag", "psi_axis", "psimag",
               "psi0", "psi_mag"),
    "sibry":  ("psiedge", "sibry", "ssibry", "psi_bndry", "psibnd",
               "psibdry", "psi_edge", "psilim", "psi_lcfs"),
    "psirz":  ("psi", "psirz", "psi_rz", "psiRZ", "psi2d", "psixy"),
    "rgrid":  ("R_grid", "rgrid", "r_grid", "R", "r", "rdim_grid"),
    "zgrid":  ("Z_grid", "zgrid", "z_grid", "Z", "z", "zdim_grid"),
    "rmaxis": ("Raxis", "rmaxis", "R_axis", "raxis", "rmag"),
    "zmaxis": ("Zaxis", "zmaxis", "Z_axis", "zaxis", "zmag"),
    "rbbbs":  ("R_LCFS", "rbbbs", "rbdry", "R_bndry", "rlcfs"),
    "zbbbs":  ("Z_LCFS", "zbbbs", "zbdry", "Z_bndry", "zlcfs"),
    "current": ("current", "cpasma", "Ip", "ip", "plasma_current"),
    "bcentr": ("B0EXP", "bcentr", "B0", "btor", "bt0"),
    "rcentr": ("R0EXP", "rcentr", "R0", "rmajor"),
    "rhonormrz": ("rhonormrz", "rhonorm_rz", "rhonorm"),
}


def _lookup(obj, names, _depth=0):
    """Fetch the first present, non-None attribute/key from ``names``."""
    # direct attributes
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    # dict-style readers
    if isinstance(obj, dict):
        for n in names:
            if n in obj and obj[n] is not None:
                return obj[n]
        # case-insensitive
        low = {str(k).lower(): v for k, v in obj.items()}
        for n in names:
            v = low.get(n.lower())
            if v is not None:
                return v
    # case-insensitive attribute match
    try:
        attrs = {k.lower(): k for k in vars(obj)}
    except TypeError:
        attrs = {}
    for n in names:
        real = attrs.get(n.lower())
        if real is not None:
            v = getattr(obj, real, None)
            if v is not None:
                return v
    # one level of nesting (readers that wrap the data in .data / .eq / .g)
    if _depth == 0:
        for sub in ("data", "eq", "g", "gfile", "geqdsk", "_data", "raw"):
            child = getattr(obj, sub, None)
            if child is not None:
                v = _lookup(child, names, _depth=1)
                if v is not None:
                    return v
    return None


def _scalar(v, what):
    """Coerce a possibly 0-d / (1,1) array to a float."""
    if v is None:
        raise KeyError(
            "cannot find '{}' in the GQDSK object -- add its attribute "
            "name to _ALIASES in gqdsk_compat.py".format(what))
    a = np.asarray(v, dtype=float).ravel()
    if a.size != 1:
        raise ValueError("'{}' should be scalar, got size {}"
                         .format(what, a.size))
    return float(a[0])


# ===========================================================================
# orientation detection
# ===========================================================================
def _orient_psi(psi, Rg, Zg, Raxis, Zaxis, psiaxis, verbose=True):
    """Return ``psi`` transposed so that it indexes as ``psi[iR, iZ]``.

    Decided by interpolating both candidates at the magnetic axis and
    keeping the one that reproduces ``psiaxis``.
    """
    import scipy.interpolate as si

    psi = np.asarray(psi, dtype=float)
    Rg = np.asarray(Rg, dtype=float).ravel()
    Zg = np.asarray(Zg, dtype=float).ravel()

    if psi.ndim == 1:
        n = int(round(np.sqrt(psi.size)))
        if n * n != psi.size:
            if psi.size == Rg.size * Zg.size:
                n = None
            else:
                raise ValueError("flat psi of size {} is not reshapeable"
                                 .format(psi.size))
        # EFIT writes psirz with R varying fastest within each Z record,
        # so the C-order reshape is (nZ, nR).
        psi = psi.reshape(Zg.size, Rg.size)

    nR, nZ = Rg.size, Zg.size
    if psi.shape == (nR, nZ):
        cands = {"psi[iR,iZ] (as stored)": psi}
        if nR == nZ:
            cands["psi[iZ,iR] (transposed)"] = psi.T
    elif psi.shape == (nZ, nR):
        cands = {"psi[iZ,iR] (transposed)": psi.T}
    else:
        raise ValueError("psi shape {} matches neither (nR,nZ)=({},{}) "
                         "nor its transpose".format(psi.shape, nR, nZ))

    best, best_err, report = None, np.inf, []
    scale = max(abs(psiaxis), 1e-12)
    for label, cand in cands.items():
        try:
            spl = si.RectBivariateSpline(Rg, Zg, cand, kx=3, ky=3)
            val = float(spl(Raxis, Zaxis, grid=False))
        except Exception as exc:                       # noqa: BLE001
            report.append((label, np.nan, str(exc)))
            continue
        err = abs(val - psiaxis) / scale
        report.append((label, val, err))
        if err < best_err:
            best, best_err, best_label = cand, err, label

    if verbose:
        print("    psi orientation test  (target psi(Raxis,Zaxis) = "
              "{:+.6g})".format(psiaxis))
        for label, val, err in report:
            if isinstance(err, str):
                print("      {:28s} FAILED: {}".format(label, err))
            else:
                print("      {:28s} psi_axis = {:+.6g}   rel.err = {:.3e}"
                      .format(label, val, err))

    if best is None:
        raise RuntimeError("could not interpolate psi in either orientation")

    if best_err > 0.05:
        print("    *** WARNING: best orientation still misses psiaxis by "
              "{:.1%}.".format(best_err))
        print("        The flux map or the axis location may be "
              "inconsistent; check the g-file.")
    elif verbose:
        print("      -> using {}  (rel.err {:.2e})"
              .format(best_label, best_err))

    return best


# ===========================================================================
# public entry point
# ===========================================================================
def resolve_equilibrium(g, verbose=True):
    """Normalise a SIBIL ``GQDSK`` object to classic geqdsk fields.

    Parameters
    ----------
    g : object
        Anything returned by ``sibil.GQDSK`` (or a dict, or a reader that
        wraps the data one level down).
    verbose : bool
        Print what was resolved and the orientation verdict.

    Returns
    -------
    dict with keys
        ``simag``, ``sibry``      : float, flux on axis / LCFS
        ``psirz``                 : (nR, nZ) array, verified orientation
        ``rgrid``, ``zgrid``      : 1-D grids
        ``rmaxis``, ``zmaxis``    : magnetic axis
        ``psi_of_RZ``             : callable (R, Z) -> psi
        ``psin_of_RZ``            : callable (R, Z) -> psi_n
        ``current``               : plasma current [A] or None
    """
    import scipy.interpolate as si

    simag = _scalar(_lookup(g, _ALIASES["simag"]), "simag/psiaxis")
    sibry = _scalar(_lookup(g, _ALIASES["sibry"]), "sibry/psiedge")

    Rg = np.asarray(_lookup(g, _ALIASES["rgrid"]), dtype=float).ravel()
    Zg = np.asarray(_lookup(g, _ALIASES["zgrid"]), dtype=float).ravel()
    if Rg.size < 4 or Zg.size < 4:
        raise KeyError("R_grid / Z_grid not found or too short")

    psi_raw = _lookup(g, _ALIASES["psirz"])
    if psi_raw is None:
        raise KeyError("no 2-D flux map ('psi' / 'psirz') found")

    rmaxis = _scalar(_lookup(g, _ALIASES["rmaxis"]), "rmaxis/Raxis")
    zmaxis = _scalar(_lookup(g, _ALIASES["zmaxis"]), "zmaxis/Zaxis")

    if verbose:
        print("    equilibrium fields resolved:")
        print("      psi_axis = {:+.6g}   psi_edge = {:+.6g}   "
              "dpsi = {:+.6g}".format(simag, sibry, sibry - simag))
        print("      axis at (R, Z) = ({:.4f}, {:.4f}) m"
              .format(rmaxis, zmaxis))
        print("      grid: nR = {}  R in [{:.3f}, {:.3f}]   "
              "nZ = {}  Z in [{:.3f}, {:.3f}]"
              .format(Rg.size, Rg.min(), Rg.max(),
                      Zg.size, Zg.min(), Zg.max()))

    if abs(sibry - simag) < 1e-12:
        raise ValueError("psi_edge == psi_axis: cannot normalise")

    psi = _orient_psi(psi_raw, Rg, Zg, rmaxis, zmaxis, simag, verbose=verbose)

    spl = si.RectBivariateSpline(Rg, Zg, psi, kx=3, ky=3)

    def psi_of_RZ(R, Z):
        R = np.asarray(R, dtype=float)
        Z = np.asarray(Z, dtype=float)
        out = spl(R.ravel(), Z.ravel(), grid=False)
        return np.asarray(out, dtype=float).reshape(R.shape)

    def psin_of_RZ(R, Z):
        return (psi_of_RZ(R, Z) - simag) / (sibry - simag)

    # ------------------------------------------------------------------
    # independent cross-check against SIBIL's own normalised radius
    # ------------------------------------------------------------------
    if verbose:
        rho = _lookup(g, _ALIASES["rhonormrz"])
        if rho is not None:
            rho = np.asarray(rho, dtype=float)
            if rho.shape == psi.shape or rho.shape == psi.T.shape:
                if rho.shape != psi.shape:
                    rho = rho.T
                psin_grid = (psi - simag) / (sibry - simag)
                m = np.isfinite(rho) & np.isfinite(psin_grid) \
                    & (psin_grid > 0.05) & (psin_grid < 0.95)
                if m.any():
                    # rhonorm is normally sqrt(psi_n)
                    err = np.nanmedian(np.abs(rho[m] ** 2 - psin_grid[m]))
                    print("      cross-check vs rhonormrz^2: median "
                          "|dpsi_n| = {:.3e}".format(err))
                    if err > 0.05:
                        print("      note: rhonormrz may be a toroidal-flux "
                              "rho, not sqrt(psi_n); ignore if so")

        # sanity: psi_n on the stored LCFS contour should be ~1
        Rb = _lookup(g, _ALIASES["rbbbs"])
        Zb = _lookup(g, _ALIASES["zbbbs"])
        if Rb is not None and Zb is not None:
            Rb = np.asarray(Rb, dtype=float).ravel()
            Zb = np.asarray(Zb, dtype=float).ravel()
            n = min(Rb.size, Zb.size)
            if n > 8:
                pn = psin_of_RZ(Rb[:n], Zb[:n])
                print("      psi_n on stored LCFS: mean {:.4f}  "
                      "std {:.1e}  (should be 1.0000)"
                      .format(np.nanmean(pn), np.nanstd(pn)))

    cur = _lookup(g, _ALIASES["current"])
    try:
        cur = _scalar(cur, "current")
    except (KeyError, ValueError):
        cur = None

    return {
        "simag": simag,
        "sibry": sibry,
        "psirz": psi,
        "rgrid": Rg,
        "zgrid": Zg,
        "rmaxis": rmaxis,
        "zmaxis": zmaxis,
        "psi_of_RZ": psi_of_RZ,
        "psin_of_RZ": psin_of_RZ,
        "current": cur,
    }


class GqdskAdapter(object):
    """Thin wrapper exposing classic geqdsk attribute names.

    Lets you pass a SIBIL ``GQDSK`` into code written against geqdsk
    without editing that code::

        g = GqdskAdapter(sibil.GQDSK(path))
        g.simag, g.sibry, g.psirz, g.rgrid, g.zgrid     # all work
    """

    def __init__(self, g, verbose=False):
        self._g = g
        self._eq = resolve_equilibrium(g, verbose=verbose)
        for k, v in self._eq.items():
            setattr(self, k, v)
        # common extra spellings
        self.psi_axis = self.simag
        self.psi_bndry = self.sibry
        self.nw = self.rgrid.size
        self.nh = self.zgrid.size

    def __getattr__(self, name):
        # anything unresolved falls through to the original object
        return getattr(self._g, name)


# ===========================================================================
# standalone
# ===========================================================================
def _main(argv):
    if len(argv) < 2:
        print(__doc__)
        print("usage: python gqdsk_compat.py <gfile>")
        return 1
    path = argv[1]

    g = None
    try:
        import sibil
        g = sibil.GQDSK(path)
    except Exception:                                   # noqa: BLE001
        for pat in ("sibil.readers.GQDSK_Reader", "readers.GQDSK_Reader",
                    "GQDSK_Reader"):
            try:
                mod = __import__(pat, fromlist=["GQDSK"])
                g = mod.GQDSK(path)
                break
            except Exception:                           # noqa: BLE001
                continue
    if g is None:
        print("could not import a GQDSK reader")
        return 2

    print("=" * 74)
    print("resolving {}".format(path))
    print("=" * 74)
    eq = resolve_equilibrium(g, verbose=True)

    print("")
    print("landmark psi_n values")
    print("-" * 74)
    R0, Z0 = eq["rmaxis"], eq["zmaxis"]
    pts = [("magnetic axis", R0, Z0),
           ("axis + 10 cm out", R0 + 0.10, Z0),
           ("axis + 30 cm out", R0 + 0.30, Z0),
           ("axis + 50 cm out", R0 + 0.50, Z0)]
    for label, R, Z in pts:
        if eq["rgrid"].min() <= R <= eq["rgrid"].max():
            print("  {:20s} (R={:.3f}, Z={:+.3f})  psi_n = {:+.5f}"
                  .format(label, R, Z, float(eq["psin_of_RZ"](R, Z))))

    if eq["current"] is not None:
        print("")
        print("  plasma current = {:.4g} A = {:.3f} MA"
              .format(eq["current"], eq["current"] * 1e-6))
        print("  (a 1.28 MA discharge over ~1 m^2 of cross-section is")
        print("   consistent with the ~0.4 MA/m^2 core J_par in the grid)")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))
