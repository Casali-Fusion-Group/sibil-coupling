#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
boutgrid_compat.py

Resolve the poloidal-flux normalisation (``psi_axis``, ``psi_bdry``) of a
hypnotoad-generated BOUT++ grid file, under whatever name the particular
hypnotoad version happened to use, and return a validated ``psi_n(x, y)``.

Why this exists
---------------
``sibil_solps_jpar_to_bout.py`` died with

    KeyError: 'grid file missing psi_bndry'

because the grid stores the boundary flux as ``psi_bdry`` -- no ``n``.  The
spelling is not stable across hypnotoad versions:

    hypnotoad (IDL)       psi_axis        psi_bdry
    hypnotoad2 (python)   psi_axis        psi_bndry      (sometimes attrs)
    some BOUT++ tools     simag           sibry
    older grids           (absent)        (absent)       -> psixy already
                                                            normalised

For d3d_shelf_184016.03200_nc3.grd.nc the file contains all four of

    psi_axis   psi_bdry   psi_axis_gfile   psi_bdry_gfile

The ``*_gfile`` pair is the raw EFIT value; the un-suffixed pair is what
hypnotoad actually used to build ``psixy`` (they can differ by a little if
the separatrix was re-found, or by a lot if psi was rescaled by 2*pi).  So
the un-suffixed pair is preferred, and the choice is *verified* rather than
assumed.

The validation
--------------
Every candidate normalisation is checked the same way: the grid knows where
the separatrix is (``ixseps1``), so

    psi_n at the ixseps1 boundary  must equal 1.0

Specifically ``psixy[ixseps1-1, :]`` is the last closed surface and
``psixy[ixseps1, :]`` is the first open one, so psi_n interpolated halfway
between them must be 1 to within the local radial resolution.  A candidate
is accepted if |psi_n(sep) - 1| < ``tol`` (default 0.02).  A 2*pi error, a
transposed pair, or picking the g-file values when hypnotoad rescaled are
all caught instantly -- they miss by O(1) or more.

Strategies, in order
--------------------
1. grid *variables*, un-suffixed names first, then ``*_gfile``, then the
   classic ``simag`` / ``sibry``
2. grid *attributes* (hypnotoad-py writes them there, not as variables)
3. the equilibrium dict from ``gqdsk_compat.resolve_equilibrium``
4. ``psixy`` is already normalised  (detected: psi_n(sep) ~ 1 with
   ``psi_axis = 0``, ``psi_bdry = 1``)
5. self-calibration -- solve the affine map that puts psi_n = 1 exactly at
   the separatrix, anchored on the magnetic axis value if available, else
   on the innermost grid column

Strategy 5 always succeeds, so this module does not raise unless the grid
lacks ``psixy`` entirely.

Usage
-----
In ``sibil_solps_jpar_to_bout.py``::

    from boutgrid_compat import resolve_grid_psi

    # BoutGrid.__init__(self, path, geq=None, verbose=True)
    info = resolve_grid_psi(self.data, geq=geq, verbose=verbose)
    pa, pb       = info["psi_axis"], info["psi_bdry"]
    self.psinxy  = info["psinxy"]

and at the call site in ``main()``::

    grid = BoutGrid(cfg["bgrid"], geq=solps.eq, verbose=verbose)

Standalone check
----------------
    python boutgrid_compat.py path/to/grid.nc  [-g path/to/gfile]

@author: ttaczak
"""

from __future__ import annotations

import sys

import numpy as np

__all__ = ["resolve_grid_psi", "grid_psin"]


# ===========================================================================
# name tables
# ===========================================================================
#: un-suffixed first -- that is what hypnotoad used to build psixy
_AXIS_NAMES = ("psi_axis", "psiaxis", "psi_axis_gfile", "psiaxis_gfile",
               "simag", "ssimag", "psi0", "psi_mag", "psimag", "psi_a")

_BDRY_NAMES = ("psi_bdry", "psi_bndry", "psibdry", "psibndry",
               "psi_bdry_gfile", "psi_bndry_gfile", "psibdry_gfile",
               "sibry", "ssibry", "psi_sep", "psisep", "psi_lcfs", "psilim",
               "psi_b")


# ===========================================================================
# low-level access -- works for DataFile, dict, or netCDF4.Dataset
# ===========================================================================
def _read_var(src, name):
    """Read a variable by name; return None if absent."""
    # dict-like
    if isinstance(src, dict):
        return src.get(name)
    # boututils DataFile
    try:
        keys = src.list()
    except AttributeError:
        keys = None
    if keys is not None:
        if name in keys:
            try:
                return src.read(name)
            except Exception:                              # noqa: BLE001
                return None
        return None
    # netCDF4.Dataset
    var = getattr(src, "variables", None)
    if isinstance(var, dict) or hasattr(var, "keys"):
        if name in var:
            return np.array(var[name][:])
        return None
    return getattr(src, name, None)


def _list_vars(src):
    if isinstance(src, dict):
        return list(src.keys())
    try:
        return list(src.list())
    except AttributeError:
        pass
    var = getattr(src, "variables", None)
    if var is not None and hasattr(var, "keys"):
        return list(var.keys())
    return []


def _read_attr(src, name):
    """Read a netCDF global attribute, if the handle exposes any."""
    # netCDF4.Dataset
    try:
        if name in src.ncattrs():
            return getattr(src, name)
    except AttributeError:
        pass
    # boututils DataFile wrapping netCDF4
    for inner in ("handle", "_handle", "impl", "_impl", "f", "ds"):
        h = getattr(src, inner, None)
        if h is None:
            continue
        try:
            if name in h.ncattrs():
                return getattr(h, name)
        except AttributeError:
            continue
        hh = getattr(h, "handle", None)
        if hh is not None:
            try:
                if name in hh.ncattrs():
                    return getattr(hh, name)
            except AttributeError:
                pass
    # DataFile.attributes(varname) style
    try:
        att = src.attributes("")
        if isinstance(att, dict) and name in att:
            return att[name]
    except Exception:                                      # noqa: BLE001
        pass
    return None


def _scalar(v):
    """Coerce a 0-d array / 1-element array / python number to float."""
    if v is None:
        return None
    try:
        a = np.asarray(v, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    if a.size < 1 or not np.isfinite(a[0]):
        return None
    return float(a[0])


def _int(v, default=None):
    s = _scalar(v)
    return default if s is None else int(round(s))


# ===========================================================================
# validation
# ===========================================================================
def _psin_at_separatrix(psixy, ixseps, psi_axis, psi_bdry):
    """psi_n averaged over the poloidal extent of the ixseps1 boundary.

    ``psixy[ixseps-1, :]`` is the last closed surface, ``psixy[ixseps, :]``
    the first open one, so the separatrix sits halfway between.
    """
    nx = psixy.shape[0]
    if ixseps is None or ixseps < 1 or ixseps > nx - 1:
        return None
    den = psi_bdry - psi_axis
    if abs(den) < 1e-30:
        return None
    lo = (psixy[ixseps - 1, :] - psi_axis) / den
    hi = (psixy[ixseps, :] - psi_axis) / den
    # core-side columns only: in the PFR the same psixy maps to psi_n < 1
    # legitimately, so average rather than take an extremum
    return float(np.nanmean(0.5 * (lo + hi)))


def _validate(psixy, ixseps, psi_axis, psi_bdry, tol):
    """Return (ok, psin_sep, err)."""
    ps = _psin_at_separatrix(psixy, ixseps, psi_axis, psi_bdry)
    if ps is None or not np.isfinite(ps):
        return False, ps, np.inf
    err = abs(ps - 1.0)
    return (err < tol), ps, err


# ===========================================================================
# main entry point
# ===========================================================================
def resolve_grid_psi(src, geq=None, tol=0.02, verbose=True):
    """Find and validate the psi normalisation of a BOUT++ grid file.

    Parameters
    ----------
    src : DataFile / dict / netCDF4.Dataset
        Open grid file, or a dict of its contents.
    geq : dict, optional
        Output of ``gqdsk_compat.resolve_equilibrium`` -- supplies the
        g-file fallback.
    tol : float
        Maximum allowed |psi_n(separatrix) - 1|.
    verbose : bool

    Returns
    -------
    dict
        ``psi_axis``, ``psi_bdry``, ``psinxy`` (2-D, same shape as psixy),
        ``psixy``, ``ixseps1``, ``source`` (which strategy won),
        ``psin_sep`` (the validation number), ``candidates`` (all tried).
    """
    def log(*a):
        if verbose:
            print(*a)

    psixy = _read_var(src, "psixy")
    if psixy is None:
        for alt in ("psi", "psi_xy", "psixy_c"):
            psixy = _read_var(src, alt)
            if psixy is not None:
                break
    if psixy is None:
        raise KeyError("grid file has no 'psixy' -- cannot build psi_n")
    psixy = np.asarray(psixy, dtype=float)

    ixseps1 = _int(_read_var(src, "ixseps1"))
    if ixseps1 is None:
        ixseps1 = _int(_read_attr(src, "ixseps1"))
    nx, ny = psixy.shape

    log("    grid psixy {}  ixseps1 = {}".format(psixy.shape, ixseps1))
    log("    psixy range: {:+.6g} .. {:+.6g}".format(
        float(np.nanmin(psixy)), float(np.nanmax(psixy))))

    cands = []          # (label, psi_axis, psi_bdry)

    # ---------------------------------------------------------------- 1
    present = set(_list_vars(src))
    va = [(n, _scalar(_read_var(src, n))) for n in _AXIS_NAMES if n in present]
    vb = [(n, _scalar(_read_var(src, n))) for n in _BDRY_NAMES if n in present]
    va = [(n, v) for n, v in va if v is not None]
    vb = [(n, v) for n, v in vb if v is not None]
    if va and vb:
        # pair like-with-like: un-suffixed with un-suffixed, gfile with gfile
        plain_a = [(n, v) for n, v in va if not n.endswith("_gfile")]
        plain_b = [(n, v) for n, v in vb if not n.endswith("_gfile")]
        gf_a = [(n, v) for n, v in va if n.endswith("_gfile")]
        gf_b = [(n, v) for n, v in vb if n.endswith("_gfile")]
        if plain_a and plain_b:
            cands.append(("grid vars {}/{}".format(plain_a[0][0], plain_b[0][0]),
                          plain_a[0][1], plain_b[0][1]))
        if gf_a and gf_b:
            cands.append(("grid vars {}/{}".format(gf_a[0][0], gf_b[0][0]),
                          gf_a[0][1], gf_b[0][1]))
        if not cands:
            cands.append(("grid vars {}/{}".format(va[0][0], vb[0][0]),
                          va[0][1], vb[0][1]))

    # ---------------------------------------------------------------- 2
    aa = next((v for v in (_scalar(_read_attr(src, n)) for n in _AXIS_NAMES)
               if v is not None), None)
    ab = next((v for v in (_scalar(_read_attr(src, n)) for n in _BDRY_NAMES)
               if v is not None), None)
    if aa is not None and ab is not None:
        cands.append(("netCDF attributes", aa, ab))

    # ---------------------------------------------------------------- 3
    if geq is not None:
        ga = geq.get("simag") if isinstance(geq, dict) else None
        gb = geq.get("sibry") if isinstance(geq, dict) else None
        if ga is not None and gb is not None:
            cands.append(("g-file (gqdsk_compat)", float(ga), float(gb)))
            # and the 2*pi-rescaled variants, in case hypnotoad used Wb
            # rather than Wb/rad
            tp = 2.0 * np.pi
            cands.append(("g-file x 2pi", float(ga) * tp, float(gb) * tp))

    # ---------------------------------------------------------------- 4
    cands.append(("psixy already normalised", 0.0, 1.0))

    # ---------------------------------------------------------------- 5
    #   affine map forced to give psi_n = 1 at the separatrix, with the
    #   inner edge of the grid as the second anchor
    if ixseps1 is not None and 1 <= ixseps1 <= nx - 1:
        psi_sep = float(np.nanmean(
            0.5 * (psixy[ixseps1 - 1, :] + psixy[ixseps1, :])))
        psi_in = float(np.nanmean(psixy[0, :]))
        # psi_n(inner) is unknown; take it from the best grid-var candidate
        # if we have one, else assume the inner edge is at psi_n = 0.
        pn_in = 0.0
        if cands:
            la, lb = cands[0][1], cands[0][2]
            if abs(lb - la) > 1e-30:
                pn_in = (psi_in - la) / (lb - la)
        if abs(1.0 - pn_in) > 1e-6:
            # solve  (psi_in - A)/(B - A) = pn_in  and (psi_sep - A)/(B-A) = 1
            B_minus_A = (psi_sep - psi_in) / (1.0 - pn_in)
            A = psi_sep - B_minus_A
            cands.append(("self-calibrated at ixseps1", A, A + B_minus_A))

    # ---------------------------------------------------------------- test
    log("")
    log("    psi normalisation candidates   (target psi_n(sep) = 1.0000)")
    log("    " + "-" * 66)
    results = []
    for label, pa, pb in cands:
        ok, ps, err = _validate(psixy, ixseps1, pa, pb, tol)
        results.append((label, pa, pb, ok, ps, err))
        log("      {:34s} axis {:+.6g}  bdry {:+.6g}".format(label, pa, pb))
        log("      {:34s} psi_n(sep) = {}   {}".format(
            "", "   n/a" if ps is None else "{:8.5f}".format(ps),
            "OK" if ok else "reject"))

    good = [r for r in results if r[3]]
    if good:
        good.sort(key=lambda r: r[5])
        label, pa, pb, _, ps, err = good[0]
    else:
        results.sort(key=lambda r: r[5])
        label, pa, pb, _, ps, err = results[0]
        print("    WARNING: no candidate put psi_n(sep) within {:.3g} of 1; "
              "using the closest ({}), psi_n(sep) = {}".format(
                  tol, label, "n/a" if ps is None else "{:.5f}".format(ps)))

    log("")
    log("    -> using {}".format(label))
    log("       psi_axis = {:+.6g}   psi_bdry = {:+.6g}   dpsi = {:+.6g}"
        .format(pa, pb, pb - pa))
    log("       psi_n(separatrix) = {:.5f}  (err {:.2e})".format(ps, err))

    psinxy = (psixy - pa) / (pb - pa)
    log("       psi_n on grid: {:.4f} .. {:.4f}".format(
        float(np.nanmin(psinxy)), float(np.nanmax(psinxy))))

    # extra diagnostic: how far the two stored pairs disagree
    if len(va) and len(vb):
        plain_a = [v for n, v in va if not n.endswith("_gfile")]
        gf_a = [v for n, v in va if n.endswith("_gfile")]
        if plain_a and gf_a and abs(plain_a[0] - gf_a[0]) > 1e-9:
            log("       note: psi_axis differs from psi_axis_gfile by "
                "{:+.3e} ({:.2f}%) -- hypnotoad re-found the axis"
                .format(plain_a[0] - gf_a[0],
                        100.0 * abs(plain_a[0] - gf_a[0])
                        / max(abs(gf_a[0]), 1e-30)))

    return {
        "psi_axis": pa,
        "psi_bdry": pb,
        "psinxy": psinxy,
        "psixy": psixy,
        "ixseps1": ixseps1,
        "nx": nx,
        "ny": ny,
        "source": label,
        "psin_sep": ps,
        "candidates": results,
    }


def grid_psin(src, geq=None, verbose=False):
    """Convenience wrapper: return just the 2-D psi_n array."""
    return resolve_grid_psi(src, geq=geq, verbose=verbose)["psinxy"]


# ===========================================================================
# standalone
# ===========================================================================
def _main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(
        description="resolve the psi normalisation of a BOUT++ grid file")
    ap.add_argument("grid")
    ap.add_argument("-g", "--gfile", default=None,
                    help="EFIT g-file, for the fallback normalisation")
    ap.add_argument("--tol", type=float, default=0.02)
    args = ap.parse_args(argv)

    from boututils.datafile import DataFile

    geq = None
    if args.gfile:
        try:
            from gqdsk_compat import resolve_equilibrium
            import sibil
            geq = resolve_equilibrium(sibil.GQDSK(args.gfile), verbose=False)
        except Exception as exc:                           # noqa: BLE001
            print("    (could not load g-file: {})".format(exc))

    print("=" * 74)
    print("resolving {}".format(args.grid))
    print("=" * 74)

    with DataFile(args.grid) as f:
        info = resolve_grid_psi(f, geq=geq, tol=args.tol, verbose=True)

        print("")
        print("separatrix check by poloidal location")
        print("-" * 74)
        pn = info["psinxy"]
        ix = info["ixseps1"]
        if ix is not None and 1 <= ix < pn.shape[0]:
            sep = 0.5 * (pn[ix - 1, :] + pn[ix, :])
            print("  psi_n at ixseps1, over y:  min {:.5f}  mean {:.5f}  "
                  "max {:.5f}  std {:.2e}".format(
                      float(np.nanmin(sep)), float(np.nanmean(sep)),
                      float(np.nanmax(sep)), float(np.nanstd(sep))))
            print("  (std should be small; a large spread means psixy is not "
                  "a flux function on this mesh)")

        print("")
        print("radial profile of psi_n at the outer midplane")
        print("-" * 74)
        jy = pn.shape[1] // 2
        col = pn[:, jy]
        idx = [0, max(1, info["nx"] // 4), max(1, info["nx"] // 2),
               (ix - 1 if ix else info["nx"] - 2), (ix if ix else
                                                    info["nx"] - 1),
               info["nx"] - 1]
        for i in sorted(set(int(k) for k in idx if 0 <= k < info["nx"])):
            tag = ""
            if ix is not None and i == ix - 1:
                tag = "   <- last closed"
            elif ix is not None and i == ix:
                tag = "   <- first open"
            print("  ix = {:4d}   psi_n = {:+.5f}{}".format(i, col[i], tag))

    return 0


if __name__ == "__main__":
    sys.exit(_main())
