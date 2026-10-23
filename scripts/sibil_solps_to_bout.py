#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sibil_solps_to_bout.py   (combined: profiles + Jpar0 + PF correction)

Loads SIBIL_config.yaml, prints a summary, asks for confirmation, then

  1. runs sibil.SIBIL_Midplane_Mapper  -> Niexp/Neexp/Tiexp/Teexp/P/pressure
     are mapped from SOLPS onto the BOUT++ grid (unchanged from before),
  2. rebuilds Jpar0 for this SOLPS iteration:
       a. if Jpar0 was smoothed by a previous iteration, unsmooth it
          (subtract the stored smoothing delta Jpar0_smooth),
       b. subtract the previous iteration's SOLPS current (Jpar0_solps),
       c. add this iteration's SOLPS edge current and re-apply the C2
          (v7) smoothing,
  3. corrects the private-flux region of the freshly mapped profiles
     (flat / flux / solps modes, as in sibil_pf_profiles.py),
  4. writes everything back into the grid in config["files"]["bgrid"].

Jpar0 is stored in the grid as its three parts so the next iteration can
undo exactly what this one did:

    Jpar0 = Jpar0_hypnotoad + Jpar0_smooth + Jpar0_solps

A grid smoothed by the old standalone v7 script (Jpar0_hypnotoad present,
no Jpar0_smooth) is unsmoothed by falling back to Jpar0_hypnotoad.

The PF correction starts from this iteration's mapped profiles; they are
saved as <name>_orig (overwritten every iteration).  'P' is not touched by
the PF correction (same as the standalone script).

Options for steps 2 and 3 live in the 'jpar' and 'pf_profiles' sections of
SIBIL_config.yaml; missing keys take the defaults below (the defaults of the
standalone scripts).

Usage:
    python sibil_solps_to_bout.py          (reads ./SIBIL_config.yaml)

@author: ttaczak
"""

import os
import sys

import numpy as np
import scipy.interpolate as si

from sibil_io_functions import load_config, sibil_config_summary, confirm

DEFAULT_CONFIG_PATH = "SIBIL_config.yaml"

MU0 = 4.0e-7 * np.pi

JPAR_DEFAULTS = {
    "mode": "add",              # add | repair-only
    "sign": 1.0,                # flip the SOLPS parallel direction
    "fch_convention": "normal", # normal | swapped (only if b2fstate has no fch_p)
    "psin_blend_start": 0.970,  # core handoff start (psi_n)
    "psin_blend_end": 0.995,    # core handoff end; c_edge = <J_hyp/B> here
    "sol_taper": 0.030,         # SOL taper width (psi_n)
    "sol_taper_mm": None,       # SOL taper width (mm at OMP), supersedes sol_taper
    "pf_taper": 0.010,          # PF taper width (psi_n)
    "pf_taper_mm": None,        # PF taper width (mm), supersedes pf_taper
    "leg_taper": False,         # also taper SOL/PF current along the legs
    "solps_on": 0.005,          # SOLPS current switch-on width outside psi_n = 1
    "solps_off": 0.010,         # SOLPS current switch-off width before domain edge
    "psin_max": None,           # trim SOLPS OMP cells above this psi_n
    "smooth": 5.0e-3,           # smoothing of the SOLPS J/B quintic spline
    "write_apar": False,        # also write diagnostic Apar0
}

PF_DEFAULTS = {
    "mode": "flat",             # flat | flux | solps
    "width": 0.005,             # psi_n blend width inside the separatrix
    "pf_frac": 1.0,             # flat mode: PF target = pf_frac * separatrix value
    "fields": ["Niexp", "Neexp", "Teexp", "Tiexp"],
    "plot": None,               # png of old/new PF profiles
}

# PF solps mode: which SOLPS profile (on the mapper's Solps object) feeds each field
SOLPS_KEY = {"Niexp": "ne", "Neexp": "ne", "Teexp": "Te", "Tiexp": "Ti"}


def _hdr(title):
    print("")
    print("=" * 74)
    print(title)
    print("=" * 74)


def _settings(config, section, defaults):
    s = dict(defaults)
    s.update(config.get(section) or {})
    print("  [{}]".format(section))
    for k, v in s.items():
        print("    {:<18} {}".format(k, v))
    return s


# ===========================================================================
# shared geometry (built from the mapper's Bout / Solps objects)
# ===========================================================================
class Geometry(object):
    """Everything steps 2 and 3 need, taken from the mapper's objects plus
    the grid-file arrays the Bout object does not hold."""

    def __init__(self, mapper, grid):
        import sibil.readers.B2fgmtry_Reader as gmr
        import sibil.readers.B2fstate_Reader as str_
        from sibil.readers.GQDSK_Reader import GQDSK

        files = mapper.config["files"]
        b, s = mapper.bout, mapper.solps
        self.bout, self.solps, self.grid = b, s, grid

        # ---- BOUT++ ------------------------------------------------------
        self.nx, self.ny = b.nx, b.ny
        self.ixseps1 = b.ixsep + 1             # Bout.ixsep = ixseps1 - 1
        self.jyseps1_1, self.jyseps2_2 = b.jyseps11, b.jyseps22
        self.psinxy = np.asarray(b.psinxy, float)
        self.Rxy = np.asarray(b.rxy, float)
        self.Zxy = np.asarray(b.zxy, float)
        self.Bxy = np.asarray(grid["Bxy"], float)

        self.legs = np.isin(np.arange(self.ny), b.j_pfr)
        inside = np.arange(self.nx)[:, None] <= b.ixsep
        shape = (self.nx, self.ny)
        self.core = np.broadcast_to(inside & ~self.legs[None, :], shape).copy()
        self.pf = np.broadcast_to(inside & self.legs[None, :], shape).copy()
        self.sol = np.broadcast_to(~inside, shape).copy()

        core_j = np.flatnonzero(~self.legs)
        self.jomp = int(core_j[np.argmax(self.Rxy[self.ixseps1, core_j])])

        # ---- equilibrium: same psi_n map the Solps object uses -----------
        self.g = GQDSK(files["gfile"])
        g = self.g
        interp = si.RegularGridInterpolator((g.R_grid, g.Z_grid), g.psi,
                                            method="cubic")

        def psin_of_RZ(R, Z):
            R = np.asarray(R, float)
            pts = np.column_stack((R.ravel(), np.asarray(Z, float).ravel()))
            psi = interp(pts).reshape(R.shape)
            return (g.psiaxis - psi) / (g.psiaxis - g.psiedge)
        self.psin_of_RZ = psin_of_RZ

        # ---- SOLPS, in b2 (poloidal, radial) order -----------------------
        self.gm = gmr.B2fgmtry(files["b2fgmtry"])
        self.st = str_.B2fstate(files["b2fstate"])
        self.s_Rc = np.asarray(s.rxy, float).T
        self.s_Zc = np.asarray(s.zxy, float).T
        self.s_psin = np.asarray(s.psinxy, float).T
        self.s_npol, self.s_nrad = self.s_Rc.shape
        R, Z = self.s_Rc[:, 1], self.s_Zc[:, 1]
        ok = np.isfinite(R) & (np.abs(Z - float(g.Zaxis)) < 0.5)
        if not ok.any():
            ok = np.isfinite(R)
        idx = np.arange(self.s_npol)[ok]
        self.s_jomp = int(idx[np.nanargmax(R[ok])])
        print("    BOUT++ OMP column j = {},  SOLPS OMP ix = {}"
              .format(self.jomp, self.s_jomp))

    def xpoint_Z(self):
        i, j = self.ixseps1, self.jyseps1_1
        return float(self.Zxy[i - 1:i + 1, j:j + 2].mean())

    def sep_value(self, f):
        """f at psi_n = 1 on the outer-midplane column."""
        p = self.psinxy[:, self.jomp]
        o = np.argsort(p)
        return float(np.interp(1.0, p[o], f[o, self.jomp]))


# ===========================================================================
# C2 building blocks
# ===========================================================================
def smoothstep5(t):
    """Quintic smoothstep 6t^5 - 15t^4 + 10t^3, clipped to [0, 1]."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def psin_per_metre(geo):
    """|dpsi_n/dr| at the separatrix [1/m], median over poloidal columns."""
    ix = geo.ixseps1
    if ix < 1 or ix >= geo.nx:
        return None
    dpsin = geo.psinxy[ix, :] - geo.psinxy[ix - 1, :]
    dr = np.hypot(geo.Rxy[ix, :] - geo.Rxy[ix - 1, :],
                  geo.Zxy[ix, :] - geo.Zxy[ix - 1, :])
    good = np.isfinite(dpsin) & np.isfinite(dr) & (dr > 0)
    if not good.any():
        return None
    return float(np.nanmedian(np.abs(dpsin[good] / dr[good])))


def resolve_widths(geo, cfg):
    """SOL / PF taper widths in psi_n, converting the *_mm options if set."""
    sol, pf = float(cfg["sol_taper"]), float(cfg["pf_taper"])
    if cfg["sol_taper_mm"] is not None or cfg["pf_taper_mm"] is not None:
        g = psin_per_metre(geo)
        if g is None:
            print("    *_mm widths requested but the grid metric is "
                  "unusable; using the psi_n values")
        else:
            if cfg["sol_taper_mm"] is not None:
                sol = g * cfg["sol_taper_mm"] * 1e-3
            if cfg["pf_taper_mm"] is not None:
                pf = g * cfg["pf_taper_mm"] * 1e-3
            print("    dpsi_n/dr at separatrix = {:.2f} 1/m".format(g))
    print("    SOL taper = {:.4f} psi_n,  PF taper = {:.4f} psi_n".format(sol, pf))
    return sol, pf


def _surface_weights(geo):
    """Flux-surface-average weights per cell: dl/Bp = hthe*dy/Bp (or 1)."""
    hthe = geo.grid.get("hthe")
    Bp = geo.bout.bpxy
    dy = geo.grid.get("dy")
    if hthe is None or Bp is None:
        return np.ones_like(geo.psinxy)
    w = np.asarray(hthe, float) / np.maximum(np.abs(np.asarray(Bp, float)), 1e-6)
    if dy is not None:
        w = w * np.asarray(dy, float)
    return np.where(np.isfinite(w) & (w > 0), w, 0.0)


def edge_coefficient(geo, jhyp, psin_ref):
    """c_edge = <J_hyp / B> on the surface psi_n = psin_ref (core columns)."""
    core = geo.core
    jb = jhyp / geo.Bxy
    wts = _surface_weights(geo)
    vals, ws = [], []
    for jy in np.flatnonzero(core.any(axis=0)):
        col = core[:, jy]
        p = geo.psinxy[col, jy]
        f = jb[col, jy]
        o = np.argsort(p)
        p, f = p[o], f[o]
        if not (p[0] <= psin_ref <= p[-1]):
            continue
        vals.append(np.interp(psin_ref, p, f))
        ws.append(np.interp(psin_ref, p, wts[col, jy][o]))
    if not vals:
        raise ValueError("no core column spans psi_n = {}".format(psin_ref))
    vals, ws = np.asarray(vals), np.asarray(ws)
    c = float(np.sum(vals * ws) / np.sum(ws)) if ws.sum() > 0 else float(np.mean(vals))
    spread = np.std(vals) / (abs(c) + 1e-300)
    print("    c_edge = <J_hyp/B> at psi_n = {:.4f}: {:.4e} A/m^2/T "
          "(poloidal spread {:.1%} over {} columns)"
          .format(psin_ref, c, spread, len(vals)))
    return c


def leg_profile(geo):
    """Per-column factor 1 at the X-point row -> 0 at the target (quintic)."""
    L = np.ones(geo.ny)
    j11, j22 = geo.jyseps1_1, geo.jyseps2_2
    if j11 >= 0:
        n = j11 + 1
        L[:n] = smoothstep5((np.arange(n) + 0.5) / n)
    if j22 < geo.ny - 1:
        n = geo.ny - 1 - j22
        L[j22 + 1:] = smoothstep5((n - 0.5 - np.arange(n)) / n)
    return L


# ===========================================================================
# step 2: Jpar0
# ===========================================================================
def solps_jpar(geo, cfg):
    """J_par [A/m^2] at SOLPS cell centres, (poloidal, radial)."""
    gm, st = geo.gm, geo.st
    hx = np.asarray(gm.hx, float)
    vol = getattr(gm, "vol", None)
    if vol is not None and np.nanmedian(np.abs(hx)) > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            Apol = np.asarray(vol, float) / hx
        print("    poloidal face area from vol/hx")
    else:
        Apol = np.asarray(gm.hy, float) * np.asarray(gm.hz, float)
        print("    poloidal face area from hy*hz")

    fch_p = getattr(st, "fch_p", None)
    if fch_p is not None and np.nanmax(np.abs(np.asarray(fch_p, float))) > 0:
        with np.errstate(divide="ignore", invalid="ignore"):
            jpar = np.asarray(fch_p, float) / Apol
        src = "fch_p"
    else:
        fch = getattr(st, "fch", None)
        if fch is None:
            raise KeyError("b2fstate has no 'fch' array -- currents were not "
                           "solved in this run")
        fch = np.asarray(fch, float)
        fchx = fch[..., 1] if cfg["fch_convention"].startswith("swapped") else fch[..., 0]
        favg = np.empty_like(fchx)
        favg[:-1, :] = 0.5 * (fchx[:-1, :] + fchx[1:, :])
        favg[-1, :] = fchx[-1, :]
        bb = np.asarray(gm.bb, float)
        med = np.array([np.nanmedian(np.abs(bb[..., k])) for k in range(bb.shape[-1])])
        itot = int(np.nanargmax(med))
        cand = [k for k in range(len(med)) if k != itot and med[k] > 1e-3 * med[itot]]
        ipol = int(min(cand, key=lambda k: med[k])) if cand else 0
        print("    bb: |B| = index {}, |B_pol| = index {}".format(itot, ipol))
        with np.errstate(divide="ignore", invalid="ignore"):
            jpar = favg / Apol * np.abs(bb[..., itot]) / np.abs(bb[..., ipol])
        src = "fchx * |B|/|B_pol|"

    jpar = np.where(np.isfinite(jpar), jpar, 0.0) * float(cfg["sign"])
    print("    J_par source: {}   max |J_par| = {:.4f} MA/m^2"
          .format(src, np.nanmax(np.abs(jpar)) * 1e-6))
    return jpar


def omp_profile(geo, jpar, psin_max=None):
    """1-D (psi_n, J_par) at the SOLPS outer midplane, sorted and unique."""
    j = geo.s_jomp
    psin = geo.s_psin[j, 1:-1].copy()
    jp = jpar[j, 1:-1].copy()
    good = np.isfinite(psin) & np.isfinite(jp)
    if psin_max is not None:
        good &= psin <= psin_max
    psin, jp = psin[good], jp[good]
    order = np.argsort(psin)
    psin, jp = psin[order], jp[order]
    psin, uniq = np.unique(psin, return_index=True)
    jp = jp[uniq]
    if psin.size < 3:
        raise ValueError("only {} usable SOLPS midplane points after trimming -- "
                         "check psin_max".format(psin.size))
    print("    OMP profile: {} points, psi_n {:.4f} .. {:.4f},  max |J_par| = {:.4f} MA/m^2"
          .format(psin.size, psin[0], psin[-1], np.nanmax(np.abs(jp)) * 1e-6))
    return psin, jp


def solps_flux_function(geo, psin_src, jpar_src, smooth):
    """c_solps(psi_n) = J_SOLPS / B at the BOUT++ OMP column, quintic spline."""
    p_omp = geo.psinxy[:, geo.jomp]
    o = np.argsort(p_omp)
    B_src = np.interp(psin_src, p_omp[o], geo.Bxy[o, geo.jomp])
    f = jpar_src / B_src
    k = 5 if psin_src.size > 6 else min(3, psin_src.size - 1)
    s = smooth * np.nanvar(f) * psin_src.size if smooth > 0 else 0.0
    spl = si.UnivariateSpline(psin_src, f, k=k, s=s)
    print("    SOLPS: quintic spline of J/B at OMP column j = {} (B = {:.3f}..{:.3f} T)"
          .format(geo.jomp, B_src.min(), B_src.max()))
    return spl, (psin_src[0], psin_src[-1])


def recover_hypnotoad_current(grid):
    """Undo the previous iteration: unsmooth, then remove its SOLPS current."""
    J = np.asarray(grid["Jpar0"], float).copy()
    if "Jpar0_smooth" in grid:
        print("    Jpar0 was smoothed by the previous iteration: subtracting Jpar0_smooth")
        J -= np.asarray(grid["Jpar0_smooth"], float)
    elif "Jpar0_hypnotoad" in grid:
        print("    Jpar0 was smoothed by the standalone v7 script (no stored parts): "
              "using Jpar0_hypnotoad")
        J = np.asarray(grid["Jpar0_hypnotoad"], float).copy()
    else:
        print("    Jpar0 has not been smoothed: using it as the hypnotoad current")
    if "Jpar0_solps" in grid:
        print("    subtracting the previous iteration's Jpar0_solps")
        J -= np.asarray(grid["Jpar0_solps"], float)
    return J


def build_jpar(geo, cfg):
    """Returns the arrays to write for Jpar0 and its stored parts."""
    _hdr("Jpar0   (mode = {})".format(cfg["mode"]))
    grid = geo.grid
    J_prev = np.asarray(grid["Jpar0"], float)
    jhyp = recover_hypnotoad_current(grid)

    psin, B = geo.psinxy, geo.Bxy
    core, sol, pf = geo.core, geo.sol, geo.pf
    a, b = cfg["psin_blend_start"], cfg["psin_blend_end"]
    sol_w, pf_w = resolve_widths(geo, cfg)
    c_edge = edge_coefficient(geo, jhyp, b)

    # smoothed hypnotoad extension (v7)
    S = smoothstep5((psin - a) / (b - a))
    E = 1.0 - smoothstep5((psin - 1.0) / sol_w)
    P = 1.0 - smoothstep5((1.0 - psin) / pf_w)
    L = leg_profile(geo)[None, :] if cfg["leg_taper"] else 1.0
    legs = geo.legs[None, :]

    j_ext = np.zeros_like(psin)
    j_ext[core] = ((1.0 - S) * jhyp + S * c_edge * B)[core]
    j_ext[sol] = (c_edge * B * E * np.where(legs, L, 1.0))[sol]
    j_ext[pf] = (c_edge * B * P * np.where(legs, L, 1.0))[pf]

    # this iteration's SOLPS edge current, SOL only
    j_solps = np.zeros_like(psin)
    if cfg["mode"] == "add":
        jpar = solps_jpar(geo, cfg)
        psin_src, jpar_src = omp_profile(geo, jpar, psin_max=cfg["psin_max"])
        spl, (lo, hi) = solps_flux_function(geo, psin_src, jpar_src, cfg["smooth"])
        on = smoothstep5((psin - 1.0) / cfg["solps_on"])
        off = 1.0 - smoothstep5((psin - (hi - cfg["solps_off"])) / cfg["solps_off"])
        inside = (psin >= lo) & (psin <= hi)
        Ws = np.where(sol & inside, on * off, 0.0)
        cs = np.where(inside, spl(np.clip(psin, lo, hi)), 0.0)
        j_solps = Ws * cs * B * np.where(legs, L, 1.0)

    J = np.where(np.isfinite(j_ext + j_solps), j_ext + j_solps, 0.0)
    report_jpar(geo, J_prev, J)

    out = {"Jpar0": J,
           "Jpar0_smooth": J - j_solps - jhyp,
           "Jpar0_solps": j_solps}
    if "Jpar0_hypnotoad" not in grid:
        out["Jpar0_hypnotoad"] = jhyp
    if cfg["write_apar"]:
        out["Apar0"] = build_apar(geo, J)
    return out


def radial_kink(geo, J):
    i0, i1 = geo.ixseps1 - 4, geo.ixseps1 + 3
    Jp, J0, Jm = J[i0 + 1:i1 + 2], J[i0:i1 + 1], J[i0 - 1:i1]
    k = np.abs(Jp - 2 * J0 + Jm) / (0.5 * np.abs(Jp - Jm) + 1e-3 * np.nanmax(np.abs(J)) + 1e-300)
    i, j = np.unravel_index(np.nanargmax(k), k.shape)
    return float(k[i, j]), i0 + i, j


def xpoint_jump(geo, J):
    jb = J / geo.Bxy
    d = np.abs(np.diff(jb[geo.ixseps1:, :], axis=1))
    i, j = np.unravel_index(np.nanargmax(d), d.shape)
    return float(d[i, j] / (np.nanmax(np.abs(jb)) + 1e-300)), geo.ixseps1 + i, j


def report_jpar(geo, J_old, J_new):
    print("")
    for lab, J in (("grid Jpar0 now", J_old), ("new", J_new)):
        k, ki, kj = radial_kink(geo, J)
        x, xi, xj = xpoint_jump(geo, J)
        print("  {:<15} radial kink K (ix {}..{}) max {:6.2f} at ix={} j={};   "
              "max one-cell |d(J/B)|/max along y in SOL {:6.3f} at ix={} j={}->{}"
              .format(lab, geo.ixseps1 - 4, geo.ixseps1 + 3, k, ki, kj, x, xi, xj, xj + 1))
    print("  max |J0|: grid Jpar0 now {:.4f} MA/m^2   new {:.4f} MA/m^2"
          .format(np.nanmax(np.abs(J_old)) * 1e-6, np.nanmax(np.abs(J_new)) * 1e-6))


def build_apar(geo, J):
    """Diagnostic A_par from radial integration of mu0 * J_par."""
    Bp = np.asarray(geo.bout.bpxy, float)
    dx = np.asarray(geo.grid["dx"], float)
    with np.errstate(divide="ignore", invalid="ignore"):
        dr = dx / (geo.Rxy * Bp)
    dr = np.where(np.isfinite(dr), dr, 0.0)
    A = -MU0 * np.cumsum(np.cumsum(J * dr, axis=0) * dr, axis=0)
    A -= A[0, :]
    return A


# ===========================================================================
# step 3: private-flux profiles
# ===========================================================================
def sep_check(geo):
    _hdr("separatrix consistency  (everything measured against the g-file psi_n)")
    R, Z, i = geo.Rxy, geo.Zxy, geo.ixseps1
    core_j = np.flatnonzero(~geo.legs)
    jomp = geo.jomp

    pg = geo.psin_of_RZ(R, Z)
    print("  grid psi_n vs g-file psi_n at Rxy,Zxy: max |diff| {:.2e}  (core+SOL cells)"
          .format(np.nanmax(np.abs(geo.psinxy - pg)[:, core_j])))
    dpdr = (pg[i, jomp] - pg[i - 1, jomp]) / (R[i, jomp] - R[i - 1, jomp])
    dr_cell = abs(R[i, jomp] - R[i - 1, jomp]) * 1e3

    pb = geo.psin_of_RZ(0.5 * (R[i - 1] + R[i]), 0.5 * (Z[i - 1] + Z[i]))[core_j]
    k = np.argmax(np.abs(pb - 1))
    po = pb[list(core_j).index(jomp)]
    print("  BOUT  separatrix (mid ix {}/{}):  psi_n at OMP {:.6f} ({:+.3f} mm, {:+.2f} cells);"
          "  worst core column j={} {:.6f}"
          .format(i - 1, i, po, (po - 1) / dpdr * 1e3, (po - 1) / dpdr * 1e3 / dr_cell,
                  core_j[k], pb[k]))
    print("        OMP radial cell at the separatrix: {:.3f} mm".format(dr_cell))

    crx = np.asarray(geo.gm.crx, float).reshape(geo.s_npol, geo.s_nrad, -1)
    cry = np.asarray(geo.gm.cry, float).reshape(geo.s_npol, geo.s_nrad, -1)
    dev = geo.psin_of_RZ(crx, cry) - 1.0
    best = np.nanmin(np.abs(dev), axis=(1, 2))
    above = geo.s_Zc[:, 1] > geo.xpoint_Z()
    jo = geo.s_jomp
    r, c = np.unravel_index(np.nanargmin(np.abs(dev[jo])), dev[jo].shape)
    p_omp = dev[jo, r, c] + 1.0
    print("  SOLPS separatrix (nearest cell corner): psi_n at OMP {:.6f} ({:+.3f} mm, {:+.2f} BOUT cells);"
          "  worst core column {:.2e}"
          .format(p_omp, (p_omp - 1) / dpdr * 1e3, (p_omp - 1) / dpdr * 1e3 / dr_cell,
                  np.nanmax(best[above])))
    print("  offsets of more than ~1 BOUT cell mean the two codes disagree on the separatrix"
          "\n  (different g-file, or Carre/hypnotoad X-point psi) -- fix before mapping profiles")


def solps_pf_profile(geo, f, zx, nbins=24):
    """Median of a SOLPS (poloidal, radial) field over PF cells, binned in psi_n."""
    p, z, v = geo.s_psin[1:-1, 1:-1], geo.s_Zc[1:-1, 1:-1], f[1:-1, 1:-1]
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


def solps_omp_sep(geo, f):
    p, v = geo.s_psin[geo.s_jomp, 1:-1], f[geo.s_jomp, 1:-1]
    o = np.argsort(p)
    return float(np.interp(1.0, p[o], v[o]))


def build_pf(geo, cfg):
    """PF-corrected profiles, starting from this iteration's mapped ones."""
    orig = {n: np.asarray(geo.grid[n], float).copy()
            for n in cfg["fields"] if n in geo.grid}
    pf = geo.pf
    S = smoothstep5((1.0 - geo.psinxy) / cfg["width"])
    zx = geo.xpoint_Z() if cfg["mode"] == "solps" else None
    new, info = {}, {}
    for name, f in orig.items():
        fsep = geo.sep_value(f)
        if cfg["mode"] == "flat":
            target = np.full_like(f, cfg["pf_frac"] * fsep)
            info[name] = (fsep, None)
        elif cfg["mode"] == "flux":
            p = geo.psinxy[:, geo.jomp]
            o = np.argsort(p)
            new[name] = f.copy()
            new[name][pf] = np.interp(geo.psinxy, p[o], f[o, geo.jomp])[pf]
            info[name] = (fsep, None)
            continue
        else:
            sv = np.asarray(getattr(geo.solps, SOLPS_KEY[name]), float).T
            scale = fsep / solps_omp_sep(geo, sv)
            pc, vc, raw = solps_pf_profile(geo, sv, zx)
            target = np.interp(geo.psinxy, pc, vc * scale)
            info[name] = (fsep, (scale, pc, vc * scale, raw[0], raw[1] * scale))
        out = f.copy()
        out[pf] = ((1.0 - S) * fsep + S * target)[pf]
        new[name] = np.maximum(out, 1e-3 * fsep)

    report_pf(geo, orig, new, info, cfg)
    if cfg["plot"]:
        plot_pf(geo, orig, new, info, cfg["plot"])

    write = {}
    for n in new:
        write[n + "_orig"] = orig[n]
        write[n] = new[n]
    # keep pressure consistent with N(Te+Ti) in the PF
    if all(k in new for k in ("Niexp", "Teexp", "Tiexp")):
        ratio = (new["Niexp"] * (new["Teexp"] + new["Tiexp"])) / \
                (orig["Niexp"] * (orig["Teexp"] + orig["Tiexp"]))
        for pk in ("pressure", "pressure_s"):
            if pk in geo.grid:
                p0 = np.asarray(geo.grid[pk], float)
                write[pk + "_orig"] = p0
                write[pk] = p0 * ratio
    return write


def report_pf(geo, orig, new, info, cfg):
    _hdr("PF profiles  (mode = {}, width = {:g} psi_n{})".format(
        cfg["mode"], cfg["width"],
        ", pf_frac = {:g}".format(cfg["pf_frac"]) if cfg["mode"] == "flat" else ""))
    pf = geo.pf
    for name in orig:
        fsep, sol = info[name]
        o, n = orig[name][pf], new[name][pf]
        print("  {:<6} sep {:10.4g}   PF old {:10.4g} .. {:10.4g}   PF new {:10.4g} .. {:10.4g}"
              .format(name, fsep, o.min(), o.max(), n.min(), n.max()))
        if sol is not None:
            print("         SOLPS scale (grid/SOLPS at OMP sep) = {:.4g}".format(sol[0]))
    j = geo.jyseps1_1
    for name in ("Teexp", "Niexp"):
        if name in orig:
            for lab, F in (("old", orig[name]), ("new", new[name])):
                c = F[:geo.ixseps1, j]
                L = np.abs(c[1:-1] / (0.5 * (c[2:] - c[:-2]) + 1e-300))
                print("  {:<6} {} min L/dx in PF column j={}: {} cells{}".format(
                    name, lab, j, "{:.3g}".format(L.min()) if L.min() < 1e6 else "flat",
                    "   <-- steep: increase width" if lab == "new" and L.min() < 10 else ""))


def plot_pf(geo, orig, new, info, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    jomp, jpf, isep = geo.jomp, geo.jyseps1_1, geo.ixseps1
    p = geo.psinxy
    fig, axs = plt.subplots(1, len(orig), figsize=(4.2 * len(orig), 3.6))
    for ax, name in zip(np.atleast_1d(axs), orig):
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
# main
# ===========================================================================
def main():
    config_path = DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    sibil_config_summary(config)
    jcfg = _settings(config, "jpar", JPAR_DEFAULTS)
    pcfg = _settings(config, "pf_profiles", PF_DEFAULTS)
    if not jcfg["psin_blend_start"] < jcfg["psin_blend_end"] < 1.0:
        raise SystemExit("need psin_blend_start < psin_blend_end < 1.0")

    if not confirm():
        print("Aborted. No changes made.")
        sys.exit(0)

    import sibil  # imported here so summary/confirmation works without dependency installed
    from boututils.datafile import DataFile

    if config["mapping"] == "midplane":
        print("\nRunning SIBIL_Midplane_Mapper...")
        mapper = sibil.SIBIL_Midplane_Mapper(config)
    else:
        print("Aborted. No recognized mapping procedure.")
        sys.exit(0)

    bgrid = config["files"]["bgrid"]
    with DataFile(bgrid) as f:
        grid = {k: f.read(k) for k in f.list()}

    _hdr("Jpar0 + PF correction on {}".format(bgrid))
    geo = Geometry(mapper, grid)
    write = build_jpar(geo, jcfg)
    sep_check(geo)
    write.update(build_pf(geo, pcfg))

    _hdr("output")
    with DataFile(bgrid, write=True) as f:
        for k, v in write.items():
            f.write(k, np.asarray(v, dtype=np.float64))
    print("  wrote {}: {}".format(bgrid, ", ".join(write)))
    print("Done.")
    return mapper


if __name__ == "__main__":
    main()
