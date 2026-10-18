#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sibil_debug_diagnostics.py  (v4)
================================

Drop-in diagnostic module for the SIBIL (SOLPS-ITER <-> elm_6f) interactive
plotter notebook.

What changed in v4
------------------
  * plot_vexb()               NEW -- replaces the noisy V_ExB profile.  2-D map,
                              radial envelope, signed profile split into a
                              coherent part and its 2*dx residual, optional
                              conversion from contravariant V^x to m/s.
  * crash_map()               NEW -- log|f| over (poloidal idx, psi_n) at several
                              times, with X-point cuts, separatrix, PF_limit cut
                              and the blanked PF region drawn on.
  * crash_trace()             NEW -- amplitude per region (core / upstream SOL /
                              leg SOL / leg PF) vs time, growth rate in 1/tau_A,
                              and where the global max sits at every output.
  * spike_locator()           NEW -- radial index of the 2*dx spikes in the legs.
                              Pinned to one index = boundary-condition switch.
  * parallel_structure()      NEW -- amplitude along y on fixed flux surfaces.
  * topology_check()          NEW -- separatrix/X-point indices, cells per leg,
                              absolute Bp/hthe floor map, local-q map, zShift
                              continuity across the PF cut, and a hyperdiffusion
                              coefficient sized to the grid.
  * equilibrium_kink_map()    NEW -- C1-kink indicator for Te0/Ti0/Ni0/Jpar0 over
                              the WHOLE poloidal plane, not just jomp.
  * check_grid_equilibrium()  Bp/hthe floors are ABSOLUTE (T, m) as in the model;
                              the force-balance row (dP/dpsi vs Jpar*Bp/B is not a
                              force balance) is replaced by local q.
  * growth_report()           tracks |f(t) - f(0)| so background-dominated
                              gradients are no longer flat; gamma in 1/tau_A.
  * sawtooth_report()         safe nt; 0.5 (isolated spike) and 1.0
                              (checkerboard) reference levels; no log drop-outs.
  * chi_artifact_report()     finite-only axis bounds (the t=11 figure saved blank).
  * poloidal_localization()   label no longer collides with the title; topology.
  * velocity_consistency()    optional V^x -> m/s conversion via R*Bp.

USAGE
-----
    import importlib, sibil_debug_diagnostics as dbg
    importlib.reload(dbg)

    topo = dbg.topology_check(grid_fname, PF_limit_range=0.98)
    dbg.crash_trace(convergent_transport_coeffs, psin_mid, topo)
    dbg.crash_map(convergent_transport_coeffs, psin_mid, topo, key="Xe_t")
    dbg.spike_locator(convergent_transport_coeffs, psin_mid, topo, key="Xe_t")
    dbg.parallel_structure(convergent_transport_coeffs, psin_mid, topo, key="Xe_t")
    dbg.plot_vexb(convergent_transport_coeffs["Vexbx_txy"], psin_mid, topo,
                  grid=grid_fname, jomp=jomp)
    dbg.equilibrium_kink_map(grid_fname, topo)

    dbg.full_report(convergent_transport_coeffs, psin_mid, grid_fname,
                    jomp=jomp, savedir="./crash_diagnostics")

Hard dependencies: numpy + matplotlib.  ipywidgets, boututils, netCDF4 optional.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

# --------------------------------------------------------------------------- #
# Naming conventions used by the SIBIL notebook
# --------------------------------------------------------------------------- #
COEFF_KEYS = ["Da_t", "Xe_t", "Xi_t"]
GRAD_KEYS = ["ddr_nitot_txy", "ddr_Tetot_txy", "ddr_Titot_txy"]
VEL_KEYS = ["Vr_t", "Vexbx_txy", "Vbtildx_txy"]

DENOM_OF = {
    "Da_t": "ddr_nitot_txy",
    "Xe_t": "ddr_Tetot_txy",
    "Xi_t": "ddr_Titot_txy",
}

LABELS = {
    "Da_t": r"$D$",
    "Xe_t": r"$\chi_e$",
    "Xi_t": r"$\chi_i$",
    "ddr_Tetot_txy": r"$\partial_r T_e$",
    "ddr_Titot_txy": r"$\partial_r T_i$",
    "ddr_nitot_txy": r"$\partial_r n_i$",
    "Vr_t": r"$V_r$",
    "Vexbx_txy": r"$V_{E\times B}$",
    "Vbtildx_txy": r"$\tilde{V}_b$",
}

COLORS = {
    "Da_t": "C0", "Xe_t": "C1", "Xi_t": "C2",
    "ddr_Tetot_txy": "C3", "ddr_Titot_txy": "C4", "ddr_nitot_txy": "C5",
    "Vr_t": "C6", "Vexbx_txy": "C7", "Vbtildx_txy": "C8",
}

REGION_COLORS = {"core": "C0", "SOL upstream": "C2", "leg SOL": "C1",
                 "leg PF": "C3"}

MASK_EPS = 1.0e-2
TINY = 1e-300

# physical constants used to rebuild the model normalisation
MU0 = 4.0e-7 * np.pi
MP = 1.6726e-27


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def get_slice(arr, j, t=0, name="<array>", strict=True):
    """
    Radial profile at poloidal index j (and time t if the array has a time axis).
    Negative indices are python-style; out-of-range raises (strict) or clamps.
    """
    arr = np.asarray(arr)

    def _chk(idx, n, axis):
        if idx is None:
            return 0
        i = int(idx)
        if -n <= i < 0:
            i += n
        if not (0 <= i < n):
            msg = (f"{name}: {axis} index {idx} out of range for axis of "
                   f"length {n} (array shape {arr.shape}). "
                   f"This array probably came from a SHORTER previous run -- "
                   f"re-run bout_save with --purge, or check nt consistency "
                   f"with dbg.check_time_axes().")
            if strict:
                raise IndexError(msg)
            warnings.warn(msg)
            i = min(max(i, 0), n - 1)
        return i

    if arr.ndim == 3:
        return arr[_chk(t, arr.shape[0], "time"), :, _chk(j, arr.shape[2], "poloidal")]
    if arr.ndim == 2:
        return arr[:, _chk(j, arr.shape[1], "poloidal")]
    return arr


def _tidx(t, nt):
    """Resolve a (possibly negative) time index against nt."""
    t = int(t)
    return min(max(nt + t if t < 0 else t, 0), nt - 1)


def _finite_max(a):
    a = np.asarray(a)
    m = np.isfinite(a)
    return np.max(np.abs(a[m])) if m.any() else np.nan


def _finite_minmax(a):
    a = np.asarray(a, dtype=float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return -1.0, 1.0
    lo, hi = float(a.min()), float(a.max())
    if lo == hi:
        lo, hi = lo - 1.0, hi + 1.0
    return lo, hi


def _log10abs(a, floor_decades=10):
    """log10|a| with non-finite -> nan and a floor `floor_decades` below max."""
    a = np.abs(np.asarray(a, dtype=float))
    a = np.where(np.isfinite(a), a, np.nan)
    mx = np.nanmax(a) if np.isfinite(a).any() else 1.0
    return np.log10(np.maximum(a, mx * 10.0 ** (-floor_decades) + TINY))


def _sep_line(ax, x=1.0):
    ax.axvline(x, color="k", lw=0.8, ls="--", alpha=0.6, zorder=0)


def _save(fig, savedir, fname):
    if savedir:
        os.makedirs(savedir, exist_ok=True)
        path = os.path.join(savedir, fname)
        fig.savefig(path, dpi=140, bbox_inches="tight")
        print(f"  saved {path}")


def _show(fig):
    plt.show()
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Shape / provenance consistency
# --------------------------------------------------------------------------- #
def check_time_axes(coeffs, psin=None, data_dir=None, verbose=True):
    """
    Verify every array in `coeffs` shares a leading (time) dimension and report
    the common nt that is safe to index.  Always build sliders from array shapes,
    never from the config scalar `l` (which the SOLPS-pass cell rebinds).
    """
    info, ndim3 = {}, {}
    for k, v in coeffs.items():
        a = np.asarray(v)
        info[k] = a.shape
        if a.ndim == 3:
            ndim3[k] = a.shape[0]

    if psin is not None and np.asarray(psin).ndim == 3:
        ndim3["psin"] = np.asarray(psin).shape[0]

    nts = sorted(set(ndim3.values()))
    nt = min(nts) if nts else 1
    ragged = len(nts) > 1

    ny_vals = {k: s[-1] for k, s in info.items()}
    ny = min(ny_vals.values()) if ny_vals else 1
    ragged_y = len(set(ny_vals.values())) > 1

    if verbose:
        print(f"{'array':<22}{'shape':<22}{'nt':>5}")
        print("-" * 50)
        for k, s in info.items():
            flag = ""
            if k in ndim3 and ragged and ndim3[k] != max(nts):
                flag = "  <-- SHORT (stale?)"
            print(f"{k:<22}{str(s):<22}{ndim3.get(k,'-'):>5}{flag}")
        print("-" * 50)
        if ragged:
            print(f"!! RAGGED TIME AXIS: nt values present = {nts}")
            print(f"   Sliders will be limited to nt = {nt}.")
            print(f"   Fix:  python bout_save_v3.py <run_dir> --purge")
        else:
            print(f"OK: all time-dependent arrays have nt = {nt}")
        if ragged_y:
            print(f"!! RAGGED POLOIDAL AXIS: {sorted(set(ny_vals.values()))}")

    if data_dir is not None:
        _report_staleness(coeffs, data_dir)

    return {"nt": nt, "nt_by_key": ndim3, "ny": ny,
            "ragged": ragged or ragged_y, "shapes": info}


def _report_staleness(coeffs, data_dir):
    """Flag any numpy/*.npy backing these arrays that predates the dump files."""
    import glob
    from pathlib import Path

    d = Path(data_dir)
    run = d.parent if d.name == "numpy" else d
    dumps = sorted(glob.glob(str(run / "BOUT.dmp.*.nc")))
    if not dumps:
        return
    newest = max(os.path.getmtime(f) for f in dumps)

    stale = []
    for k in coeffs:
        p = d / f"{k}.npy"
        if p.exists() and os.path.getmtime(p) < newest:
            stale.append((k, (newest - os.path.getmtime(p)) / 3600.0))
    if stale:
        print("\n!! STALE .npy FILES (older than the newest BOUT.dmp):")
        for k, age in sorted(stale, key=lambda r: -r[1]):
            print(f"   {k:<22} older by {age:7.2f} h")


def harmonise(coeffs, nt=None, verbose=True):
    """Truncate every time-dependent array to a common nt."""
    if nt is None:
        nt = check_time_axes(coeffs, verbose=False)["nt"]
    out = {k: (np.asarray(v)[:nt] if np.asarray(v).ndim == 3 else v)
           for k, v in coeffs.items()}
    if verbose:
        print(f"harmonise: all time-dependent arrays truncated to nt = {nt}")
    return out


def n_y_max(coeffs):
    return min(np.asarray(v).shape[-1] for v in coeffs.values())


def _safe_nt(coeffs, keys):
    nts = [np.asarray(coeffs[k]).shape[0] for k in keys
           if np.asarray(coeffs[k]).ndim == 3]
    return min(nts) if nts else 1


# --------------------------------------------------------------------------- #
# chi masking
# --------------------------------------------------------------------------- #
def gradient_mask(denom, eps=MASK_EPS):
    """True where |grad| > eps*max, i.e. where chi = q/(n grad T) is defined."""
    denom = np.asarray(denom, dtype=float)
    scale = _finite_max(denom)
    if not np.isfinite(scale) or scale == 0.0:
        return np.zeros_like(denom, dtype=bool)
    return np.abs(denom) > eps * scale


def masked_profile(coeffs, name, j, t=0, eps=MASK_EPS):
    y = np.array(get_slice(coeffs[name], j, t), dtype=float)
    dkey = DENOM_OF.get(name)
    if dkey is None or dkey not in coeffs:
        return y, np.ones_like(y, dtype=bool)
    m = gradient_mask(get_slice(coeffs[dkey], j, t), eps)
    return np.where(m, y, np.nan), m


# =========================================================================== #
# Grid I/O and topology
# =========================================================================== #
def _open_grid(grid_or_path):
    """Accept an open DataFile-like object, a dict, or a path."""
    if hasattr(grid_or_path, "__getitem__") and not isinstance(grid_or_path, str):
        return grid_or_path, (lambda g, k: np.asarray(g[k]))
    try:
        from boututils.datafile import DataFile
        g = DataFile(str(grid_or_path))
        return g, (lambda g, k: np.asarray(g[k]))
    except Exception:
        import netCDF4 as nc
        g = nc.Dataset(str(grid_or_path))
        return g, (lambda g, k: np.asarray(g.variables[k][:]))


def _try(reader, g, *names):
    for n in names:
        try:
            v = reader(g, n)
            if v is not None:
                return np.asarray(v), n
        except Exception:
            continue
    return None, None


def _scalar(reader, g, name, default=None):
    v, _ = _try(reader, g, name)
    return default if v is None else float(np.asarray(v).ravel()[0])


def load_topology(grid_or_path, PF_limit_range=0.98, PF_limit=True):
    """
    Read separatrix / X-point indices exactly as elm_6f does and build the
    region masks the model uses.

    Region definitions follow PF_filter() / sink_PF() in the .cxx:
      leg rows   : j <= jyseps1_1, jyseps2_1 < j <= jyseps1_2, j > jyseps2_2
      leg PF     : leg rows with ix <  ixseps1
      PF blanked : leg rows with ix <  PF_limit_range*ixseps1   (PF_filter)
      SBC switch : leg rows, ix <= PF_limit_range*ixseps1 get zero BCs instead
                   of the sheath BC (SBC_* functions)
    """
    g, R = _open_grid(grid_or_path)
    nx = int(_scalar(R, g, "nx"))
    ny = int(_scalar(R, g, "ny"))
    ixsep = int(_scalar(R, g, "ixseps1", nx))
    ixsep2 = int(_scalar(R, g, "ixseps2", nx))
    j11 = int(_scalar(R, g, "jyseps1_1", -1))
    j21 = int(_scalar(R, g, "jyseps2_1", ny // 2))
    j12 = int(_scalar(R, g, "jyseps1_2", ny // 2))
    j22 = int(_scalar(R, g, "jyseps2_2", ny - 1))

    jj = np.arange(ny)
    leg_y = (jj <= j11) | ((jj > j21) & (jj <= j12)) | (jj > j22)
    ix = np.arange(nx)[:, None]
    leg = np.broadcast_to(leg_y[None, :], (nx, ny))
    pf = leg & (ix < ixsep)
    # PF_filter only zeroes the dumped PF cells when PF_limit = true
    blank = (leg & (ix < PF_limit_range * ixsep)) if PF_limit \
        else np.zeros((nx, ny), dtype=bool)
    sol = ix >= ixsep
    region = np.full((nx, ny), "core", dtype=object)
    region[sol & ~leg] = "SOL upstream"
    region[sol & leg] = "leg SOL"
    region[pf] = "leg PF"

    cut_ix = int(np.ceil(PF_limit_range * ixsep))
    topo = dict(nx=nx, ny=ny, ixsep=ixsep, ixsep2=ixsep2,
                jyseps1_1=j11, jyseps2_1=j21, jyseps1_2=j12, jyseps2_2=j22,
                leg_y=leg_y, pf=pf, blank=blank, region=region,
                PF_limit=PF_limit, PF_limit_range=PF_limit_range, pf_cut_ix=cut_ix,
                n_leg_inner=int(j11 + 1), n_leg_outer=int(ny - 1 - j22),
                n_leg_upper=int(max(j12 - j21, 0)))
    return topo


def _xpt_cuts(topo):
    cuts = [topo["jyseps1_1"] + 0.5, topo["jyseps2_2"] + 0.5]
    if topo["jyseps1_2"] > topo["jyseps2_1"]:
        cuts += [topo["jyseps2_1"] + 0.5, topo["jyseps1_2"] + 0.5]
    return cuts


def _psi_at(psin, ixf):
    """psi_n at fractional radial index ixf (cell-centre interpolation)."""
    psin = np.asarray(psin)
    return float(np.interp(ixf, np.arange(psin.size), psin))


def _overlay_topology(ax, topo, psin, show_cut=True):
    """X-point cuts, separatrix and PF_limit cut on a (j, psi_n) axis."""
    if topo is None:
        return
    for c in _xpt_cuts(topo):
        ax.axvline(c, color="w", lw=1.0, ls="--", alpha=0.9)
        ax.axvline(c, color="k", lw=0.5, ls="--", alpha=0.9)
    psep = _psi_at(psin, topo["ixsep"] - 0.5)
    ax.axhline(psep, color="w", lw=1.0)
    ax.axhline(psep, color="k", lw=0.5)
    if show_cut and topo.get("PF_limit", True):
        pcut = _psi_at(psin, topo["PF_limit_range"] * topo["ixsep"] - 0.5)
        ny = topo["ny"]
        for lo, hi in _leg_spans(topo):
            ax.hlines(pcut, lo - 0.5, hi + 0.5, colors="m", lw=1.4)
        ax.text(0.01, pcut, "PF_limit cut", color="m", fontsize=7,
                va="top", ha="left", transform=ax.get_yaxis_transform())
        ax.set_xlim(-0.5, ny - 0.5)


def _leg_spans(topo):
    j = np.flatnonzero(topo["leg_y"])
    if j.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(j) > 1)
    starts = np.r_[j[0], j[breaks + 1]]
    ends = np.r_[j[breaks], j[-1]]
    return list(zip(starts, ends))


# =========================================================================== #
# Q1.  The chi spike is a 0/0 artifact
# =========================================================================== #
def chi_artifact_report(coeffs, psin, j=44, t=12, eps=MASK_EPS, savedir=None):
    """
    col 1 raw chi, col 2 the denominator with the |grad| < eps*max band shaded,
    col 3 masked chi.  If col 3 is smooth the spike was 0/0.
    """
    x = np.asarray(psin)
    names = [k for k in COEFF_KEYS if k in coeffs]
    t = _tidx(t, _safe_nt(coeffs, names))

    fig, axs = plt.subplots(len(names), 3, figsize=(13, 2.5 * len(names)),
                            sharex=True, squeeze=False)

    for row, name in enumerate(names):
        y_raw = np.asarray(get_slice(coeffs[name], j, t), dtype=float)
        dkey = DENOM_OF[name]
        d = (np.asarray(get_slice(coeffs[dkey], j, t), dtype=float)
             if dkey in coeffs else np.ones_like(y_raw))
        m = gradient_mask(d, eps)
        y_msk = np.where(m, y_raw, np.nan)

        a0, a1, a2 = axs[row]
        a0.plot(x, np.where(np.isfinite(y_raw), y_raw, np.nan),
                color=COLORS[name], lw=1.0)
        a0.set_ylabel(f"{LABELS[name]}  raw", fontsize=12)

        lo, hi = _finite_minmax(d)
        a1.plot(x, np.where(np.isfinite(d), d, np.nan), color=COLORS[dkey], lw=1.0)
        a1.fill_between(x, lo, hi, where=~m, color="red", alpha=0.18,
                        step="mid", label=f"|grad| < {eps:g} max")
        a1.set_ylim(lo, hi)
        a1.set_ylabel(LABELS[dkey], fontsize=12)
        if row == 0:
            a1.legend(fontsize=7, loc="lower left")

        a2.plot(x, y_msk, color=COLORS[name], lw=1.2)
        a2.set_ylabel(f"{LABELS[name]}  masked", fontsize=12)

        for a in (a0, a1, a2):
            _sep_line(a)
            a.grid(True, alpha=0.3)
            a.tick_params(labelsize=7)
            a.yaxis.get_offset_text().set_fontsize(7)

        nbad = int((~np.isfinite(y_raw)).sum() + (~np.isfinite(d)).sum())
        if nbad:
            print(f"[{name}] {nbad} non-finite cell(s) at t={t}, j={j}")
        if (~m).sum():
            ratio = _finite_max(y_raw) / (_finite_max(y_msk) + TINY)
            print(f"[{name}] {(~m).sum()} masked cell(s); peak drops by "
                  f"x{ratio:.1f} when the vanishing-gradient cells are removed.")

    for a in axs[-1]:
        a.set_xlabel(r"$\psi_n$", fontsize=12)
    fig.suptitle(f"chi 0/0 artifact check   |   pol idx {j}, time idx {t}", y=1.01)
    _save(fig, savedir, f"chi_artifact_j{j}_t{t}.png")
    _show(fig)
    return fig


# =========================================================================== #
# Q2.  Equilibrium / grid-file sanity at one poloidal index
# =========================================================================== #
def check_grid_equilibrium(grid_or_path, jomp=44, eps_dx=0.05,
                           bpxy_constraint=0.04, hthe_constraint=0.04,
                           savedir=None):
    """
    Rows at poloidal index `jomp`:
      1 Te0, Ti0, Ni0 (norm.)     2 d/dpsi     3 d2/dpsi2 (C1 kinks)
      4 Jpar0 and Jpar0''         5 dx and |d(dx)|/dx
      6 Bpxy [T] and hthe [m] against the ABSOLUTE limit_jacobi floors
        (field_larger(Bpxy, bpxy_constraint) is applied in tesla, before
        normalisation -- it is not a fraction of max Bp)
      7 local q = hthe*Bt/Bp (what local_q=true feeds to the flux limiter and
        to chi_neo ~ q^2)
    """
    g, R = _open_grid(grid_or_path)

    psixy, _ = _try(R, g, "psixy")
    psi_axis, _ = _try(R, g, "psi_axis")
    psi_bdry, _ = _try(R, g, "psi_bdry", "psi_bndry")
    if psixy is None:
        raise KeyError("grid file has no psixy")
    x = ((psixy - psi_axis) / (psi_bdry - psi_axis))[:, jomp]

    def col(*names):
        v, used = _try(R, g, *names)
        if v is None:
            return None, None
        return (v[:, jomp] if v.ndim == 2 else v), used

    Te0, nTe = col("Teexp", "Te0", "Te_exp")
    Ti0, nTi = col("Tiexp", "Ti0", "Ti_exp")
    Ni0, nNi = col("Niexp", "Ni0", "Ne0", "Niexp0")
    Jpar0, nJ = col("Jpar0", "jpar0")
    dx, _ = col("dx")
    Bpxy, _ = col("Bpxy")
    Btxy, _ = col("Btxy")
    hthe, _ = col("hthe")

    print("grid variables used:", {"Te0": nTe, "Ti0": nTi, "Ni0": nNi, "Jpar0": nJ})

    def d1(y):
        return np.gradient(y, x, edge_order=2)

    def d2(y):
        return np.gradient(d1(y), x, edge_order=2)

    fig, axs = plt.subplots(7, 1, figsize=(7.5, 14.0), sharex=True)
    prof = ((Te0, r"$T_{e0}$", "Te0", "C3"), (Ti0, r"$T_{i0}$", "Ti0", "C4"),
            (Ni0, r"$n_{i0}$", "ni0", "C5"))

    for y, lab, _, c in prof:
        if y is not None:
            axs[0].plot(x, y / _finite_max(y), c, lw=1.2, label=lab)
            dy = d1(y)
            axs[1].plot(x, dy / (_finite_max(dy) + TINY), c, lw=1.2, label=lab + "'")
    axs[0].set_ylabel("profile\n(norm.)", fontsize=12)
    axs[1].set_ylabel(r"$d/d\psi_n$" + "\n(norm.)", fontsize=12)

    worst = {}
    for y, lab, name, c in prof:
        if y is not None:
            ddy = d2(y)
            s = _finite_max(ddy) + TINY
            axs[2].plot(x, ddy / s, c, lw=1.2, label=lab + "''")
            near = np.abs(x - 1.0) < 0.03
            worst[name] = (np.nanmax(np.abs(ddy[near])) / s) if near.any() else np.nan
    axs[2].set_ylabel(r"$d^2/d\psi_n^2$" + "\n(norm.)", fontsize=12)
    for k, v in worst.items():
        flag = "  <-- SPIKE AT SEPARATRIX" if v > 0.5 else ""
        print(f"  |{k}''| within |psi_n-1|<0.03 reaches {v:.2f} of its global max{flag}")

    ax = axs[3]
    if Jpar0 is not None:
        ax.plot(x, Jpar0 / (_finite_max(Jpar0) + TINY), "k", lw=1.2, label=r"$J_{\|0}$")
        dd = d2(Jpar0)
        ax.plot(x, dd / (_finite_max(dd) + TINY), "0.6", lw=1.0, label=r"$J_{\|0}''$")
    else:
        ax.text(0.5, 0.5, "Jpar0 not in grid file", ha="center", transform=ax.transAxes)
    ax.set_ylabel("Jpar0\n(norm.)", fontsize=12)

    ax = axs[4]
    if dx is not None:
        ax.plot(x, dx / _finite_max(dx), "C0", lw=1.2, label=r"$dx$ (norm.)")
        nu = np.abs(np.gradient(dx)) / (np.abs(dx) + TINY)
        ax.plot(x, nu, "C1", lw=1.0, label=r"$|\Delta(dx)|/dx$")
        ax.axhline(eps_dx, color="r", ls=":", lw=0.9)
        print(f"  dx max/min = {_finite_max(dx) / np.nanmin(np.abs(dx)):.1f}; "
              f"|d(dx)|/dx max {np.nanmax(nu):.3f} at psi_n={x[np.nanargmax(nu)]:.4f}")
        if np.any(np.diff(np.sign(np.diff(x))) != 0):
            print("  -> WARNING: psi_n is non-monotonic along x at this poloidal index.")
    ax.set_ylabel("grid\nspacing", fontsize=12)

    ax = axs[5]
    if Bpxy is not None:
        ax.plot(x, Bpxy, "C2", lw=1.2, label=r"$B_p$ [T]")
        ax.axhline(bpxy_constraint, color="C2", ls=":", lw=0.9,
                   label=f"bpxy_constraint = {bpxy_constraint} T")
        print(f"  Bpxy below the {bpxy_constraint} T floor in "
              f"{int((Bpxy < bpxy_constraint).sum())} cells at j={jomp} "
              f"(use topology_check for the whole plane).")
    if hthe is not None:
        ax.plot(x, hthe, "C6", lw=1.2, label=r"$h_\theta$ [m]")
        ax.axhline(hthe_constraint, color="C6", ls=":", lw=0.9,
                   label=f"hthe_constraint = {hthe_constraint} m")
    ax.set_ylabel("metrics\n(abs.)", fontsize=12)

    ax = axs[6]
    if Bpxy is not None and Btxy is not None and hthe is not None:
        qloc = np.abs(hthe * Btxy / np.maximum(Bpxy, bpxy_constraint))
        ax.plot(x, qloc, "C9", lw=1.2, label=r"$q_{loc}=h_\theta B_t/B_p$")
        ax.set_ylabel("local q", fontsize=12)
    else:
        ax.text(0.5, 0.5, "need hthe, Btxy, Bpxy", ha="center", transform=ax.transAxes)

    for a in axs:
        _sep_line(a)
        a.grid(True, alpha=0.3)
        a.tick_params(labelsize=7)
        if a.get_legend_handles_labels()[0]:
            a.legend(fontsize=7, ncol=3)
    axs[-1].set_xlabel(r"$\psi_n$", fontsize=12)
    axs[-1].set_xlim(max(x.min(), 0.85), min(x.max(), 1.10))
    fig.suptitle(f"Equilibrium / grid check at poloidal index {jomp}", y=1.0)
    fig.tight_layout()
    _save(fig, savedir, f"grid_equilibrium_j{jomp}.png")
    _show(fig)
    return fig


# =========================================================================== #
# Q2b.  Whole-plane grid / topology check  (NEW)
# =========================================================================== #
def topology_check(grid_or_path, PF_limit_range=0.98, PF_limit=True, bpxy_constraint=0.04,
                   hthe_constraint=0.04, ZPERIOD=5, growth_rate=None,
                   savedir=None):
    """
    Everything about the grid that only shows up away from the midplane.

    Prints
      * ixseps1, jyseps*, poloidal cells in each divertor leg
      * the PF_limit cut index and its psi_n -- inside that cut PF_filter zeroes
        the DUMPED fields and the sheath BC is swapped for zero BCs
      * cells where Bp < bpxy_constraint [T] / hthe < hthe_constraint [m]
      * zShift continuity across the PF connection (j=jyseps1_1 <-> jyseps2_2+1)
      * smallest radial cell (normalised to rmag) and the hyperdiff_perp_*4
        coefficient that damps the 2*dx mode at a chosen rate

    Figure: Bp with the floor contour, local q, zShift, radial cell size.
    Returns the topology dict used by the crash plots.
    """
    topo = load_topology(grid_or_path, PF_limit_range, PF_limit)
    g, R = _open_grid(grid_or_path)
    nx, ny, ixsep = topo["nx"], topo["ny"], topo["ixsep"]

    psixy, _ = _try(R, g, "psixy")
    psi_axis, _ = _try(R, g, "psi_axis")
    psi_bdry, _ = _try(R, g, "psi_bdry", "psi_bndry")
    psin2d = (psixy - psi_axis) / (psi_bdry - psi_axis)
    Bp, _ = _try(R, g, "Bpxy")
    Bt, _ = _try(R, g, "Btxy")
    hthe, _ = _try(R, g, "hthe")
    Rxy, _ = _try(R, g, "Rxy")
    dx, _ = _try(R, g, "dx")
    zS, _ = _try(R, g, "zShift", "qinty")
    lbar = _scalar(R, g, "rmag", 1.0)

    print("=" * 72)
    print("TOPOLOGY")
    print("=" * 72)
    print(f"  nx={nx} ny={ny} ixseps1={ixsep} ixseps2={topo['ixsep2']}")
    print(f"  jyseps1_1={topo['jyseps1_1']} jyseps2_1={topo['jyseps2_1']} "
          f"jyseps1_2={topo['jyseps1_2']} jyseps2_2={topo['jyseps2_2']}")
    print(f"  poloidal cells: inner leg {topo['n_leg_inner']}, outer leg "
          f"{topo['n_leg_outer']}" +
          (f", upper legs {topo['n_leg_upper']}" if topo["n_leg_upper"] else ""))
    if min(topo["n_leg_inner"], topo["n_leg_outer"]) < 12:
        print("  -> FEWER THAN 12 CELLS IN A LEG.  C2 parallel derivatives plus a "
              "sheath BC on that few points is under-resolved.")
    cut = topo["pf_cut_ix"]
    psi_mid = psin2d[:, ny // 2] if psin2d is not None else None
    if psi_mid is not None:
        print(f"  PF_limit cut: ix >= {cut} (= ceil({PF_limit_range}*{ixsep})), "
              f"psi_n = {_psi_at(psi_mid, PF_limit_range * ixsep - 0.5):.4f}; "
              f"{ixsep - cut} PF cell(s) between the cut and the separatrix.")
    print("  PF_filter() overwrites Ni,Ti,Te,U,P,phi,Jpar,Apar but NOT ddt():")
    print("  the PF region keeps evolving in CVODE, it is only zeroed in the dumps.")
    for w in (0.01, 0.02):
        print(f"  PF_sink damping that fades out before the separatrix: "
              f"PFs_width = {w}, PFs_length = {ixsep / nx - 3 * w:.3f}")

    if Bp is not None:
        nb = (Bp < bpxy_constraint)
        print(f"  Bp < {bpxy_constraint} T in {int(nb.sum())} cells "
              f"(j = {sorted(set(np.nonzero(nb)[1].tolist()))[:12]}...)")
    if hthe is not None:
        nh = (hthe < hthe_constraint)
        print(f"  hthe < {hthe_constraint} m in {int(nh.sum())} cells")

    qloc = None
    if Bp is not None and Bt is not None and hthe is not None:
        qloc = np.abs(hthe * Bt / np.maximum(Bp, bpxy_constraint))
        core = ~topo["leg_y"]
        print(f"  local q: core rows median {np.median(qloc[:, core]):.2f}, "
              f"leg rows max {np.max(qloc[:, topo['leg_y']]):.2f} "
              f"-> chi_neo ~ q^2 is x{(np.max(qloc[:, topo['leg_y']]) / np.median(qloc[:, core]))**2:.0f} "
              f"larger in the legs")

    if zS is not None and topo["jyseps1_1"] >= 0:
        j1, j2 = topo["jyseps1_1"], topo["jyseps2_2"] + 1
        if 1 <= j1 and j2 + 1 < ny:
            # j1 and j2 are neighbours along a PF field line, so the step across
            # the cut should look like the steps on either side of it.
            pf_ix = np.arange(min(ixsep, nx))
            step_pf = np.abs(zS[pf_ix, j2] - zS[pf_ix, j1])
            normal = 0.5 * (np.abs(zS[pf_ix, j1] - zS[pf_ix, j1 - 1]) +
                            np.abs(zS[pf_ix, j2 + 1] - zS[pf_ix, j2]))
            r = step_pf / (normal + TINY)
            ratio = float(np.max(r))
            print(f"  zShift step across PF connection (j={j1}->{j2}) / adjacent "
                  f"steps: max {ratio:.1f} at ix={int(np.argmax(r))}")
            if ratio > 5:
                print("  -> zShift is DISCONTINUOUS across the PF branch cut. "
                      "Field lines in the PF region are mis-connected.")

    dr = None
    if dx is not None and Rxy is not None and Bp is not None:
        dr = np.abs(dx) / (Rxy * np.maximum(Bp, bpxy_constraint)) / lbar
        upstream = ~topo["leg_y"]
        dr_min = np.min(dr[:, upstream])
        print(f"  smallest radial cell (upstream): {dr_min * lbar * 1e3:.3f} mm "
              f"= {dr_min:.3e} Lbar")
        rates = [growth_rate] if growth_rate else [0.1, 1.0, 10.0]
        for r in rates:
            d4 = r * dr_min ** 4 / 16.0
            print(f"    hyperdiff_perp_*4 = {d4:.2e} damps the 2*dx mode at "
                  f"{r:g} / tau_A   (k=kmax/2 damped 16x slower)")

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    jx = np.arange(ny)
    yv = psi_mid if psi_mid is not None else np.arange(nx)

    def _map(ax, z, title, **kw):
        pc = ax.pcolormesh(jx, yv, z, shading="nearest", **kw)
        fig.colorbar(pc, ax=ax, pad=0.01)
        _overlay_topology(ax, topo, yv)
        ax.set_title(title, fontsize=9)

    if Bp is not None:
        _map(axs[0, 0], np.log10(np.maximum(Bp, 1e-4)), r"$\log_{10} B_p$ [T]", cmap="viridis")
        axs[0, 0].contour(jx, yv, Bp, levels=[bpxy_constraint], colors="r", linewidths=1.0)
    if qloc is not None:
        _map(axs[0, 1], np.log10(qloc), r"$\log_{10} q_{loc}$ (floored $B_p$)", cmap="magma")
    if zS is not None:
        _map(axs[1, 0], zS, "zShift [rad]", cmap="twilight")
    if dr is not None:
        _map(axs[1, 1], np.log10(dr * lbar * 1e3), r"$\log_{10}\Delta r$ [mm]", cmap="cividis")
    for a in axs[-1]:
        a.set_xlabel("poloidal index")
    for a in axs[:, 0]:
        a.set_ylabel(r"$\psi_n$ (midplane map)")
    fig.suptitle("Grid topology: red = Bp floor contour, dashed = X-point cuts, "
                 "magenta = PF_limit cut", fontsize=10)
    fig.tight_layout()
    _save(fig, savedir, "topology_check.png")
    _show(fig)
    return topo


def equilibrium_kink_map(grid_or_path, topo=None, names=None, savedir=None):
    """
    Scale-free C1-kink indicator over the whole (x, y) plane:

        K = |f[i+1] - 2 f[i] + f[i-1]| / ( |f[i+1] - f[i-1]|/2 + 1e-3*|f|max )

    K ~ 0 for smooth profiles, K -> O(1) at a corner.  Computed radially (in x)
    and poloidally (in y), so kinks from the SOLPS -> BOUT mapping in the legs and
    at the X-point cuts are visible, not just at jomp.
    """
    g, R = _open_grid(grid_or_path)
    topo = topo or load_topology(grid_or_path)
    names = names or [("Teexp", "Te0"), ("Tiexp", "Ti0"), ("Niexp", "Ni0"),
                      ("Jpar0", "jpar0"),
                      # geometry: a poloidal kink here at the same j as a Jpar0
                      # kink means the grid segments were stitched there
                      ("Bpxy",), ("hthe",), ("Rxy",), ("Btxy",), ("Bxy",),
                      ("dy",), ("bxcvz",)]
    psixy, _ = _try(R, g, "psixy")
    psi_axis, _ = _try(R, g, "psi_axis")
    psi_bdry, _ = _try(R, g, "psi_bdry", "psi_bndry")
    yv = ((psixy - psi_axis) / (psi_bdry - psi_axis))[:, topo["ny"] // 2]
    jx = np.arange(topo["ny"])

    fields = []
    for alts in names:
        v, used = _try(R, g, *alts)
        if v is not None and v.ndim == 2:
            fields.append((used, v))

    def kink(f, axis):
        f = np.asarray(f, dtype=float)
        fp = np.roll(f, -1, axis)
        fm = np.roll(f, 1, axis)
        k = np.abs(fp - 2 * f + fm) / (0.5 * np.abs(fp - fm) + 1e-3 * _finite_max(f) + TINY)
        sl = [slice(None)] * 2
        sl[axis] = [0, -1]
        k[tuple(sl)] = 0.0
        return k

    fig, axs = plt.subplots(len(fields), 2, figsize=(12, 3.0 * len(fields)),
                            sharex=True, sharey=True, squeeze=False)
    for row, (name, f) in enumerate(fields):
        kx, ky = kink(f, 0), kink(f, 1)
        # the X-point cut is a topological jump in y -- do not score it
        for c in _xpt_cuts(topo):
            for jc in (int(c - 0.5), int(c + 0.5)):
                if 0 <= jc < topo["ny"]:
                    ky[:, jc] = 0.0
        for col_, (k, lab) in enumerate(((kx, "radial"), (ky, "poloidal"))):
            ax = axs[row, col_]
            pc = ax.pcolormesh(jx, yv, np.clip(k, 0, 2), shading="nearest",
                               cmap="inferno", vmin=0, vmax=2)
            fig.colorbar(pc, ax=ax, pad=0.01)
            _overlay_topology(ax, topo, yv, show_cut=False)
            ax.set_title(f"{name}: {lab} kink indicator", fontsize=9)
            imax = np.unravel_index(np.argmax(k), k.shape)
            print(f"  {name:<7} {lab:<9} max K = {k[imax]:.2f} at ix={imax[0]} "
                  f"(psi_n~{yv[imax[0]]:.4f}), j={imax[1]} [{topo['region'][imax]}]")
    for a in axs[-1]:
        a.set_xlabel("poloidal index")
    for a in axs[:, 0]:
        a.set_ylabel(r"$\psi_n$")
    fig.tight_layout()
    _save(fig, savedir, "equilibrium_kink_map.png")
    _show(fig)
    return fig


# =========================================================================== #
# Q3a.  Grid-scale (2*dx) content
# =========================================================================== #
def odd_even_amplitude(y):
    """
    |[-1, 2, -1]/4 * y|.  For a pure (-1)^i checkerboard of amplitude A this
    returns A (fraction 1.0).  For a single-cell spike of height A it returns
    A/2 at the spike (fraction 0.5).  A plateau at exactly 0.5 therefore means
    the profile max is an isolated one-cell spike.
    """
    y = np.asarray(y, dtype=float)
    out = np.zeros_like(y)
    out[1:-1] = (-y[:-2] + 2.0 * y[1:-1] - y[2:]) / 4.0
    out[0], out[-1] = out[1], out[-2]
    return np.abs(out)


def sawtooth_report(coeffs, psin, j=44, keys=None, savedir=None):
    """
    Left : radial profile of the 2*dx fraction at the last two outputs.
    Right: max 2*dx fraction vs time.  0.5 = single-cell spike, 1.0 = checkerboard.
    """
    keys = keys or [k for k in COEFF_KEYS + VEL_KEYS if k in coeffs]
    x = np.asarray(psin)
    nt = _safe_nt(coeffs, keys)

    fig, axs = plt.subplots(len(keys), 2, figsize=(11, 1.9 * len(keys)), squeeze=False)

    for row, name in enumerate(keys):
        arr = np.asarray(coeffs[name])
        a0, a1 = axs[row]

        for t, al in ((nt - 2, 0.5), (nt - 1, 1.0)):
            if t < 0:
                continue
            y = get_slice(arr, j, t)
            frac = odd_even_amplitude(y) / (_finite_max(y) + TINY)
            a0.semilogy(x, np.maximum(frac, 1e-12), color=COLORS.get(name, "k"),
                        alpha=al, lw=1.1, label=f"t={t}")
        a0.set_ylabel(f"{LABELS.get(name, name)}\n2dx frac.", fontsize=9)
        a0.legend(fontsize=6)
        _sep_line(a0)

        tr = np.array([np.nanmax(odd_even_amplitude(get_slice(arr, j, t))) /
                       (_finite_max(get_slice(arr, j, t)) + TINY) for t in range(nt)])
        a1.semilogy(np.arange(nt), tr, "o-", color=COLORS.get(name, "k"), ms=3)
        for lvl, txt in ((0.5, "single-cell spike"), (1.0, "checkerboard")):
            a1.axhline(lvl, color="0.4", ls=":", lw=0.8)
            a1.text(0.0, lvl, txt, fontsize=6, va="bottom", color="0.3")
        a1.set_ylabel("max 2dx frac.", fontsize=9)
        plateau = np.abs(tr[nt // 2:] - 0.5) < 0.02
        if plateau.all() and plateau.size:
            a1.set_title(f"{LABELS.get(name, name)}: pinned at 0.5 -> max is an "
                         f"isolated one-cell spike", fontsize=8)

        for a in (a0, a1):
            a.grid(True, alpha=0.3)
            a.tick_params(labelsize=7)

    axs[-1][0].set_xlabel(r"$\psi_n$", fontsize=12)
    axs[-1][1].set_xlabel("time index", fontsize=12)
    fig.suptitle(f"2$\\Delta x$ content, poloidal index {j}", y=1.0)
    fig.tight_layout()
    _save(fig, savedir, f"sawtooth_j{j}.png")
    _show(fig)
    return fig


# =========================================================================== #
# Q3b.  Growth tracking
# =========================================================================== #
def _fit_gamma(tr, timestep, tfit=None):
    """Least-squares exponential growth rate [1/tau_A] over the time indices tfit."""
    tr = np.asarray(tr, dtype=float)
    nt = tr.size
    idx = np.arange(nt) if tfit is None else np.asarray(tfit)
    ok = idx[np.isfinite(tr[idx]) & (tr[idx] > 0)]
    if ok.size < 3:
        return np.nan
    return np.polyfit(ok * timestep, np.log(tr[ok]), 1)[0]


def growth_report(coeffs, psin, keys=None, timestep=10.0, perturbation=True,
                  savedir=None):
    """
    Semilog max|f(t) - f(0)| (perturbation=True) over the whole plane vs time.
    The gradient fields include the background, so max|f| alone stays flat.
    Growth rate fitted over the last third of the run, in 1/tau_A.
    """
    keys = keys or [k for k in COEFF_KEYS + GRAD_KEYS + VEL_KEYS if k in coeffs]
    x = np.asarray(psin)
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4))

    print(f"{'field':16s} {'first bad t':>12s} {'gamma [1/tauA]':>15s} "
          f"{'peak @ last t':>30s}")
    print("-" * 78)
    for name in keys:
        arr = np.asarray(coeffs[name], dtype=float)
        if arr.ndim != 3:
            continue
        nt = arr.shape[0]
        d = arr - arr[0] if perturbation else arr
        amp = np.array([_finite_max(d[t]) for t in range(nt)])
        if perturbation:
            amp[0] = np.nan
        c = COLORS.get(name, None)
        ax0.semilogy(np.arange(nt), amp, "o-", ms=3, color=c, label=LABELS.get(name, name))
        ref = amp[np.isfinite(amp) & (amp > 0)]
        if ref.size:
            ax1.semilogy(np.arange(nt), amp / ref[0], "o-", ms=3, color=c,
                         label=LABELS.get(name, name))

        gam = _fit_gamma(amp, timestep, np.arange(nt - max(nt // 3, 3), nt))
        bad_t = next((t for t in range(nt) if (~np.isfinite(arr[t])).any()), "-")
        last = d[-1]
        if np.isfinite(last).any():
            ix = np.unravel_index(np.nanargmax(np.abs(np.where(np.isfinite(last), last, np.nan))),
                                  last.shape)
            where = f"psi_n {x[ix[0]]:.4f} (ix={ix[0]}, iy={ix[1]})"
        else:
            where = "all non-finite"
        print(f"{name:16s} {str(bad_t):>12s} {gam:15.3f} {where:>30s}")

    ttl = "max|f - f(t=0)|" if perturbation else "max|f|"
    for a, sub in ((ax0, "absolute"), (ax1, "normalised to first output")):
        a.set_xlabel("time index", fontsize=12)
        a.grid(True, alpha=0.3, which="both")
        a.set_title(f"{ttl} vs time ({sub})", fontsize=9)
        a.tick_params(axis="both", which="both", labelsize=7)
        a.legend(fontsize=7, ncol=2)
    ax0.set_ylabel(ttl, fontsize=12)
    fig.tight_layout()
    _save(fig, savedir, "growth.png")
    _show(fig)
    return fig


# =========================================================================== #
# Q3c.  Crash localisation  (NEW)
# =========================================================================== #
def crash_trace(coeffs, psin, topo, keys=None, timestep=10.0, perturbation=True,
                savedir=None):
    """
    THE crash plot.  For each field, max|f - f(0)| inside each region
      core | SOL upstream | leg SOL | leg PF (outside the PF_limit blanking)
    vs time, with a growth rate per region; plus the (ix, iy) of the global max
    at every output.  If the leg curves sit orders of magnitude above core and
    the max never leaves the legs, the crash is a divertor-leg numerical mode.
    """
    keys = keys or [k for k in COEFF_KEYS + ["Vexbx_txy"] if k in coeffs]
    x = np.asarray(psin)
    region = topo["region"]
    # The SIBIL arrays (Vexb, ddr_*, fluxes -> D, chi) are computed inside the
    # RHS BEFORE PF_filter runs, so they are NOT blanked even with PF_limit=true:
    # they show the solver's real PF state.  Do not mask them.
    visible = np.ones_like(topo["blank"])
    regs = ["core", "SOL upstream", "leg SOL", "leg PF"]

    fig, axs = plt.subplots(1, len(keys) + 1, figsize=(4.2 * (len(keys) + 1), 4.0),
                            squeeze=False)
    axs = axs[0]
    for col_, name in enumerate(keys):
        arr = np.asarray(coeffs[name], dtype=float)
        nt = arr.shape[0]
        d = np.abs(arr - arr[0]) if perturbation else np.abs(arr)
        d = np.where(np.isfinite(d), d, np.nan)
        ax = axs[col_]
        print(f"\n{name}:  growth rate per region (last third of run, 1/tau_A)")
        for rg in regs:
            m = (region == rg) & visible
            if not m.any():
                continue
            tr = np.array([np.nanmax(d[t][m]) if np.isfinite(d[t][m]).any() else np.nan
                           for t in range(nt)])
            if perturbation:
                tr[0] = np.nan
            gam = _fit_gamma(tr, timestep, np.arange(nt - max(nt // 3, 3), nt))
            ax.semilogy(np.arange(nt), tr, "o-", ms=3, color=REGION_COLORS[rg],
                        label=f"{rg}  ({gam:+.2f})")
            print(f"   {rg:<14} final {tr[-1]:.2e}   gamma {gam:+.3f}")
        ax.set_title(LABELS.get(name, name), fontsize=10)
        ax.set_xlabel("time index")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(fontsize=7, title=r"region ($\gamma\,\tau_A$)", title_fontsize=7)
    axs[0].set_ylabel("max|f - f(0)| in region" if perturbation else "max|f| in region")

    # where is the global max at every output?
    ax = axs[-1]
    name = keys[0]
    arr = np.asarray(coeffs[name], dtype=float)
    d = np.abs(arr - arr[0]) if perturbation else np.abs(arr)
    d = np.where(np.isfinite(d) & visible[None], d, np.nan)
    print(f"\nlocation of max|{name}| per output:")
    pts = []
    for t in range(1 if perturbation else 0, arr.shape[0]):
        if not np.isfinite(d[t]).any():
            continue
        i, jj = np.unravel_index(np.nanargmax(d[t]), d[t].shape)
        pts.append((t, i, jj))
        print(f"   t={t:2d}  ix={i:3d} (psi_n {x[i]:.4f})  iy={jj:2d}  [{region[i, jj]}]")
    if pts:
        p = np.array(pts)
        sc = ax.scatter(p[:, 2], x[p[:, 1]], c=p[:, 0], cmap="plasma", s=30, zorder=3)
        fig.colorbar(sc, ax=ax, label="time index", pad=0.01)
    ax.set_xlim(-0.5, topo["ny"] - 0.5)
    ax.set_ylim(x.min(), x.max())
    _overlay_topology(ax, topo, x)
    ax.set_xlabel("poloidal index")
    ax.set_ylabel(r"$\psi_n$")
    ax.set_title(f"location of max|{LABELS.get(name, name)}|", fontsize=10)
    fig.tight_layout()
    _save(fig, savedir, "crash_trace.png")
    _show(fig)
    return fig


def crash_map(coeffs, psin, topo, key="Xe_t", times=None, perturbation=True,
              decades=8, savedir=None):
    """
    log10|f - f(0)| over (poloidal index, psi_n) at several times, one colour
    scale for all panels.  Grey = PF cells PF_filter blanks in the dumps
    (they are still evolving in the solver).
    """
    arr = np.asarray(coeffs[key], dtype=float)
    nt = arr.shape[0]
    x = np.asarray(psin)
    times = times or sorted(set([1, nt // 2, nt - 2, nt - 1]))
    times = [_tidx(t, nt) for t in times]
    d = np.abs(arr - arr[0]) if perturbation else np.abs(arr)
    d = np.where(np.isfinite(d), d, np.nan)
    vmax = np.log10(np.nanmax(d[times]) + TINY)
    vmin = vmax - decades

    fig, axs = plt.subplots(1, len(times), figsize=(3.6 * len(times), 4.2),
                            sharey=True, squeeze=False)
    axs = axs[0]
    jx = np.arange(arr.shape[2])
    for ax, t in zip(axs, times):
        z = np.log10(d[t] + TINY)
        pc = ax.pcolormesh(jx, x, z, shading="nearest", cmap="inferno",
                           vmin=vmin, vmax=vmax)
        if topo["blank"].any():
            ax.contour(jx, x, topo["blank"].astype(float), levels=[0.5],
                       colors="c", linewidths=1.0)
        _overlay_topology(ax, topo, x)
        ax.set_title(f"t = {t}", fontsize=10)
        ax.set_xlabel("poloidal index")
    axs[0].set_ylabel(r"$\psi_n$ (midplane map)")
    fig.colorbar(pc, ax=list(axs), pad=0.01,
                 label=(r"$\log_{10}|f-f_0|$" if perturbation else r"$\log_{10}|f|$"))
    fig.suptitle(f"{LABELS.get(key, key)}: where the crash lives "
                 f"(cyan outline = PF cells zeroed in the Ni/Te/phi dumps by PF_filter; "
                 f"these derived arrays are not zeroed)",
                 fontsize=10)
    _save(fig, savedir, f"crash_map_{key}.png")
    _show(fig)
    return fig


def spike_locator(coeffs, psin, topo, key="Xe_t", t_from=None, savedir=None):
    """
    For every leg row and every output from t_from on, find the radial index of
    the largest 2*dx (one-cell) feature.  A histogram pinned to one index means
    the spike is anchored to something at that index:
      * at the PF_limit cut  -> sheath-BC / zero-BC switch (SBC_* with PF_limit)
      * at ixsep             -> separatrix discontinuity in the equilibrium
      * spread out           -> generic under-resolution
    """
    arr = np.asarray(coeffs[key], dtype=float)
    nt, nx, ny = arr.shape
    x = np.asarray(psin)
    t_from = nt // 2 if t_from is None else _tidx(t_from, nt)
    legs = np.flatnonzero(topo["leg_y"])
    ups = np.flatnonzero(~topo["leg_y"])
    # The SIBIL arrays (Vexb, ddr_*, fluxes -> D, chi) are computed inside the
    # RHS BEFORE PF_filter runs, so they are NOT blanked even with PF_limit=true:
    # they show the solver's real PF state.  Do not mask them.
    visible = np.ones_like(topo["blank"])

    def hist(rows):
        w = np.zeros(nx)
        for t in range(t_from, nt):
            for jj in rows:
                y = np.where(visible[:, jj], arr[t, :, jj], 0.0)
                y = np.where(np.isfinite(y), y, 0.0)
                oe = odd_even_amplitude(y)
                if oe.max() > 0:
                    w[np.argmax(oe)] += 1
        return w

    w_leg, w_up = hist(legs), hist(ups)
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ix = np.arange(nx)
    ax.bar(ix - 0.2, w_leg, width=0.4, color="C3", label="leg rows")
    ax.bar(ix + 0.2, w_up, width=0.4, color="C0", label="upstream rows")
    ax.axvline(topo["pf_cut_ix"] - 0.5, color="m", lw=1.4, label="PF_limit cut")
    ax.axvline(topo["ixsep"] - 0.5, color="k", ls="--", lw=1.0, label="separatrix")
    ax.set_xlabel("radial index ix")
    ax.set_ylabel("# (row, t) whose largest\n1-cell feature sits here")
    ax.set_title(f"{LABELS.get(key, key)}: radial position of the 2dx spike, t >= {t_from}",
                 fontsize=10)
    lo = max(0, min(topo["pf_cut_ix"], topo["ixsep"]) - 25)
    ax.set_xlim(lo, min(nx, topo["ixsep"] + 25))
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    top = int(np.argmax(w_leg))
    print(f"  leg rows: modal spike index ix={top} (psi_n {x[top]:.4f}), "
          f"{w_leg[top] / max(w_leg.sum(), 1):.0%} of samples; "
          f"PF_limit cut at ix={topo['pf_cut_ix']}, separatrix at ix={topo['ixsep']}")
    fig.tight_layout()
    _save(fig, savedir, f"spike_locator_{key}.png")
    _show(fig)
    return fig


def parallel_structure(coeffs, psin, topo, key="Xe_t", psi_list=None,
                       times=None, perturbation=True, savedir=None):
    """
    Amplitude along y on a few flux surfaces.  Growth pinned to the target cells
    (j=0, ny-1) points at the sheath BC; growth pinned to the X-point cut points
    at the metric / zShift connection; growth peaked at jomp is ballooning.
    """
    arr = np.asarray(coeffs[key], dtype=float)
    nt, nx, ny = arr.shape
    x = np.asarray(psin)
    psi_list = psi_list or [x[topo["pf_cut_ix"]], x[topo["ixsep"]], 1.02]
    times = [_tidx(t, nt) for t in (times or [nt // 2, nt - 1])]
    d = np.abs(arr - arr[0]) if perturbation else np.abs(arr)
    bad = ~np.isfinite(d)
    d = np.where(bad, np.nan, d)
    floor = np.nanmax(d[times]) * 1e-12 if np.isfinite(d[times]).any() else TINY

    fig, axs = plt.subplots(1, len(psi_list), figsize=(4.6 * len(psi_list), 3.6),
                            sharey=True, squeeze=False)
    for ax, p in zip(axs[0], psi_list):
        i = int(np.argmin(np.abs(x - p)))
        for t, ls in zip(times, ("--", "-")):
            ax.semilogy(np.arange(ny), np.maximum(d[t, i], floor), ls, color="C1",
                        label=f"t={t}")
            jb = np.flatnonzero(bad[t, i])
            if jb.size:
                ax.plot(jb, np.full(jb.size, floor), "rx", ms=5,
                        label="non-finite" if ls == "-" else None)
        for c in _xpt_cuts(topo):
            ax.axvline(c, color="k", ls="--", lw=0.8)
        for jt in (0, ny - 1):
            ax.axvline(jt, color="0.5", lw=3, alpha=0.3)
        ax.set_title(rf"ix={i}, $\psi_n$={x[i]:.4f}" +
                     ("  (PF in legs)" if i < topo["ixsep"] else ""), fontsize=9)
        ax.set_xlabel("poloidal index")
        ax.grid(True, alpha=0.3, which="both")
        ax.set_ylim(bottom=floor)
    axs[0][0].set_ylabel(f"|{LABELS.get(key, key)} - f0|  (grey = target)", fontsize=9)
    axs[0][0].legend(fontsize=7)
    fig.tight_layout()
    _save(fig, savedir, f"parallel_structure_{key}.png")
    _show(fig)
    return fig


# =========================================================================== #
# Q3d.  Poloidal localisation
# =========================================================================== #
def poloidal_localization(coeffs, psin, t=-1, keys=None, jomp=None, topo=None,
                          psi_window=(0.97, 1.03), savedir=None):
    """
    Max amplitude inside a radial window around the separatrix vs poloidal index.
    Peak at the outboard midplane = ballooning; peak in the legs / at the X-point
    cut = metric / boundary / numerical.
    """
    keys = keys or [k for k in COEFF_KEYS + VEL_KEYS if k in coeffs]
    x = np.asarray(psin)
    sel = (x >= psi_window[0]) & (x <= psi_window[1])
    t = _tidx(t, _safe_nt(coeffs, keys))

    fig, ax = plt.subplots(figsize=(8, 4))
    for name in keys:
        arr = np.asarray(coeffs[name])
        if arr.ndim != 3:
            continue
        sl = arr[t]
        prof = np.array([_finite_max(sl[sel, j]) for j in range(sl.shape[1])])
        ax.semilogy(np.arange(sl.shape[1]), prof / (np.nanmax(prof) + TINY), "o-", ms=3,
                    color=COLORS.get(name, None), label=LABELS.get(name, name))
    if jomp is not None:
        ax.axvline(jomp, color="k", ls="--", lw=1.0)
        ax.annotate("OMP", (jomp, 0.02), xycoords=("data", "axes fraction"),
                    fontsize=8, ha="right", rotation=90)
    if topo is not None:
        for lo, hi in _leg_spans(topo):
            ax.axvspan(lo - 0.5, hi + 0.5, color="C3", alpha=0.08)
        for c in _xpt_cuts(topo):
            ax.axvline(c, color="C3", ls=":", lw=1.0)
    ax.set_xlabel("poloidal index (red band = divertor legs)", fontsize=12)
    ax.set_ylabel(f"max in psi_n {psi_window} (norm.)", fontsize=11)
    ax.set_title(f"Poloidal localisation at time index {t}", fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    _save(fig, savedir, f"poloidal_localization_t{t}.png")
    _show(fig)
    return fig


# =========================================================================== #
# V_ExB  (rewritten)
# =========================================================================== #
def _norm_from_grid(grid_or_path, AA=2.0, density=1.0e20):
    """Rebuild Lbar, Bbar, Va exactly as elm_6f does (rmag, bmag, Nixexp)."""
    g, R = _open_grid(grid_or_path)
    lbar = _scalar(R, g, "rmag", 1.0)
    bbar = _scalar(R, g, "bmag", 1.0)
    nbar = _scalar(R, g, "Nixexp", 1.0)
    va = np.sqrt(bbar ** 2 / (MU0 * AA * MP * nbar * density))
    Rxy, _ = _try(R, g, "Rxy")
    Bp, _ = _try(R, g, "Bpxy")
    return dict(lbar=lbar, bbar=bbar, va=va, Rxy=Rxy, Bpxy=Bp)


def vexb_to_ms(vx, grid_or_path, bpxy_constraint=0.04, AA=2.0, density=1.0e20):
    """
    Vexb.x dumped by elm_6f is the CONTRAVARIANT x-component V.grad(psi) in
    normalised units (psi / (Lbar^2 Bbar) per tau_A).  Radial velocity in m/s:

        V_r = V^x / |grad psi| * Va  with |grad psi| = (R/Lbar)(Bp/Bbar)

    Bp is floored as limit_jacobi does.  Works on (..., nx, ny) arrays.
    """
    n = _norm_from_grid(grid_or_path, AA, density)
    rbp = (n["Rxy"] / n["lbar"]) * (np.maximum(n["Bpxy"], bpxy_constraint) / n["bbar"])
    return np.asarray(vx) / rbp * n["va"]


def plot_vexb(vexb, psin, topo=None, grid=None, t=-1, jomp=44, j=None,
              savedir=None):
    """
    Replaces the old V_ExB profile panel.

    Why the old panel was noise
    ---------------------------
      1. At jomp V_ExB is ~1e-5 of its value in the legs (see
         poloidal_localization).  At the midplane you are looking at leakage
         from the leg mode, which is dominated by its 2*dx component.
      2. V^x is SIGNED.  |V^x| on a log axis drops to -inf at every zero
         crossing, so a smooth oscillation renders as a comb.
      3. V^x ~ d(phi)/dz.  If Vexbx_txy was built by averaging over z, the
         coherent n>0 part averages to zero, leaving only round-off.  Pass the
         4-D array (t, x, y, z) and this routine uses the z-RMS instead.

    Panels
      a  log10 envelope over (j, psi_n)            -- same structure as the
                                                      poloidal-localisation plot
      b  radial envelope at jomp and at the peak j -- smooth, log axis is safe
      c  signed profile at the peak j (symlog) split into a 1-2-1 smoothed
         (coherent) part and its 2*dx residual
      d  max envelope vs time per region
    """
    v = np.asarray(vexb, dtype=float)
    if v.ndim == 4:                                       # (t, x, y, z)
        mean = v.mean(axis=-1)
        rms = np.sqrt(np.mean((v - mean[..., None]) ** 2, axis=-1))
        env_all = np.sqrt(mean ** 2 + rms ** 2)
        signed = v[..., 0]
        print(f"plot_vexb: 4-D input; max|<V>_z| / max RMS_z = "
              f"{_finite_max(mean) / (_finite_max(rms) + TINY):.2e} "
              f"(<< 1 means a z-average would have shown only round-off)")
    else:
        signed = v
        env_all = None

    if grid is not None:
        signed = vexb_to_ms(signed, grid)
        if env_all is not None:
            env_all = vexb_to_ms(env_all, grid)
        unit = "m/s"
    else:
        unit = "norm. contravariant"

    nt = signed.shape[0]
    t = _tidx(t, nt)
    x = np.asarray(psin)

    def envelope(s):
        """Local max |.| over +/-1 cell in x: removes zero-crossing drop-outs."""
        a = np.abs(np.where(np.isfinite(s), s, np.nan))
        p = np.pad(a, ((1, 1), (0, 0)), mode="edge")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            e = np.nanmax(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)
        return np.where(np.isfinite(s), e, np.nan)

    env = (env_all if env_all is not None
           else np.array([envelope(signed[k]) for k in range(nt)]))
    env_t = env[t]
    if j is None:
        jj = np.arange(env_t.shape[1])
        j = int(jj[np.argmax(np.nanmax(env_t, axis=0))])

    fig = plt.figure(figsize=(13, 8))
    gs = fig.add_gridspec(2, 2)
    a0 = fig.add_subplot(gs[0, 0])
    a1 = fig.add_subplot(gs[0, 1])
    a2 = fig.add_subplot(gs[1, 0])
    a3 = fig.add_subplot(gs[1, 1])

    nbad = int((~np.isfinite(signed[t])).sum())
    if nbad:
        print(f"plot_vexb: {nbad} non-finite cells at t={t} (drawn cyan in the map)")
    z = _log10abs(env_t, 10)
    cm = plt.get_cmap("inferno").copy()
    cm.set_bad("cyan")
    pc = a0.pcolormesh(np.arange(env_t.shape[1]), x, np.ma.masked_invalid(z),
                       shading="nearest", cmap=cm)
    fig.colorbar(pc, ax=a0, pad=0.01, label=rf"$\log_{{10}}|V_{{E\times B}}|$ [{unit}]")
    if topo is not None:
        _overlay_topology(a0, topo, x)
    a0.axvline(jomp, color="c", lw=1.0)
    a0.axvline(j, color="lime", lw=1.0)
    a0.set_xlabel("poloidal index (cyan = jomp, green = peak)")
    a0.set_ylabel(r"$\psi_n$")
    a0.set_title(f"envelope, t = {t}", fontsize=10)

    for jj_, c, lab in ((jomp, "c", f"j={jomp} (OMP)"), (j, "g", f"j={j} (peak)")):
        fl = np.nanmax(env_t) * 1e-12
        a1.semilogy(x, np.maximum(env_t[:, jj_], fl), color=c, lw=1.3, label=lab)
    _sep_line(a1)
    a1.set_xlabel(r"$\psi_n$")
    a1.set_ylabel(f"|V_ExB| envelope [{unit}]")
    a1.set_title(f"radial envelope; peak/OMP = "
                 f"{_finite_max(env_t[:, j]) / (_finite_max(env_t[:, jomp]) + TINY):.1e}",
                 fontsize=10)
    a1.legend(fontsize=8)
    a1.grid(True, alpha=0.3, which="both")

    s = np.where(np.isfinite(signed[t, :, j]), signed[t, :, j], 0.0)
    smooth = s.copy()
    smooth[1:-1] = 0.25 * s[:-2] + 0.5 * s[1:-1] + 0.25 * s[2:]
    resid = s - smooth
    lt = max(_finite_max(s) * 1e-4, TINY)
    a2.plot(x, s, color="0.7", lw=0.8, label="raw")
    a2.plot(x, smooth, color="C1", lw=1.4, label="1-2-1 smoothed (coherent)")
    a2.plot(x, resid, color="C3", lw=0.8, label=r"2$\Delta x$ residual")
    a2.set_yscale("symlog", linthresh=lt)
    _sep_line(a2)
    if topo is not None and topo["leg_y"][j]:
        a2.axvline(_psi_at(x, topo["PF_limit_range"] * topo["ixsep"] - 0.5),
                   color="m", lw=1.2)
    frac = np.sqrt(np.sum(resid ** 2) / (np.sum(s ** 2) + TINY))
    if v.ndim == 3 and frac > 0.3:
        print(f"plot_vexb: 2dx residual is {frac:.0%} of the signal at j={j}. "
              f"If Vexbx_txy is a z-average, rebuild it from the 4-D Vexbx dump "
              f"(V^x ~ d(phi)/dz averages to zero over z).")
    a2.set_title(f"signed, j={j}: rms(2dx residual)/rms(raw) = {frac:.2f}", fontsize=10)
    a2.set_xlabel(r"$\psi_n$")
    a2.set_ylabel(f"V_ExB [{unit}]")
    a2.legend(fontsize=7)
    a2.grid(True, alpha=0.3)

    if topo is not None:
        for rg in ("core", "SOL upstream", "leg SOL", "leg PF"):
            m = topo["region"] == rg
            if m.any():
                a3.semilogy(np.arange(nt), [np.nanmax(env[k][m]) for k in range(nt)],
                            "o-", ms=3, color=REGION_COLORS[rg], label=rg)
    a3.semilogy(np.arange(nt), [np.nanmax(env[k][:, jomp]) for k in range(nt)],
                "k--", lw=1.0, label=f"j={jomp}")
    a3.set_xlabel("time index")
    a3.set_ylabel(f"max envelope [{unit}]")
    a3.set_title("V_ExB growth by region", fontsize=10)
    a3.legend(fontsize=7)
    a3.grid(True, alpha=0.3, which="both")

    fig.tight_layout()
    _save(fig, savedir, f"vexb_t{t}.png")
    _show(fig)
    return fig


# =========================================================================== #
# Velocity decomposition consistency
# =========================================================================== #
def velocity_consistency(coeffs, psin, j=44, t=-1, grid=None, savedir=None):
    """
    Checks V_r ?= V_ExB + Vbtilde.  Vexb.x and Vbtilde.x are dumped as
    contravariant x components (psi per tau_A); pass `grid` to convert both to
    m/s before comparing with V_r.  Without the conversion the three are not in
    the same units and the decomposition cannot close.
    """
    for k in VEL_KEYS:
        if k not in coeffs:
            print(f"missing {k} -- cannot run velocity consistency check")
            return None

    x = np.asarray(psin)
    t = _tidx(t, _safe_nt(coeffs, VEL_KEYS))
    vr = get_slice(coeffs["Vr_t"], j, t)
    ve = get_slice(coeffs["Vexbx_txy"], j, t)
    vb = get_slice(coeffs["Vbtildx_txy"], j, t)
    if grid is not None:
        ve = vexb_to_ms(np.asarray(coeffs["Vexbx_txy"])[t], grid)[:, j]
        vb = vexb_to_ms(np.asarray(coeffs["Vbtildx_txy"])[t], grid)[:, j]
        print("V_ExB and Vbtilde converted to m/s via R*Bp and Va.")
    tot = ve + vb

    print(f"max|V_r|        = {_finite_max(vr):.3e}")
    print(f"max|V_ExB|      = {_finite_max(ve):.3e}   (ratio to V_r: "
          f"{_finite_max(ve) / (_finite_max(vr) + TINY):.2e})")
    print(f"max|Vbtilde|    = {_finite_max(vb):.3e}   (ratio to V_r: "
          f"{_finite_max(vb) / (_finite_max(vr) + TINY):.2e})")
    resid = _finite_max(vr - tot) / (_finite_max(vr) + TINY)
    print(f"relative residual |V_r - (V_ExB+Vb)| / max|V_r| = {resid:.3e}")

    fig, axs = plt.subplots(2, 1, figsize=(7.5, 5), sharex=True)
    axs[0].plot(x, vr, color=COLORS["Vr_t"], lw=1.3, label=r"$V_r$")
    axs[0].plot(x, tot, "k--", lw=1.0, label=r"$V_{E\times B}+\tilde V_b$")
    axs[0].legend(fontsize=8)
    axs[0].set_ylabel("velocity", fontsize=12)
    for k, y in (("Vr_t", vr), ("Vexbx_txy", ve), ("Vbtildx_txy", vb)):
        axs[1].semilogy(x, np.abs(y) + TINY, color=COLORS[k], lw=1.1, label=LABELS[k])
    axs[1].set_ylabel("|v| (log)", fontsize=12)
    axs[1].set_xlabel(r"$\psi_n$", fontsize=12)
    axs[1].legend(fontsize=8, ncol=3)
    for a in axs:
        _sep_line(a)
        a.grid(True, alpha=0.3)
        a.tick_params(labelsize=7)
    fig.suptitle(f"Velocity decomposition, pol idx {j}, time idx {t}")
    fig.tight_layout()
    _save(fig, savedir, f"velocity_consistency_j{j}_t{t}.png")
    _show(fig)
    return fig


# =========================================================================== #
# v5 additions
# =========================================================================== #
def nonfinite_report(coeffs, psin, topo=None, keys=None, savedir=None):
    """
    First output at which each field contains NaN/Inf, how many cells, and a
    map of where they are at that output.  NaN cells were previously drawn as
    zero (black pixels / 1e-300 drop-outs) in the envelope plots.
    """
    keys = keys or [k for k in coeffs if np.asarray(coeffs[k]).ndim == 3]
    x = np.asarray(psin)
    first = {}
    print(f"{'field':<16}{'first bad t':>12}{'# cells':>9}{'# at last t':>13}   region of first bad")
    print("-" * 78)
    for k in keys:
        a = np.asarray(coeffs[k], dtype=float)
        bad = ~np.isfinite(a)
        ts = np.flatnonzero(bad.reshape(a.shape[0], -1).any(axis=1))
        if ts.size == 0:
            print(f"{k:<16}{'-':>12}{0:>9}{0:>13}")
            continue
        t0 = int(ts[0])
        first[k] = t0
        regs = ""
        if topo is not None:
            r, c = np.unique(topo["region"][bad[t0]], return_counts=True)
            regs = ", ".join(f"{a_}:{b_}" for a_, b_ in zip(r, c))
        print(f"{k:<16}{t0:>12}{int(bad[t0].sum()):>9}{int(bad[-1].sum()):>13}   {regs}")
    if not first:
        print("no non-finite values in any field")
        return None

    k0 = min(first, key=first.get)
    a = np.asarray(coeffs[k0], dtype=float)
    t0 = first[k0]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.pcolormesh(np.arange(a.shape[2]), x, (~np.isfinite(a[t0])).astype(float),
                  shading="nearest", cmap="Reds", vmin=0, vmax=1)
    if topo is not None:
        _overlay_topology(ax, topo, x)
    ax.set_xlabel("poloidal index")
    ax.set_ylabel(r"$\psi_n$")
    ax.set_title(f"non-finite cells of {k0} at its first bad output t={t0}", fontsize=10)
    fig.tight_layout()
    _save(fig, savedir, "nonfinite_map.png")
    _show(fig)
    return fig


def background_consistency(coeffs, psin, grid_or_path, jomp=44, savedir=None):
    """
    Compares the t=0 radial gradients in `coeffs` (ddr_nitot, ddr_Tetot,
    ddr_Titot = sqrt(g11)*DDX(f0+f)) with R*Bp*d f0/dpsi rebuilt from the grid
    profiles at jomp.  Both are normalised to their own peak so units do not
    matter: the SHAPES must agree.  If they do not, either the radial axis
    (psin) does not match the arrays, or the coefficients were built from a
    different background than the one BOUT++ is running.
    """
    g, R = _open_grid(grid_or_path)
    psixy, _ = _try(R, g, "psixy")
    pa, _ = _try(R, g, "psi_axis")
    pb, _ = _try(R, g, "psi_bdry", "psi_bndry")
    Rxy, _ = _try(R, g, "Rxy")
    Bp, _ = _try(R, g, "Bpxy")
    xg = ((psixy - pa) / (pb - pa))[:, jomp]
    x = np.asarray(psin)
    pairs = (("ddr_nitot_txy", ("Niexp", "Ni0")), ("ddr_Tetot_txy", ("Teexp", "Te0")),
             ("ddr_Titot_txy", ("Tiexp", "Ti0")))
    pairs = [p for p in pairs if p[0] in coeffs]
    if not pairs:
        print("no ddr_* arrays in coeffs")
        return None

    fig, axs = plt.subplots(len(pairs), 1, figsize=(7.5, 2.4 * len(pairs)),
                            sharex=True, squeeze=False)
    for ax, (k, gnames) in zip(axs[:, 0], pairs):
        f0, used = _try(R, g, *gnames)
        dpsi = np.gradient(psixy[:, jomp])
        ref = Rxy[:, jomp] * Bp[:, jomp] * np.gradient(f0[:, jomp]) / dpsi
        sim = np.asarray(get_slice(coeffs[k], jomp, 0), dtype=float)
        refn = ref / (_finite_max(ref) + TINY)
        simn = sim / (_finite_max(sim) + TINY)
        ax.plot(xg, refn, "k", lw=1.4, label=f"grid: R Bp d{used}/dpsi")
        ax.plot(x, simn, color=COLORS.get(k, "C1"), lw=1.1, ls="--", label=f"{k}, t=0")
        if x.size == xg.size:
            cc = np.corrcoef(np.nan_to_num(refn), np.nan_to_num(simn))[0, 1]
            shift = int(np.argmax(np.abs(simn))) - int(np.argmax(np.abs(refn)))
            print(f"  {k:<15} shape correlation with grid = {cc:+.3f}; "
                  f"peak offset = {shift:+d} cells")
            if cc < 0.9:
                print(f"    -> {k} does NOT have the shape of the grid background")
        ax.set_ylabel("norm.")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        _sep_line(ax)
    axs[-1, 0].set_xlabel(r"$\psi_n$")
    fig.suptitle(f"Background gradient consistency at j={jomp}", fontsize=10)
    fig.tight_layout()
    _save(fig, savedir, f"background_consistency_j{jomp}.png")
    _show(fig)
    return fig


def oddeven_map(coeffs, psin, topo=None, key="Xe_t", times=None, savedir=None):
    """
    Radial 2*dx fraction over the whole (j, psi_n) plane at a few times.
    Horizontal striping in crash_map = this map lighting up away from the legs:
    odd-even decoupling of the C2 radial stencils (and of BRACKET_STD ExB
    advection) everywhere, not just a leg problem.
    """
    a = np.asarray(coeffs[key], dtype=float)
    nt = a.shape[0]
    x = np.asarray(psin)
    times = [_tidx(t, nt) for t in (times or [nt // 2, nt - 1])]
    fig, axs = plt.subplots(1, len(times), figsize=(5 * len(times), 4),
                            sharey=True, squeeze=False)
    for ax, t in zip(axs[0], times):
        f = np.where(np.isfinite(a[t]), a[t], 0.0)
        oe = np.zeros_like(f)
        oe[1:-1] = np.abs(-f[:-2] + 2 * f[1:-1] - f[2:]) / 4
        # local amplitude over +/-3 cells
        p = np.pad(np.abs(f), ((3, 3), (0, 0)), mode="edge")
        loc = np.max(np.stack([p[k:k + f.shape[0]] for k in range(7)]), axis=0)
        frac = np.where(loc > 0, oe / (loc + TINY), np.nan)
        pc = ax.pcolormesh(np.arange(f.shape[1]), x, frac, shading="nearest",
                           cmap="viridis", vmin=0, vmax=1)
        if topo is not None:
            _overlay_topology(ax, topo, x)
        ax.set_title(f"t = {t}", fontsize=10)
        ax.set_xlabel("poloidal index")
        print(f"  t={t}: median 2dx fraction {np.nanmedian(frac):.2f} "
              f"(core/SOL upstream rows {np.nanmedian(frac[:, ~topo['leg_y']]) if topo else np.nan:.2f})")
    axs[0][0].set_ylabel(r"$\psi_n$")
    fig.colorbar(pc, ax=list(axs[0]), label="local radial 2dx fraction "
                 "(0.5 one-cell spike, 1 checkerboard)")
    fig.suptitle(f"{LABELS.get(key, key)}: radial odd-even content", fontsize=10)
    _save(fig, savedir, f"oddeven_map_{key}.png")
    _show(fig)
    return fig


def vorticity_terms(terms, psin, topo, timestep=10.0, savedir=None):
    """
    Which term drives the vorticity growth, per region.  Needs
    output_transfer = true in [highbeta] (no rebuild); then load from the dumps
    e.g. terms = {k: collect(k, path=run) for k in ("T_R","T_M","T_C","T_G")}
      T_C  curvature drive          -> ballooning / interchange
      T_M  B^2 Grad_par(J/B)        -> current / sheath-current (peeling, SBC)
      T_R  Reynolds stress (ExB)    -> nonlinear / numerical advection
      T_G  gyroviscous terms
    Arrays may be (t,x,y,z) or (t,x,y); RMS over z is used.
    """
    names = {"T_C": "curvature", "T_M": "J_par (Maxwell/sheath)",
             "T_R": "Reynolds (ExB)", "T_G": "gyroviscous"}
    regs = ["core", "SOL upstream", "leg SOL", "leg PF"]
    fig, axs = plt.subplots(1, len(regs), figsize=(4.2 * len(regs), 3.8),
                            sharey=True, squeeze=False)
    for k, v in terms.items():
        a = np.asarray(v, dtype=float)
        if a.ndim == 4:
            a = np.sqrt(np.nanmean(a ** 2, axis=-1))
        a = np.abs(np.where(np.isfinite(a), a, np.nan))
        for ax, rg in zip(axs[0], regs):
            m = topo["region"] == rg
            tr = [np.nanmax(a[t][m]) if np.isfinite(a[t][m]).any() else np.nan
                  for t in range(a.shape[0])]
            ax.semilogy(np.arange(a.shape[0]), tr, "o-", ms=3, label=names.get(k, k))
    for ax, rg in zip(axs[0], regs):
        ax.set_title(rg, fontsize=10)
        ax.set_xlabel("time index")
        ax.grid(True, alpha=0.3, which="both")
    axs[0][0].set_ylabel("max |term| in region")
    axs[0][0].legend(fontsize=7)
    fig.suptitle("Vorticity-equation term balance (output_transfer)", fontsize=10)
    fig.tight_layout()
    _save(fig, savedir, "vorticity_terms.png")
    _show(fig)
    return fig


def floor_watch(pert, back, psin, topo, frac=0.05, savedir=None):
    """
    Is the TOTAL profile (background + perturbation) collapsing somewhere?

    pert : {"Te": Te(t,x,y[,z]), "Ni": ..., "Ti": ...}  from collect() on BOUT.dmp
    back : {"Te": Te0(x,y), ...}  in the SAME normalisation (Teexp/Tebar, Niexp/Nbar;
           with Tebar = Nbar = 1 as in your logs that is just Teexp, Niexp)

    elm_6f floors totals at Low_limit (field_larger), and kappa_par ~ T^(5/2),
    c_s ~ sqrt(T), exp(-e phi/Te) in the sheath current all become extreme as
    T or n -> 0.  A slowly shrinking minimum here is the usual cause of a
    steadily rising RHS-eval count that ends in mxstep (flag -3).

    Prints per output: min(total/background), its location and region, and the
    number of cells below `frac` of their background.  Plots the trace and a
    map of min_z(total/background) at the last output.
    """
    x = np.asarray(psin)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))
    last_map = None
    for name, p in pert.items():
        p = np.asarray(p, dtype=float)
        b = np.asarray(back[name], dtype=float)
        mxg = (p.shape[1] - b.shape[0]) // 2
        if mxg > 0:
            p = p[:, mxg:-mxg]
        ratio = (b[None, ..., None] + p) / b[None, ..., None] if p.ndim == 4 \
            else (b[None] + p) / b[None]
        rmin = ratio.min(axis=-1) if ratio.ndim == 4 else ratio
        tr, nlow = [], []
        print(f"\n{name}: min(total/background) per output")
        for t in range(rmin.shape[0]):
            r = rmin[t]
            i, j = np.unravel_index(np.nanargmin(r), r.shape)
            tr.append(r[i, j])
            nlow.append(int((r < frac).sum()))
            if t % max(1, rmin.shape[0] // 12) == 0 or t == rmin.shape[0] - 1:
                print(f"  t={t:3d}  min {r[i, j]:+.3f} at ix={i} (psi_n {x[i]:.4f}) "
                      f"iy={j} [{topo['region'][i, j]}]   cells<{frac:g}: {nlow[-1]}")
        axs[0].plot(tr, "o-", ms=3, label=name)
        if last_map is None or np.nanmin(rmin[-1]) < np.nanmin(last_map[1]):
            last_map = (name, rmin[-1])
    axs[0].axhline(frac, color="r", ls=":", lw=0.9)
    axs[0].axhline(0, color="k", lw=0.8)
    axs[0].set_xlabel("time index")
    axs[0].set_ylabel("min over plane of total / background")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3)
    name, m = last_map
    pc = axs[1].pcolormesh(np.arange(m.shape[1]), x, np.clip(m, 0, 2), shading="nearest",
                           cmap="RdBu", vmin=0, vmax=2)
    fig.colorbar(pc, ax=axs[1], label=f"min_z {name}_total / {name}0 (last output)")
    _overlay_topology(axs[1], topo, x)
    axs[1].set_xlabel("poloidal index")
    axs[1].set_ylabel(r"$\psi_n$")
    fig.tight_layout()
    _save(fig, savedir, "floor_watch.png")
    _show(fig)
    return fig


def failed_state_report(run_dir, topo, psin, varnames=("Ni", "Te", "Ti", "U", "phi",
                        "Vipar", "Apar"), prefix="BOUT.failed", ref_prefix="BOUT.dmp",
                        savedir=None):
    """
    Where is the solver state when CVODE gave up?  Reads the state BOUT++ writes
    on failure (prefix BOUT.failed) and the last good dump, and for each evolved
    variable reports max|f| and max|f_failed - f_last| with location/region.
    A crash with flag -3 (mxstep) has NO non-finite values: this is how you find
    the cells that forced the tiny time steps.
    """
    from boutdata import collect
    x = np.asarray(psin)
    rows = []
    for v in varnames:
        try:
            ff = np.asarray(collect(v, path=run_dir, prefix=prefix, info=False))
        except Exception as exc:
            print(f"  {v:<6} not in {prefix}: {exc}")
            continue
        ff = ff[-1] if ff.ndim == 4 else ff            # (x, y, z)
        try:
            fl = np.asarray(collect(v, path=run_dir, prefix=ref_prefix, tind=-1, info=False))
            fl = fl[-1] if fl.ndim == 4 else fl
        except Exception:
            fl = np.zeros_like(ff)
        mxg = (ff.shape[0] - x.size) // 2
        if mxg > 0:
            ff, fl = ff[mxg:-mxg], fl[mxg:-mxg]
        d = np.abs(ff - fl)
        rms = np.sqrt(np.nanmean(np.where(np.isfinite(d), d, np.nan) ** 2, axis=-1))
        nbad = int((~np.isfinite(ff)).sum())
        i, j = np.unravel_index(np.nanargmax(rms), rms.shape)
        rows.append((v, rms))
        print(f"  {v:<6} non-finite {nbad:>6}   max rms|f_fail - f_last| = {rms[i, j]:.3e} "
              f"at ix={i} (psi_n {x[i]:.4f}), iy={j} [{topo['region'][i, j]}]")
    if not rows:
        return None
    fig, axs = plt.subplots(1, len(rows), figsize=(3.6 * len(rows), 4), sharey=True,
                            squeeze=False)
    for ax, (v, rms) in zip(axs[0], rows):
        pc = ax.pcolormesh(np.arange(rms.shape[1]), x, _log10abs(rms, 8),
                           shading="nearest", cmap="inferno")
        fig.colorbar(pc, ax=ax, pad=0.01)
        _overlay_topology(ax, topo, x)
        ax.set_title(f"{v}: log10 rms_z|f_fail - f_last|", fontsize=8)
        ax.set_xlabel("poloidal index")
    axs[0][0].set_ylabel(r"$\psi_n$")
    fig.tight_layout()
    _save(fig, savedir, "failed_state.png")
    _show(fig)
    return fig


# =========================================================================== #
# Interactive widget
# =========================================================================== #
def make_debug_widget(coeffs, psin, jomp=None, grid=None, data_dir=None,
                      default_j=44, default_t=None, auto_harmonise=True):
    """
    Profile widget: mask toggle, eps slider, log-y, 2dx overlay, psi zoom,
    sliders sized from the data (never from the config scalar `l`).
    """
    try:
        from ipywidgets import (interactive_output, IntSlider, FloatLogSlider,
                                Checkbox, FloatRangeSlider, HBox, VBox, HTML, Output)
    except Exception:
        warnings.warn("ipywidgets unavailable -- falling back to a static plot")
        return plot_profiles_v2(coeffs, psin, default_j, 0)

    chk = check_time_axes(coeffs, psin=psin, data_dir=data_dir, verbose=True)
    nt, ny, ragged = chk["nt"], chk["ny"], chk["ragged"]
    if ragged and auto_harmonise:
        coeffs = harmonise(coeffs, nt=nt, verbose=True)
        chk = check_time_axes(coeffs, verbose=False)
        nt, ny = chk["nt"], chk["ny"]
    if nt < 1:
        raise ValueError("no time-dependent arrays found in coeffs")
    default_t = nt - 1 if default_t is None else min(max(int(default_t), 0), nt - 1)
    default_j = min(max(int(default_j), 0), ny - 1)

    banner = (f"<b>nt = {nt}</b> &nbsp; <b>ny = {ny}</b> &nbsp; "
              f"nx = {np.asarray(psin).shape[0]}")
    if ragged:
        banner += ("<br><span style='color:#b00'><b>RAGGED INPUT</b> &mdash; "
                   "sliders limited to the shortest array.</span>")
    if jomp is not None:
        banner += f" &nbsp;|&nbsp; jomp = {jomp}"
    hdr = HTML(value=f"<div style='font-family:monospace;font-size:11px'>{banner}</div>")

    j_sl = IntSlider(min=0, max=ny - 1, value=default_j, description="Pol. idx:",
                     continuous_update=False)
    t_sl = IntSlider(min=0, max=nt - 1, value=default_t, description="Time idx:",
                     continuous_update=False)
    mask_cb = Checkbox(value=True, description="mask 0/0")
    log_cb = Checkbox(value=False, description="log y")
    noise_cb = Checkbox(value=False, description="2dx noise")
    eps_sl = FloatLogSlider(value=MASK_EPS, base=10, min=-4, max=-0.5, step=0.1,
                            description="mask eps:", continuous_update=False)
    xr_sl = FloatRangeSlider(value=[float(np.min(psin)), float(np.max(psin))],
                             min=float(np.min(psin)), max=float(np.max(psin)),
                             step=0.005, description="psi range:",
                             continuous_update=False, readout_format=".3f")
    err_out = Output()

    def _plot(j, t, use_mask, logy, show_noise, eps, xr):
        err_out.clear_output()
        try:
            plot_profiles_v2(coeffs, psin, j, t, use_mask=use_mask, logy=logy,
                             show_noise=show_noise, eps=eps, xlim=xr, jomp=jomp)
        except Exception:
            import traceback
            with err_out:
                print("PLOT FAILED -- this is the real error, not a blank plot:")
                traceback.print_exc()

    out = interactive_output(_plot, {"j": j_sl, "t": t_sl, "use_mask": mask_cb,
                                     "logy": log_cb, "show_noise": noise_cb,
                                     "eps": eps_sl, "xr": xr_sl})
    return VBox([hdr, HBox([j_sl, t_sl]), HBox([mask_cb, log_cb, noise_cb]),
                 HBox([eps_sl, xr_sl]), out, err_out])


def plot_profiles_v2(coeffs, psin, j, t=0, use_mask=True, logy=False,
                     show_noise=False, eps=MASK_EPS, xlim=None, jomp=None,
                     savedir=None):
    """Static version of the profile plot.  Signed fields on log-y use |f|."""
    x = np.asarray(psin)
    keys = list(coeffs.keys())
    n = len(keys)

    nt_by_key = {k: np.asarray(coeffs[k]).shape[0]
                 for k in keys if np.asarray(coeffs[k]).ndim == 3}
    if nt_by_key and t >= min(nt_by_key.values()):
        short = {k: v for k, v in nt_by_key.items() if v <= t}
        raise IndexError(f"time index {t} requested but these arrays only have "
                         f"nt = {short}. Regenerate with bout_save_v3.py --purge.")

    fig, axs = plt.subplots(n, 1, figsize=(5.2, 1.35 * n), sharex=True)
    axs = np.atleast_1d(axs)

    report = []
    for ax, name in zip(axs, keys):
        y_raw = np.array(get_slice(coeffs[name], j, t, name=name), dtype=float)
        if y_raw.shape[0] != x.shape[0]:
            raise ValueError(f"{name}: radial length {y_raw.shape[0]} does not "
                             f"match psin length {x.shape[0]}.")
        if use_mask and name in DENOM_OF:
            y, m = masked_profile(coeffs, name, j, t, eps)
            nmask = int((~m).sum())
        else:
            y, nmask = y_raw, 0
        yp = np.abs(y) if logy else y
        ax.plot(x, yp, color=COLORS.get(name, "k"), lw=1.1, label=LABELS.get(name, name))
        if use_mask and nmask:
            ax.plot(x, np.where(np.isnan(y), np.abs(y_raw) if logy else y_raw, np.nan),
                    ":", color="0.6", lw=0.9)
        if show_noise:
            ax.plot(x, odd_even_amplitude(y_raw), "--", color="0.35", lw=0.8)
        if logy:
            ax.set_yscale("log")
        ax.set_ylabel(LABELS.get(name, name), fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=7)
        ax.yaxis.get_offset_text().set_fontsize(7)
        _sep_line(ax)
        if xlim is not None:
            ax.set_xlim(*xlim)
        report.append((name, nmask, int((~np.isfinite(y_raw)).sum())))

    axs[-1].set_xlabel(r"$\psi_n$", fontsize=12)
    _nt = min(nt_by_key.values()) if nt_by_key else 1
    ttl = f"pol idx {j}/{n_y_max(coeffs) - 1}   |   time idx {t}/{_nt - 1}"
    if jomp is not None and j == jomp:
        ttl += "   (OUTBOARD MIDPLANE)"
    fig.suptitle(ttl, fontsize=9, y=1.003)
    _save(fig, savedir, f"profiles_j{j}_t{t}.png")
    _show(fig)

    bad = [r for r in report if r[2] > 0]
    if bad:
        print("NON-FINITE CELLS PRESENT:", ", ".join(f"{r[0]}({r[2]})" for r in bad))
    msk = [r for r in report if r[1] > 0]
    if msk:
        print("masked (vanishing-gradient) cells:", ", ".join(f"{r[0]}({r[1]})" for r in msk))
    return fig


# =========================================================================== #
# Frozen-field check
# =========================================================================== #
def frozen_field_report(coeffs, rtol=1e-12):
    """Report how much each time-dependent array changes from t=0."""
    print(f"{'array':<22}{'max|f(t)-f(0)|':>16}{'rel':>12}   verdict")
    print("-" * 70)
    for k, v in coeffs.items():
        a = np.asarray(v, dtype=float)
        if a.ndim != 3 or a.shape[0] < 2:
            continue
        d = np.abs(a[1:] - a[0])
        dmax = float(np.nanmax(np.where(np.isfinite(d), d, np.nan))) if np.isfinite(d).any() else 0.0
        rel = dmax / (_finite_max(a[0]) + TINY)
        verdict = "FROZEN  <-- stale or init-only" if rel < rtol else "evolving"
        print(f"{k:<22}{dmax:16.4e}{rel:12.2e}   {verdict}")


# =========================================================================== #
# One-shot driver
# =========================================================================== #
def full_report(coeffs, psin, grid_or_path, jomp=44, j=None, t=-1,
                savedir="crash_diagnostics", data_dir=None, auto_harmonise=True,
                PF_limit_range=0.98, timestep=10.0, crash_key="Xe_t",
                PF_limit=True):
    """Run every diagnostic in order and save all figures to `savedir`."""
    j = jomp if j is None else j

    print("=" * 72)
    print("0  array shape / provenance consistency")
    print("=" * 72)
    chk = check_time_axes(coeffs, psin=psin, data_dir=data_dir, verbose=True)
    if chk["ragged"] and auto_harmonise:
        coeffs = harmonise(coeffs, nt=chk["nt"])
    nt = chk["nt"]
    t = _tidx(t, nt)
    j = min(max(int(j), 0), chk["ny"] - 1)
    print(f"\nusing j = {j}, t = {t}  (nt = {nt}, ny = {chk['ny']})\n")

    print("=" * 72)
    print("1  grid topology, metric floors, zShift, hyperdiffusion sizing")
    print("=" * 72)
    topo = topology_check(grid_or_path, PF_limit_range=PF_limit_range,
                          PF_limit=PF_limit, savedir=savedir)

    print("=" * 72)
    print("2  WHERE IS THE CRASH")
    print("=" * 72)
    nonfinite_report(coeffs, psin, topo, savedir=savedir)
    crash_trace(coeffs, psin, topo, timestep=timestep, savedir=savedir)
    if crash_key in coeffs:
        crash_map(coeffs, psin, topo, key=crash_key, savedir=savedir)
        spike_locator(coeffs, psin, topo, key=crash_key, savedir=savedir)
        parallel_structure(coeffs, psin, topo, key=crash_key, savedir=savedir)
        oddeven_map(coeffs, psin, topo, key=crash_key, savedir=savedir)
    poloidal_localization(coeffs, psin, t=t, jomp=jomp, topo=topo, savedir=savedir)
    background_consistency(coeffs, psin, grid_or_path, jomp=jomp, savedir=savedir)

    print("=" * 72)
    print("3  growth and grid-scale content")
    print("=" * 72)
    growth_report(coeffs, psin, timestep=timestep, savedir=savedir)
    sawtooth_report(coeffs, psin, j=j, savedir=savedir)

    print("=" * 72)
    print("4  V_ExB")
    print("=" * 72)
    if "Vexbx_txy" in coeffs:
        plot_vexb(coeffs["Vexbx_txy"], psin, topo, grid=grid_or_path, t=t,
                  jomp=jomp, savedir=savedir)
    velocity_consistency(coeffs, psin, j=j, t=t, grid=grid_or_path, savedir=savedir)

    print("=" * 72)
    print("5  equilibrium smoothness")
    print("=" * 72)
    check_grid_equilibrium(grid_or_path, jomp=jomp, savedir=savedir)
    equilibrium_kink_map(grid_or_path, topo, savedir=savedir)

    print("=" * 72)
    print("6  chi 0/0 artifact and frozen fields")
    print("=" * 72)
    chi_artifact_report(coeffs, psin, j=j, t=t, savedir=savedir)
    frozen_field_report(coeffs)

    print("\nAll figures written to:", os.path.abspath(savedir))
    return topo


INTERPRETATION = """
READING THE DIAGNOSTICS
-----------------------
crash_trace / crash_map
    Leg curves orders of magnitude above core, max pinned in the legs
      -> divertor-leg numerical mode, not ballooning.  Go to spike_locator.
    Core / upstream SOL curve leading
      -> physical peeling-ballooning; check resolution and dissipation instead.

spike_locator
    Modal index at the PF_limit cut  -> sheath BC switches to zero BCs at that
                                        index (SBC_* with PF_limit).  Set
                                        PF_limit=false and damp the PF with
                                        PF_sink instead.
    Modal index at ixsep             -> equilibrium discontinuity at the
                                        separatrix (equilibrium_kink_map).

parallel_structure
    Peak at j=0 / ny-1  -> sheath boundary.     Peak at X-point cut -> metric/zShift.

topology_check
    < 12 cells per leg, Bp floor active over many cells, zShift jump ratio > 5,
    q_loc in legs >> core: each is a grid-construction problem.

plot_vexb
    Envelope map should look like poloidal_localization.  If the 4-D input
    reports max|<V>_z| << RMS_z, the old z-averaged array was round-off.

sawtooth_report
    Max fraction pinned at 0.5 = isolated one-cell spike; 1.0 = checkerboard.
"""

if __name__ == "__main__":
    print(__doc__)
    print(INTERPRETATION)