#!/usr/bin/env python3
"""
One-slide HERMES-3 transport figure.

  (a) outboard-midplane D, chi_e, chi_i over a late time window
  (b) D at the outboard midplane vs time, per psi_n band            (needs --numpy)

Coefficients are flux-gradient ratios of AVERAGED quantities,
    D = -<Gamma> / <dn/dr>,   chi = -<q> / (<n> <dT/dr>),
with <.> over the time window (a) or a sliding window of outputs (b) and over a few
radial cells.  Per-output ratios (Da_t) flip sign wherever the instantaneous flux is
up-gradient, which is what made the raw median profile a comb on a log axis.
The band in (a) is the spread of the same estimate over 4 sub-windows.

Run from the directory above the HERMES run (figure goes to ./figs):
    python hermes_slide_figure.py ./HERMES
"""
import argparse
import os
import sys

import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sibil_talk_figures import RC, CAT, INK2, finish, save, sep_line   # noqa: E402

LAB = {"D": r"$D_\perp$", "Xe": r"$\chi_e$", "Xi": r"$\chi_i$"}
BANDS = (((0.97, 1.00), r"pedestal  $\psi_n$ 0.97-1.00"),
         ((1.00, 1.02), r"near SOL  $\psi_n$ 1.00-1.02"))
BAND_COL = ("#0d366b", "#6da7ec")      # both D_perp: dark / light blue
REL_EPS = 0.02


def find_grid(run_dir="."):
    """Grid named in BOUT.inp: top-level `grid =` or [mesh] `file =`."""
    section = ""
    for line in open(os.path.join(run_dir, "BOUT.inp")):
        s = line.split("#")[0].strip()
        if s.startswith("["):
            section = s.strip("[] ").lower()
            continue
        if "=" not in s:
            continue
        k, v = (x.strip() for x in s.split("=", 1))
        if (section == "" and k.lower() == "grid") or (section == "mesh" and k.lower() == "file"):
            p = os.path.expandvars(v.strip('"').strip("'"))
            return p if os.path.isabs(p) else os.path.join(run_dir, p)
    raise SystemExit("no grid in BOUT.inp; pass --grid")


def load_grid(path):
    from netCDF4 import Dataset
    with Dataset(path) as g:
        R = np.asarray(g["Rxy"][:], float)
        psi = np.asarray(g["psixy"][:], float)
        pa = float(np.ravel(g["psi_axis"][...])[0])
        pb = float(np.ravel(g["psi_bdry" if "psi_bdry" in g.variables else "psi_bndry"][...])[0])
        ix = int(np.ravel(g["ixseps1"][...])[0])
        j0 = int(np.ravel(g["jyseps1_1"][...])[0]) + 1
        j1 = int(np.ravel(g["jyseps2_2"][...])[0]) + 1
    jomp = j0 + int(np.argmax(R[min(ix, R.shape[0] - 1), j0:j1]))
    return (psi[:, jomp] - pa) / (pb - pa), jomp


def smooth(a, w, axis=0):
    """nan-aware running mean of width w along axis."""
    if w <= 1:
        return a
    a = np.moveaxis(np.asarray(a, float), axis, 0)
    ok = np.isfinite(a)
    k = np.ones(w)
    num = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, np.where(ok, a, 0.0))
    den = np.apply_along_axis(lambda v: np.convolve(v, k, "same"), 0, ok.astype(float))
    return np.moveaxis(np.where(den > 0, num / np.maximum(den, 1), np.nan), 0, axis)


SIGNED = True          # draw up-gradient stretches (D < 0) dashed as |D| instead of leaving gaps


def ratio(num, den):
    """-num/den (signed), NaN where the gradient is flat."""
    scale = np.nanmax(np.abs(den))
    out = -num / np.where(np.abs(den) > REL_EPS * scale, den, np.nan)
    return out if SIGNED else np.where(out > 0, out, np.nan)


def signed_line(ax, x, y, c, lw, label=None, z=3):
    """Solid where y > 0; dashed |y| where y < 0 (net flux up-gradient)."""
    y = np.asarray(y, float)
    pos, neg = np.where(y > 0, y, np.nan), np.where(y < 0, -y, np.nan)
    for v, ok in ((pos, y > 0), (neg, y < 0)):        # extend each run by one point so the
        e = ok | np.r_[ok[1:], False] | np.r_[False, ok[:-1]]    # solid/dashed pieces join
        v[e & ~ok] = np.abs(y[e & ~ok])
    ax.plot(x, pos, color=c, lw=lw, label=label, zorder=z)
    if np.isfinite(neg).any():
        ax.plot(x, neg, color=c, lw=lw * 0.8, ls=(0, (2.5, 1.8)), alpha=0.75, zorder=z - 1)
        return True
    return False


def neg_legend(ax, loc, **kw):
    from matplotlib.lines import Line2D
    h, l = ax.get_legend_handles_labels()
    h.append(Line2D([], [], color=INK2, lw=1.8, ls=(0, (2.5, 1.8))))
    l.append("dashed: net inward flux")
    ax.legend(h, l, loc=loc, **kw)


def load_series(nd, jomp):
    """(t, x) OMP flux, gradient and density arrays for D, chi_e, chi_i."""
    def ld(k):
        p = os.path.join(nd, f"{k}.npy")
        return np.asarray(np.load(p, mmap_mode="r")[:, :, jomp], float) if os.path.exists(p) else None
    n = ld("ni_txy")
    S = {"D": (ld("Gamma_txy"), ld("ddr_nitot_txy"), None),
         "Xe": (ld("qe_txy"), ld("ddr_Tetot_txy"), n),
         "Xi": (ld("qi_txy"), ld("ddr_Titot_txy"), n)}
    return {k: v for k, v in S.items() if v[0] is not None and v[1] is not None}


def coeff(flux, grad, n, sl, wx):
    f = smooth(np.nanmean(flux[sl], 0), wx)
    g = np.nanmean(grad[sl], 0) * (np.nanmean(n[sl], 0) if n is not None else 1.0)
    return ratio(f, smooth(g, wx))


def profile_from_numpy(S, nt, window, wx):
    lo, hi = window if window else (nt - max(4, nt // 3), nt)
    prof = {}
    for k, (fl, gr, n) in S.items():
        mid = np.multiply(coeff(fl, gr, n, slice(lo, hi), wx),10)
        edges = np.linspace(lo, hi, 5).astype(int)
        subs = np.multiply(np.array([coeff(fl, gr, n, slice(a, b), wx) for a, b in zip(edges[:-1], edges[1:])]),10)
        prof[k] = (mid, np.nanmin(np.abs(subs), 0), np.nanmax(np.abs(subs), 0))
    return prof, (lo, hi)


def auto_ylim(vals):
    v = np.abs(np.concatenate([np.ravel(a) for a in vals]))
    v = v[np.isfinite(v) & (v > 0)]
    return 10 ** np.floor(np.log10(np.percentile(v, 10))), 10 ** np.ceil(np.log10(2.5 * v.max()))


def shared_ylim(prof, x, xlim, decades):
    """Same y range on both panels: `decades` below the profile peak up to 3x the peak."""
    inx = (x >= xlim[0]) & (x <= xlim[1])
    peak = max(np.nanmax(np.abs(p[0][inx])) for p in prof.values())
    return peak / 10 ** decades, 3.0 * peak


def panel_profile(ax, x, prof, ref, title, ylim, xlim, spread):
    if ref:
        ax.axhspan(ref[0], ref[1], color="#efeee9", zorder=0)
        ax.text(0.02, np.sqrt(ref[0] * ref[1]), r"$D_\perp$ given to SOLPS-ITER" "\n"
                "(elm-6f coupled iterations)", transform=ax.get_yaxis_transform(),
                va="center", ha="left", fontsize=12, color=INK2)
    anyneg = False
    for (k, (mid, lo, hi)), c in zip(prof.items(), CAT):
        if spread:
            ax.fill_between(x, lo, hi, color=c, alpha=0.12, lw=0, zorder=1)
        anyneg |= signed_line(ax, x, mid, c, 2.6, LAB[k])
    ax.set_yscale("log")
    inx = (x >= xlim[0]) & (x <= xlim[1])
    ax.set_ylim(*(ylim or auto_ylim([p[0][inx] for p in prof.values()])))
    ax.set_xlim(*xlim)
    sep_line(ax)
    ax.set_xlabel(r"normalised poloidal flux  $\psi_n$")
    ax.set_ylabel(r"transport coefficient  [m$^2$/s]")
    ax.set_title(title)
    if anyneg:
        neg_legend(ax, "upper right", handlelength=1.8, fontsize=12)
    else:
        ax.legend(loc="upper right", handlelength=1.8)
    finish(ax)


def panel_time(ax, S, psin, t_us, win, wt, ylim, nbands):
    fl, gr, _ = S["D"]
    anyneg, ys = False, {}
    for ((a, b), lab), c in zip(BANDS[:nbands], BAND_COL):
        m = (psin >= a) & (psin < b)
        if not m.any():
            continue
        f = smooth(np.nanmean(fl[:, m], 1), wt)
        g = smooth(np.nanmean(gr[:, m], 1), wt)
        ys[lab] = ratio(f, g)
        anyneg = signed_line(ax, t_us, ys[lab], c, 2.4, lab) or anyneg
    lo_, hi_ = ylim or auto_ylim([y[win[0]:] for y in ys.values()])
    ax.set_yscale("log")
    ax.set_ylim(lo_, hi_)
    ax.axvspan(t_us[win[0]], t_us[win[1] - 1], color="#efeee9", zorder=0)
    ax.text(0.5 * (t_us[win[0]] + t_us[win[1] - 1]), 0.985, "quasi-steady\n(averaged at left)",
            transform=ax.get_xaxis_transform(), ha="center", va="top", fontsize=11,
            color=INK2, style="italic")
    y0 = next(iter(ys.values()))                        # innermost band: onset of transport
    ref = np.nanmean(np.abs(y0[win[0]:win[1]]))
    hit = np.flatnonzero(np.abs(y0) > 0.1 * ref)
    if hit.size:
        t_on = t_us[hit[0]]
        ax.axvline(t_on, color=INK2, lw=1.0, ls=":", zorder=2)
        ax.annotate("turbulent transport\ndevelops", xy=(t_on, lo_ * 1.15), xytext=(-6, 0),
                    textcoords="offset points", ha="right", va="bottom", fontsize=11, color=INK2,
                    bbox=dict(fc="white", ec="none", pad=1.5), zorder=6)
    ax.set_xlim(t_us[0], t_us[-1])
    ax.set_xlabel(r"simulation time  [$\mu$s]")
    ax.set_ylabel(r"$D_\perp$ at outboard midplane  [m$^2$/s]")
    ax.set_title(r"$D_\perp$ at the outboard midplane vs time")
    neg_legend(ax, "upper left", fontsize=11) if anyneg else ax.legend(loc="upper left")
    finish(ax)


def main():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default=".", help="HERMES run directory (numpy/, BOUT.inp)")
    ap.add_argument("--grid", default=None, help="default: the grid named in <run>/BOUT.inp")
    ap.add_argument("--window", type=int, nargs=2, default=None, help="outputs [lo, hi) for (a)")
    ap.add_argument("--wx", type=int, default=9, help="radial smoothing [cells]")
    ap.add_argument("--wt", type=int, default=None, help="time smoothing [outputs] (default nt/120)")
    ap.add_argument("--decades", type=float, default=None,
                    help="shared y range on both panels, this many decades below the peak")
    ap.add_argument("--bands", type=int, default=2, help="radial bands in the time panel (1-2)")
    ap.add_argument("--no-spread", action="store_true", help="no sub-window spread shading in (a)")
    ap.add_argument("--ref-band", type=float, nargs=2, default=None)
    ap.add_argument("--xlim", type=float, nargs=2, default=(0.98, 1.05))
    ap.add_argument("--ylim", type=float, nargs=2, default=None, help="default: fit each panel")
    ap.add_argument("--keys", nargs="+", default=["D", "Xe"], choices=["D", "Xe", "Xi"])
    ap.add_argument("--title", default="Edge turbulence in HERMES-3: transport coefficients "
                                       "extracted for SOLPS-ITER coupling")
    ap.add_argument("--out", default="figs")
    ap.add_argument("--name", default="hermes_slide")
    ap.add_argument("--transparent", action="store_true")
    ap.add_argument("--hide-negative", action="store_true",
                    help="leave gaps where D < 0 instead of drawing |D| dashed")
    a = ap.parse_args()
    global SIGNED
    SIGNED = not a.hide_negative

    with plt.rc_context(RC):
        nd = os.path.join(a.run, "numpy")
        psin, jomp = load_grid(a.grid or find_grid(a.run))
        S = {k: v for k, v in load_series(nd, jomp).items() if k in a.keys}
        nt = next(iter(S.values()))[0].shape[0]
        tp = os.path.join(nd, "t_array.npy")
        t_us = np.load(tp)[:nt] * 1e6 if os.path.exists(tp) else np.arange(nt, dtype=float)
        prof, win = profile_from_numpy(S, nt, a.window, a.wx)
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(15, 6.0),
                                     gridspec_kw={"width_ratios": [1.1, 1]})
        ylim = a.ylim or (shared_ylim(prof, psin, a.xlim, a.decades) if a.decades else None)
        panel_profile(ax, psin, prof, a.ref_band,
                      f"Outboard midplane, average over {t_us[win[0]]:.0f}-"
                      f"{t_us[win[1] - 1]:.0f} $\\mu$s", ylim, a.xlim, not a.no_spread)
        panel_time(bx, S, psin, t_us, win, a.wt or max(1, nt // 120), ylim, a.bands)
        fig.suptitle(a.title, x=0.01, ha="left", fontsize=18)
        save(fig, a.out, a.name, a.transparent)


if __name__ == "__main__":
    main()
