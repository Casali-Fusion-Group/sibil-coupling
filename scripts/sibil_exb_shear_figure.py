#!/usr/bin/env python3
"""
sibil_exb_shear_figure.py -- one figure: is the equilibrium ExB shear what
holds the pedestal (and so D, chi inside the separatrix) down?

Run from bout-runs after bout_save:
    python sibil_exb_shear_figure.py --dbg <dir with sibil_debug_diagnostics.py>

Controlled pair: r9.1 (diamag_phi0 = true) vs r10.0 (identical but
diamag_phi0 = false).  r9.7 and any finished r11.* runs are overlaid as further
shear-on cases.  Linear gamma comes from r10.4 / r10.5 (n = 10, 20).

  A  OMP E_r (diamag, what the runs use) and the Hahm-Burrell shearing rate
     omega_E = (R Bp)^2/B d/dpsi (E_r / R Bp), against the largest linear
     growth rate: where |omega_E| > gamma_lin the shear can quench the mode.
  B  pedestal fluctuation level vs time (max dTe_rms/Te0, psi_n 0.97-1.0,
     upstream rows), shear on vs off; SOL dashed.
  C  local pedestal growth rate gamma(t) tau_A for the same runs, with
     omega_E tau_A (pedestal median) marked.
  D  D at the OMP vs psi_n over each run's last outputs, shear on vs off, with
     the quench ratio |omega_E| / gamma_lin on the right axis.
"""
import os
import re
import sys
import glob
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from netCDF4 import Dataset

ap = argparse.ArgumentParser()
ap.add_argument("--dbg", required=True)
ap.add_argument("--root", default=".")
ap.add_argument("--out", default="exb_shear_evidence.png")
ap.add_argument("--on", nargs="*", default=["r9_scan/runs/r9.1_*", "r9_scan/runs/r9.7_*",
                                            "r11_scan/runs/r11.*"])
ap.add_argument("--off", nargs="*", default=["r10_scan/runs/r10.0_*"])
ap.add_argument("--linear", nargs="*", default=["r10_scan/runs/r10.4_*", "r10_scan/runs/r10.5_*"])
ap.add_argument("--ped", type=float, nargs=2, default=[0.97, 1.0])
ap.add_argument("--sol", type=float, nargs=2, default=[1.0, 1.03])
ap.add_argument("--last", type=int, default=10, help="outputs averaged for panel D")
ap.add_argument("--zi", type=float, default=1.0)
a = ap.parse_args()

sys.path.insert(0, os.path.abspath(a.dbg))
import sibil_debug_diagnostics as dbg  # noqa: E402

ROOT = Path(a.root).resolve()


def runs(pats):
    out = []
    for p in pats:
        out += sorted(Path(x) for x in glob.glob(str(ROOT / p))
                      if (Path(x) / "numpy").is_dir() and glob.glob(str(Path(x) / "numpy" / "*.npy")))
    return out


def grid_of(run):
    g = Path(os.path.expandvars(dbg._read_inp(str(run))["grid"]))
    if g.is_absolute():
        return g
    for c in (run / g, run.parent / g, ROOT / g):
        if c.exists():
            return c.resolve()
    return next(ROOT.rglob(g.name)).resolve()


def tstep(run):
    return float(dbg._read_inp(str(run)).get("timestep", 10.0))


def va(run):
    p = run / "numpy" / "Va.npy"
    return float(np.asarray(np.load(p)).ravel()[0]) if p.exists() else 1.542e6


on, off, lin = runs(a.on), runs(a.off), runs(a.linear)
if not on or not off:
    raise SystemExit(f"need at least one shear-on and one shear-off run with numpy/ "
                     f"(found on={[r.name for r in on]}, off={[r.name for r in off]})")

# ------------------------------------------------------------------ geometry
gpath = grid_of(on[0])
with Dataset(gpath) as g:
    G = {k: np.asarray(g[k][:], dtype=float) for k in
         ("Rxy", "Bpxy", "Btxy", "dx", "Niexp", "Tiexp", "Teexp", "psixy")}
    for k in ("psi_axis", "ixseps1", "jyseps1_1", "jyseps2_2"):
        G[k] = float(np.asarray(g[k][...]).ravel()[0])
    G["psi_bndry"] = float(np.asarray(g["psi_bdry" if "psi_bdry" in g.variables
                                        else "psi_bndry"][...]).ravel()[0])
R, Bp, dx = G["Rxy"], G["Bpxy"], G["dx"]
if dx.ndim == 1:
    dx = dx[:, None] * np.ones_like(R)
RBp, B = R * Bp, np.sqrt(G["Bpxy"] ** 2 + G["Btxy"] ** 2)
nx, ny = R.shape
ixs = int(G["ixseps1"])
j0, j1 = int(G["jyseps1_1"]) + 1, int(G["jyseps2_2"]) + 1
jomp = j0 + int(np.argmax(R[ixs, j0:j1]))
psin2 = (G["psixy"] - G["psi_axis"]) / (G["psi_bndry"] - G["psi_axis"])
x = psin2[:, jomp]
ddx = lambda f: np.gradient(f, axis=0) / dx                                   # noqa: E731
Er = RBp / (a.zi * G["Niexp"]) * ddx(G["Niexp"] * G["Tiexp"])                 # V/m
wE = RBp ** 2 / B * ddx(Er / RBp)                                             # 1/s
ped = (x >= a.ped[0]) & (x < a.ped[1])
up = np.zeros((nx, ny), bool)
up[:, j0:j1] = True
ped2 = up & (psin2 >= a.ped[0]) & (psin2 < a.ped[1])
sol2 = up & (psin2 >= a.sol[0]) & (psin2 < a.sol[1])
wE_ped = np.nanmedian(np.abs(wE[ped, jomp]))

# ------------------------------------------------------------------ linear gamma
glin = {}
for r in lin:
    red = dbg._load_reduced(str(r / "numpy"), "Te", nx, ny, verbose=False)
    if red is None:
        continue
    amp = np.nanmax(red["rms"].reshape(red["rms"].shape[0], -1), axis=1)
    nt = amp.size
    g = dbg._fit_gamma(amp, tstep(r), np.arange(nt - max(nt // 3, 3), nt))
    n = dbg._linear_setup(str(r), str(r / "numpy")).get("n_tor") or r.name
    glin[n] = (g, g * va(r))                                                   # (1/tA, 1/s)
gmax_s = max((v[1] for v in glin.values()), default=np.nan)
gmax_t = max((v[0] for v in glin.values()), default=np.nan)


# ------------------------------------------------------------------ nonlinear traces
def trace(run):
    red = dbg._load_reduced(str(run / "numpy"), "Te", nx, ny, verbose=False)
    if red is None:
        return None
    rel = red["rms"] / np.abs(G["Teexp"])[None]
    rel = np.where(np.isfinite(rel), rel, np.nan)
    tp = np.nanmax(rel[:, ped2], axis=1)
    ts = np.nanmax(rel[:, sol2], axis=1)
    dt = tstep(run)
    D = None
    p = run / "numpy" / "Da_t.npy"
    if p.exists():
        Da = np.load(p, mmap_mode="r")
        k = min(a.last, Da.shape[0])
        D = np.nanmedian(np.abs(np.asarray(Da[-k:], dtype=float))[:, :, j0:j1], axis=(0, 2))
    return dict(t=np.arange(tp.size) * dt, ped=tp, sol=ts,
                gam=dbg._sliding_gamma(tp, dt, 5), D=D, dt=dt)


data = {}
for r in on + off:
    tr = trace(r)
    if tr is not None:
        data[r.name] = (tr, r in on)

# ------------------------------------------------------------------ figure
fig, axs = plt.subplots(2, 2, figsize=(14, 9.5))
A, Bx, C, Dx = axs.ravel()

A.semilogy(x, np.abs(wE[:, jomp]), color="k", lw=2, label=r"$|\omega_E|$ (diamag $E_r$)")
for n, (gt, gs) in glin.items():
    A.axhline(gs, color="C3", ls="--", lw=1)
    A.text(x[0], gs * 1.15, rf"$\gamma_{{lin}}$ n={n}", color="C3", fontsize=9)
A.fill_between(x, 1, np.abs(wE[:, jomp]), where=np.abs(wE[:, jomp]) > gmax_s,
               color="0.85", label=r"$|\omega_E| > \gamma_{lin,max}$")
A.set_ylim(max(1e2, np.nanmin(np.abs(wE[:, jomp])[np.abs(wE[:, jomp]) > 0]) * 0.5),
           np.nanmax(np.abs(wE[:, jomp])) * 3)
A2 = A.twinx()
A2.plot(x, Er[:, jomp] / 1e3, color="C0", lw=1)
A2.set_ylabel(r"$E_r$ [kV/m]", color="C0", fontsize=12)
A.set_title(r"A  OMP shearing rate vs linear growth", fontsize=15)
A.set_ylabel(r"rate [s$^{-1}$]", fontsize=12)
A.set_xlabel(r"$\psi_n$", fontsize=12)
A.axvline(1.0, color="k", ls=":", lw=0.8)
A.legend(fontsize=8, loc="lower left")

for name, (tr, is_on) in data.items():
    col = "C0" if is_on else "C3"
    lw = 2 if name.startswith(("r9.1", "r10.0")) else 0.9
    Bx.semilogy(tr["t"], tr["ped"], color=col, lw=lw, label=f"{name} ({'shear on' if is_on else 'shear OFF'})")
    Bx.semilogy(tr["t"], tr["sol"], color=col, lw=0.7, ls="--")
    C.plot(tr["t"], tr["gam"], color=col, lw=lw, label=name)
    if tr["D"] is not None:
        Dx.semilogy(x, tr["D"], color=col, lw=lw,
                    label=f"{name}: last {min(a.last, tr['t'].size)} outputs")
Bx.set_title("B  pedestal fluctuation level (solid) and SOL (dashed)", fontsize=15)
Bx.set_ylabel(r"max $\delta T_{e,rms}/T_{e0}$", fontsize=12)
Bx.set_xlabel(r"t [$\tau_A$]", fontsize=12)
Bx.legend(fontsize=7)

C.axhline(0, color="k", lw=0.6)
C.axhline(gmax_t, color="C3", ls="--", lw=1, label=rf"$\gamma_{{lin,max}}\tau_A$ = {gmax_t:.3f}")
C.set_title(r"C  pedestal growth rate $\gamma(t)\,\tau_A$", fontsize=15)
C.set_ylabel(r"$\gamma\,\tau_A$", fontsize=12)
C.set_xlabel(r"t [$\tau_A$]", fontsize=12)
tauA = 1.0 / va(on[0])
C.text(0.02, 0.55, rf"pedestal median $|\omega_E|\,\tau_A$ = {wE_ped * tauA:.2f}"
       rf"  ($\approx${wE_ped * tauA / gmax_t:.0f}$\times\gamma_{{lin,max}}$)",
       transform=C.transAxes, fontsize=10, va="center",
       bbox=dict(facecolor="w", edgecolor="0.7"))
C.legend(fontsize=7, loc="lower right")

Dx.set_title(r"D  $D$ at the OMP (upstream-row median)", fontsize=15)
Dx.set_ylabel(r"$|D|$ [m$^2$/s]", fontsize=12)
Dx.set_xlabel(r"$\psi_n$", fontsize=12)
Dx.axvspan(*a.ped, color="0.92", zorder=0)
Dx.axvline(1.0, color="k", ls=":", lw=0.8)
D2 = Dx.twinx()
D2.semilogy(x, np.abs(wE[:, jomp]) / gmax_s, color="k", lw=1, ls="-.")
D2.axhline(1, color="k", lw=0.5)
D2.set_ylabel(r"$|\omega_E| / \gamma_{lin,max}$", fontsize=12)
Dx.legend(fontsize=7, loc="lower left")

for ax in (A, Bx, C, Dx, A2, D2):
    ax.tick_params(labelsize=9)
for ax in (A, Bx, C, Dx):
    ax.grid(True, alpha=0.3, which="both")
fig.suptitle("Equilibrium ExB shear vs pedestal turbulence: r9.1 (shear on) vs r10.0 "
             "(identical, diamag_phi0 = false)", fontsize=15)
fig.tight_layout()
fig.savefig(a.out, dpi=140)

print(f"grid {gpath.name}, jomp {jomp}")
print(f"pedestal psi_n {a.ped[0]}-{a.ped[1]}: median |omega_E| = {wE_ped:.3e} 1/s "
      f"= {wE_ped * tauA:.3f} / tau_A")
for n, (gt, gs) in glin.items():
    print(f"linear n={n}: gamma = {gt:+.4f} / tau_A = {gs:.3e} 1/s   "
          f"-> |omega_E|/gamma in pedestal ~ {wE_ped / gs:.0f}")
for name, (tr, is_on) in data.items():
    g = tr["gam"][np.isfinite(tr["gam"])]
    print(f"{name:<34} shear {'on ' if is_on else 'OFF'}  peak ped gamma tau_A "
          f"{np.nanmax(g) if g.size else np.nan:+.4f}   final ped level {tr['ped'][-1]:.2e}")
print(f"wrote {os.path.abspath(a.out)}")
