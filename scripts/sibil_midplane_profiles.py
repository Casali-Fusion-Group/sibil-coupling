#!/usr/bin/env python3
"""
extract_midplane_profiles.py

Extract outer-midplane electron density (ne) and temperature (Te) profiles
from a set of SOLPS-ITER `balance.nc` files and plot them together.

Assumptions (override with the flags below if your setup differs):
  - Each run directory contains a `balance.nc` file (standard SOLPS-ITER
    post-processing output).
  - The outer-midplane poloidal index is stored as `jxa` in balance.nc.
    If missing, pass --jxa <index> to override.
  - Radial coordinate is the cell-center major radius R, computed from the
    `crx` corner-coordinate array (shape (4, ny, nx)) at poloidal index jxa.
    If `crx` isn't found, falls back to plotting vs. radial grid index.
  - `te` is assumed to be in eV; if values look like raw B2 units (Joules,
    ~1e-17), it's auto-converted to eV.

Usage:
    python extract_midplane_profiles.py run1 run2 run3 -o midplane_profiles.png
    python extract_midplane_profiles.py "runs/case_*/" -o profiles.png --rsep 2.15
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

try:
    import xarray as xr
except ImportError:
    xr = None
try:
    import netCDF4 as nc
except ImportError:
    nc = None

EV = 1.602176634e-19  # J per eV


def open_balance(path):
    if xr is not None:
        return xr.open_dataset(path)
    if nc is not None:
        return nc.Dataset(path)
    raise RuntimeError("Need either 'xarray' or 'netCDF4' installed to read balance.nc")


def get_var(ds, names):
    """Return the first matching variable (as a numpy array) or None."""
    for name in names:
        if name in ds.variables:
            v = ds.variables[name]
            return np.asarray(v.values if hasattr(v, "values") else v[:])
    return None


def find_jxa(ds, override=None):
    if override is not None:
        return int(override)
    val = get_var(ds, ["jxa"])
    if val is not None:
        return int(np.squeeze(val))
    raise RuntimeError(
        "Could not find 'jxa' (outer midplane poloidal index) in balance.nc. "
        "Pass --jxa <index> explicitly."
    )


def midplane_radius(ds, jxa):
    """Cell-center major radius R [m] along the radial (ny) direction at jxa."""
    crx = get_var(ds, ["crx"])
    if crx is None or crx.ndim != 3:
        return None
    try:
        return crx[:, :, jxa].mean(axis=0)  # -> shape (ny,)
    except Exception:
        return None


def extract_profiles(rundir, jxa_override=None):
    path = os.path.join(rundir, "balance.nc")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No balance.nc found in {rundir}")

    ds = open_balance(path)
    try:
        jxa = find_jxa(ds, jxa_override)

        ne = get_var(ds, ["ne"])
        te = get_var(ds, ["te"])
        if ne is None or te is None:
            raise RuntimeError(
                f"Could not find 'ne'/'te' in {path}. "
                f"Available variables: {list(ds.variables.keys())}"
            )

        # (time, ny, nx) -> take last time slice if a time dim is present
        if ne.ndim == 3:
            ne = ne[-1]
        if te.ndim == 3:
            te = te[-1]

        ne_mid = np.asarray(ne)[:, jxa]
        te_mid = np.asarray(te)[:, jxa]

        # Auto-detect raw Joule units and convert to eV
        if np.nanmax(np.abs(te_mid)) < 1e-3:
            te_mid = te_mid / EV

        r = midplane_radius(ds, jxa)
        if r is None:
            r, r_label = np.arange(ne_mid.shape[0]), "Radial grid index"
        else:
            r_label = "R at outer midplane [m]"

        return r, r_label, ne_mid, te_mid, jxa
    finally:
        ds.close()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("rundirs", nargs="+", help="SOLPS run directories (each containing balance.nc)")
    parser.add_argument("-o", "--output", default="midplane_profiles.png", help="Output plot filename")
    parser.add_argument("--jxa", type=int, default=None, help="Override outer-midplane poloidal index")
    parser.add_argument(
        "--rsep", type=float, default=None,
        help="Separatrix major radius [m]; if given, plots R - Rsep instead of R"
    )
    args = parser.parse_args()

    # Expand globs in case the shell passed a quoted pattern through
    rundirs = []
    for entry in args.rundirs:
        matches = glob.glob(entry)
        rundirs.extend(matches if matches else [entry])

    fig, (ax_ne, ax_te) = plt.subplots(1, 2, figsize=(11, 4.5))
    xlabel = "Radial grid index"
    any_ok = False

    for rundir in rundirs:
        label = os.path.basename(os.path.normpath(rundir))
        try:
            r, r_label, ne_mid, te_mid, jxa = extract_profiles(rundir, args.jxa)
        except Exception as e:
            print(f"[skip] {rundir}: {e}", file=sys.stderr)
            continue

        any_ok = True
        if args.rsep is not None and r_label.startswith("R"):
            x, xlabel = r - args.rsep, "R - Rsep [m]"
        else:
            x, xlabel = r, r_label

        ax_ne.plot(x, ne_mid, marker="o", ms=3, label=label)
        ax_te.plot(x, te_mid, marker="o", ms=3, label=label)

    if not any_ok:
        print("No runs could be processed successfully.", file=sys.stderr)
        sys.exit(1)

    ax_ne.set_xlabel(xlabel)
    ax_ne.set_ylabel(r"$n_e$ [m$^{-3}$]")
    ax_ne.set_title("Outer midplane density")
    ax_ne.legend(fontsize=8)
    ax_ne.grid(alpha=0.3)

    ax_te.set_xlabel(xlabel)
    ax_te.set_ylabel(r"$T_e$ [eV]")
    ax_te.set_title("Outer midplane temperature")
    ax_te.legend(fontsize=8)
    ax_te.grid(alpha=0.3)

    fig.tight_layout()
    outpath = os.path.join(os.getcwd(), args.output)
    fig.savefig(outpath, dpi=150)
    print(f"Saved plot to {outpath}")


if __name__ == "__main__":
    main()