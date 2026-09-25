#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 09:24:54 2025

@author: ttaczak
"""
import matplotlib.pyplot as plt
from ..readers.GQDSK_Reader import GQDSK
from .grid import Grid
from matplotlib.collections import PolyCollection
from boututils.datafile import DataFile
import numpy as np
from matplotlib.colors import LogNorm, Normalize
import copy
from matplotlib.cm import ScalarMappable

class Static_grid(Grid):
    def __init__(self, grid_name, config):
        Grid.__init__(self, grid_name)
        self.config = config
        self.R_LCFS, self.Z_LCFS, self.R_limits, self.Z_limits = self._get_LCFS_and_wall(config["files"]["gfile"])

    def _get_LCFS_and_wall(self, gfile_fname):
        g = GQDSK(gfile_fname)
        R_LCFS = np.append(g.R_LCFS, g.R_LCFS[0])
        Z_LCFS = np.append(g.Z_LCFS, g.Z_LCFS[0])
        return R_LCFS, Z_LCFS, g.R_limits, g.Z_limits

    def _make_fresh_polycollection(self):
        verts = self.grid_polygons.get_paths()
        return PolyCollection([p.vertices for p in verts], cmap='viridis',
                               edgecolors='black', facecolors='white', alpha=0.1)

    def plot_divertor(self, R_range=(1.0, 1.7), Z_range=(-1.5, -1.0), ax=None, dpi=1200):
        if ax is None:
            fig, ax = plt.subplots()
        ax = self.plot_grid(ax=ax, dpi=dpi)
        ax.set_xlim(R_range)
        ax.set_ylim(Z_range)
        return ax

    def plot_grid(self, ax=None, dpi=200):
        if ax is None:
            axis_passed = False
            fig, ax = plt.subplots()
        else:
            axis_passed = True

        ax.add_collection(self._make_fresh_polycollection())

        border = 0.1
        ax.set_xlim(self.R_limits.min() - border, self.R_limits.max() + border)
        ax.set_ylim(self.Z_limits.min() - border, self.Z_limits.max() + border)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'$R(m)$')
        ax.set_ylabel(r'$Z(m)$')
        ax.set_title(rf'{self.grid_name}')

        ax.plot(self.R_LCFS, self.Z_LCFS, 'k', linewidth=0.1)
        ax.plot(self.R_limits, self.Z_limits, 'k', linewidth=1.0, label="LCFS")
        ax.legend()

        if not axis_passed:
            plt.show()
            return None
        else:
            return ax

    # ------------------------------------------------------------------
    # Known/named profiles. Add entries here to keep using plot_profile("x")
    # ------------------------------------------------------------------
    def _known_profile(self, var_name):
        table = {
            "ni": (lambda: self.ni, "$n_i$", "$m^{-3}$", 2e18, 4e20),
            "ne": (lambda: self.ne, "$n_e$", "$m^{-3}$", 2e18, 4e20),
            "Ti": (lambda: self.Ti, "$T_i$", "$eV$", 0.0, 1400),
            "Te": (lambda: self.Te, "$T_e$", "$eV$", 0.0, 1400),
        }
        return table.get(var_name)

    def plot_profile(self, var_name=None, data=None, ax=None, scale="log", cmap_name="jet",
                      pvar_name=None, pvar_unit=None, pval_min=None, pval_max=None):
        """
        Plot a scalar field on the grid.

        Two ways to call it:
          1) Named/known field (unchanged behavior):
                 mapper.bout.plot_profile("ni")
          2) Arbitrary array (e.g. a transport coefficient loaded from disk),
             already interpolated onto this grid's points:
                 mapper.bout.plot_profile(data=D_perp, pvar_name="$D_\\perp$",
                                           pvar_unit="$m^2/s$", scale="log")

        `pvar_name`, `pvar_unit`, `pval_min`, `pval_max` override defaults for
        known fields too, and are required only if `data` is given and you
        want a friendly label (otherwise sensible fallbacks are used and
        min/max are inferred from the data).
        """
        if ax is None:
            axis_passed = False
            fig, ax = plt.subplots()
        else:
            axis_passed = True
            fig = ax.get_figure()

        # ---- resolve which values to plot -----------------------------
        if data is not None:
            pval = np.asarray(data)
            pvar_name = pvar_name if pvar_name is not None else (var_name or "field")
            pvar_unit = pvar_unit if pvar_unit is not None else ""

            finite = pval[np.isfinite(pval)]
            if scale == "log":
                finite = finite[finite > 0]
            if pval_min is None:
                pval_min = float(finite.min()) if finite.size else 0.0
            if pval_max is None:
                pval_max = float(finite.max()) if finite.size else 1.0

        else:
            known = self._known_profile(var_name)
            if known is None:
                raise ValueError(
                    f"Variable {var_name!r} not recognized/implemented for plotting. "
                    f"Pass `data=<array>` to plot an arbitrary field."
                )
            getter, default_name, default_unit, default_min, default_max = known
            pval = getter()
            pvar_name = pvar_name if pvar_name is not None else default_name
            pvar_unit = pvar_unit if pvar_unit is not None else default_unit
            pval_min = pval_min if pval_min is not None else default_min
            pval_max = pval_max if pval_max is not None else default_max

        # ---- color mapping ---------------------------------------------
        if scale == "log":
            norm = LogNorm(vmin=pval_min, vmax=pval_max)
        else:
            norm = Normalize(vmin=pval_min, vmax=pval_max)

        cmap = plt.get_cmap(cmap_name)

        # ---- build colored polygons --------------------------------------
        paths = self.grid_polygons.get_paths()
        verts = [p.vertices for p in paths]

        if self.grid_name == "SOLPS-ITER":
            flat = pval.flatten(order="F")
        elif self.grid_name == "BOUT++":
            flat = pval.flatten(order="C")
        else:
            flat = pval.flatten()

        # ------------------------------------------------------------------
        # HARD CHECK: PolyCollection does NOT raise an error if len(facecolors)
        # != len(paths) -- it silently applies colors modulo/truncated across
        # polygons, which looks exactly like a "blank" or garbled plot with no
        # traceback. Catch that mismatch here explicitly instead of guessing.
        # ------------------------------------------------------------------
        if flat.size != len(verts):
            raise ValueError(
                f"Shape mismatch: data has {flat.size} values (shape={pval.shape}, "
                f"grid_name={self.grid_name!r}) but grid_polygons has {len(verts)} "
                f"polygons. These MUST match 1:1 or the plot will silently render "
                f"blank/garbled with no error. This usually means the array you're "
                f"plotting was built using the wrong grid dimensions (e.g. a "
                f"transport-block size like nx=42) instead of the actual mesh "
                f"resolution backing grid_polygons. Check "
                f"len(self.grid_polygons.get_paths()) against whatever nx/ny you "
                f"used to build this array."
            )

        facecolors = cmap(norm(flat))

        ppolygons = PolyCollection(
            verts,
            facecolors=facecolors,
            edgecolors='none',
            linewidths=0.1,
            alpha=0.5
        )

        ax.add_collection(ppolygons)

        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(flat)

        border = 0.1
        ax.set_xlim(self.R_limits.min() - border, self.R_limits.max() + border)
        ax.set_ylim(self.Z_limits.min() - border, self.Z_limits.max() + border)

        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'$R(m)$')
        ax.set_ylabel(r'$Z(m)$')
        ax.set_title(rf'{self.grid_name} {pvar_name}')

        ax.plot(self.R_LCFS, self.Z_LCFS, 'k', linewidth=0.1)
        ax.plot(self.R_limits, self.Z_limits, 'k', linewidth=1.0, label="LCFS")
        ax.legend()

        if not axis_passed:
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(rf"{pvar_unit}")
            plt.tight_layout()
            plt.show()
            return None
        else:
            return ax, sm, pvar_unit

    # ------------------------------------------------------------------
    # Convenience wrapper specifically for transport coefficients
    # ------------------------------------------------------------------
    def plot_transport_coeff(self, data, name, unit="", ax=None, scale="log",
                              cmap_name="jet", vmin=None, vmax=None):
        """
        e.g. mapper.bout.plot_transport_coeff(D_perp, name="$D_\\perp$", unit="$m^2/s$")
        """
        return self.plot_profile(
            data=data, ax=ax, scale=scale, cmap_name=cmap_name,
            pvar_name=name, pvar_unit=unit, pval_min=vmin, pval_max=vmax
        )
