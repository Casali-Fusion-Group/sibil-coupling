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
from matplotlib.colors import LogNorm,Normalize
import copy
from matplotlib.cm import ScalarMappable

class Static_grid(Grid):
    def __init__(self,grid_name,config):
        Grid.__init__(self, grid_name)
        self.config = config
        self.R_LCFS, self.Z_LCFS, self.R_limits, self.Z_limits = self._get_LCFS_and_wall(config["gfile_loc"])
        
    
    def _get_LCFS_and_wall(self,gfile_fname):
        g = GQDSK(gfile_fname)
        R_LCFS = np.append(g.R_LCFS, g.R_LCFS[0])
        Z_LCFS = np.append(g.Z_LCFS, g.Z_LCFS[0])
        return R_LCFS, Z_LCFS, g.R_limits, g.Z_limits
    
    def _make_fresh_polycollection(self):
        verts = self.grid_polygons.get_paths()
        return PolyCollection([p.vertices for p in verts], cmap='viridis', edgecolors='black',facecolors='white',alpha=0.1)

    
    def plot_divertor(self, R_range=(1.0,1.7), Z_range=(-1.5,-1.0), ax=None, dpi=1200):
        if ax is None:
            fig, ax = plt.subplots()
        ax = self.plot_grid(ax=ax, dpi=dpi)
        ax.set_xlim(R_range)
        ax.set_ylim(Z_range)
        
        verts = self.grid_polygons.get_paths()
        return ax
    
    def plot_grid(self, ax=None, dpi=200):
        #def plot_meshes_side_by_side(mesh_points,bgrid_loc,bout,solps,g):
        
        if ax is None:
            axis_passed = False
            fig, ax = plt.subplots()
        else:
            axis_passed = True
        
        ax.add_collection(self._make_fresh_polycollection())
        
        # Adjust axis limits to better visualize the polygons
        border = 0.1 # Width of the plot border in [m]
        ax.set_xlim(self.R_limits.min() - border, self.R_limits.max() + border)
        ax.set_ylim(self.Z_limits.min() - border, self.Z_limits.max() + border)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'$R(m)$')
        ax.set_ylabel(r'$Z(m)$')
        ax.set_title(rf'{self.grid_name}')
        
        ax.plot(self.R_LCFS,self.Z_LCFS,'k',linewidth=0.1)
        ax.plot(self.R_limits,self.Z_limits,'k', linewidth = 1.0, label = "LCFS")
        ax.legend()
        
        if not axis_passed:
            plt.show()
            return None
        else:
            return ax
        
    def plot_profile(self, var_name, ax=None, scale = "log", cmap_name="jet"):
        
        if ax is None:
            axis_passed = False
            fig, ax = plt.subplots()
        else:
            axis_passed = True
            
        # pval -> plotting_value for coloring the polygons
        if var_name == "ni":
            pval = self.ni
            pvar_name = "$n_i$"
            pvar_unit = "$m^{-3}$"
            pval_min = 2e18
            pval_max = 4e20
            
        elif var_name == "ne":
            pval = self.ne
            pvar_name = "$n_e$"
            pvar_unit = "$m^{-3}$"
            pval_min = 2e18
            pval_max = 4e20
            
        elif var_name == "Ti":
            pval = self.Ti
            pvar_name = "$T_i$"
            pvar_unit = "$eV$"
            pval_min = 0.0
            pval_max = 1400
            
        elif var_name == "Te":
            pval = self.Te
            pvar_name = "$T_e$"
            pvar_unit = "$eV$"
            pval_min = 0.0
            pval_max = 1400
            
        # elif var_name == "hx":
        #     pval = self.thx2d
        #     pvar_name = "$hx$"
        #     pvar_unit = "$m$"
        #     pval_min = np.min(self.thx2d)
        #     pval_max = np.max(self.thx2d)
        
        # elif var_name == "dl":
        #     pval = self.dl2d
        #     pvar_name = "$dl$"
        #     pvar_unit = "$m$"
        #     pval_min = np.min(self.dl2d)
        #     pval_max = np.max(self.dl2d)
            
            
        else:
            raise ValueError(f"Variable {var_name} not recognized/implemented for plotting.")
        ###############
        # Create color maps
        if scale == "log":
            norm = LogNorm(vmin=pval_min, vmax=pval_max) #LogNorm(vmin=pval.min(), vmax=pval.max())
        else:
            norm = Normalize(vmin=pval_min, vmax=pval_max) #Normalize(vmin=pval.min(), vmax=pval.max())
            
        cmap = plt.get_cmap(cmap_name)  # Choose a colormap
        
        ###########
        # Create polygons and color them correctly
        paths = self.grid_polygons.get_paths()
        verts = [p.vertices for p in paths]
        
        if self.grid_name == "SOLPS-ITER":
            facecolors = cmap(norm(pval.flatten(order = "F"))) #cmap(norm(np.reshape(pval,(self.nxy,1), order = "F").flatten()))
        elif self.grid_name == "BOUT++":
            facecolors = cmap(norm(pval.flatten(order = "C"))) #cmap(norm(np.reshape(pval,(self.nxy,1), order = "C").flatten()))
        
        ppolygons = PolyCollection(
            verts,
            facecolors=facecolors,
            edgecolors='none',
            linewidths=0.1,
            alpha=0.5
        )
        
        ###########
        # Plot the results
        if ax is None:
            axis_passed = False
            fig, ax = plt.subplots()
        else:
            axis_passed = True
        
        ax.add_collection(ppolygons)
        
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array(pval)  # required for matplotlib < 3.1
        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        
        # Adjust axis limits to better visualize the polygons
        border = 0.1 # Width of the plot border in [m]
        ax.set_xlim(self.R_limits.min() - border, self.R_limits.max() + border)
        ax.set_ylim(self.Z_limits.min() - border, self.Z_limits.max() + border)
        
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel(r'$R(m)$')
        ax.set_ylabel(r'$Z(m)$')
        ax.set_title(rf'{self.grid_name} {pvar_name}')
        
        ax.plot(self.R_LCFS,self.Z_LCFS,'k',linewidth=0.1)
        ax.plot(self.R_limits,self.Z_limits,'k', linewidth = 1.0, label = "LCFS")
        ax.legend()
        
        if not axis_passed:
            cbar = fig.colorbar(sm, ax=ax)
            cbar.set_label(rf"{pvar_unit}")
            plt.tight_layout()
            plt.show()
            return None
        else:
            return ax, sm, pvar_unit
    