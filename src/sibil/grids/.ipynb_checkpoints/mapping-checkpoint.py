#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  2 15:29:19 2025

@author: ttaczak
"""
from .grid import Grid
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as si

class Mapping(Grid):
    def __init__(self, name, bgrid, sgrid, config):
        Grid.__init__(self, name)
        self.config = config
        self.sgrid = sgrid
        self.bgrid = bgrid
        # first cublic spline interpoloation along radial direction
        self.ne=np.zeros((bgrid.nx,sgrid.ny))
        self.Te=np.zeros((bgrid.nx,sgrid.ny))
        self.ni=np.zeros((bgrid.nx,sgrid.ny))
        self.Ti=np.zeros((bgrid.nx,sgrid.ny))
        self.dl=np.zeros((bgrid.nx,sgrid.ny)) 
        self.theta=np.zeros((bgrid.nx,sgrid.ny)) 
        
        self._define_grid(name, bgrid, sgrid)
        self._calculate_radial_profiles()
        self.calc_flux_tube_lengths()
        
    def _calculate_radial_profiles(self):
        
        radial_spline_fcn = self.radial_spline_softmax
        
        self.ni = radial_spline_fcn(self.sgrid.ni, self.sgrid, self.bgrid, 
                                        self.config["softmax_scales"]["ni"][0], 
                                        self.config["softmax_scales"]["ni"][1])
        self.ne = radial_spline_fcn(self.sgrid.ne, self.sgrid,self.bgrid, 
                                        self.config["softmax_scales"]["ne"][0], 
                                        self.config["softmax_scales"]["ne"][1])
        self.Ti = radial_spline_fcn(self.sgrid.Ti, self.sgrid,self.bgrid, 
                                        self.config["softmax_scales"]["Ti"][0], 
                                        self.config["softmax_scales"]["Ti"][1])
        self.Te = radial_spline_fcn(self.sgrid.Te, self.sgrid, self.bgrid, 
                                        self.config["softmax_scales"]["Te"][0], 
                                        self.config["softmax_scales"]["Te"][1])
        
    def radial_spline_softmax(self, var, sgrid, bgrid, scale, threshold, linear_cutoff_ind=1):
        interp_var = np.zeros((bgrid.nx, sgrid.ny))
        for j in range(self.ny):
            if j < self.jyseps11+1 or j > sgrid.jyseps22:
                x_interp = bgrid.psinxy[:,bgrid.jyseps11-1]
            else:
                x_interp = bgrid.psinxy[:,bgrid.jyseps11+1]
                
            inner_interp_func = si.interp1d(sgrid.psinxy[:-linear_cutoff_ind, j], var[:-linear_cutoff_ind, j], kind='quadratic', fill_value='extrapolate')
            outer_interp_func = si.interp1d(sgrid.psinxy[:-linear_cutoff_ind, j], var[:-linear_cutoff_ind, j], kind='linear', fill_value='extrapolate')
            outside_range = (x_interp > np.max(sgrid.psinxy[:-linear_cutoff_ind, j])) | (x_interp < np.min(sgrid.psinxy[:, j]))
            interpolated_values = np.where(outside_range, outer_interp_func(x_interp), inner_interp_func(x_interp))
            interpolated_values = softmax(interpolated_values,threshold,scale=scale)
            
            interp_var[:, j] = interpolated_values

        return interp_var
    
    
        
    def calc_flux_tube_lengths(self):
        
        self.dl2d=np.zeros((self.nx,self.ny)) 
        self.dl2d[:,self.j_pfr] = np.cumsum(self.dl[:,self.j_pfr],axis=1)
        self.dl2d[:,self.j_cfs] = np.cumsum(self.dl[:,self.j_cfs],axis=1)
        
        # for i in range(self.bgrid.nx):
        #     if (i<self.bgrid.ixsep): 
                
        #         # Flux tube length including both closed flux surface and privite flux regions
                
        #         # Closed flux surface 
        #         dl1d = np.cumsum(self.dl[i,self.j_cfs]) 
        #         self.dl2d[i,self.j_cfs] = dl1d
                
        #         # dl1d_end = self.bgrid.dl2d[i,self.bgrid.jyseps22]
        #         # dl1d_end = dl1d[-1]
                
        #         # # rescale
        #         # dl1d_rescaled = dl1d * (dl1d_end / dl1d_end)
        #         # self.dl2d[i,self.j_cfs] = dl1d_rescaled
                
        #         # for privite flux surface region 
        #         dl1d = np.cumsum(self.dl[i,self.sgrid.j_pfr]) #-np.sum(self.dl[i,:self.jyseps11+1]) 
        #         self.dl2d[i,self.j_pfr] = np.copy(dl1d)
                
        #     else:
        #         # dl1d=np.cumsum(self.dl[i,:])
        #         # dl1d = np.cumsum(self.bgrid.dl[i,:]) 
                
        #         # dl1d=dl1d*self.bgrid.dl1d[-1]/dl1d[-1]
        #         # self.dl2d[i,:]=np.copy(dl1d)
                
        #         dl1d=np.cumsum(self.dl[i,:])
        #         self.dl2d[i,:]=dl1d
        #         # t_omp = self.theta[i,self.jomp]
        #         # t_omp_p = self.theta[i,self.jomp+1]
        #         # t_omp_m = self.theta[i,self.jomp-1]
        #         # h_omp = self.dl[i,self.jomp]
        #         # h_omp_p = self.dl[i,self.jomp+1]
        #         # # calculate offset wrt omp
        #         # toff=np.sum(self.dl[i,:self.jomp+1])
        #         # if (self.theta[i,self.jomp]<0):
        #         #     toff=toff-t_omp/(t_omp_p-t_omp)*h_omp_p
        #         # else:
        #         #     toff=toff-t_omp_m/(t_omp-t_omp_m)*h_omp
                    
        #         # dl1d=dl1d-toff
        #         # self.dl2d[i,:]=dl1d
                
    def renormalize_flux_tube_lengths(self,bgrid):
        for i in range(self.nx):
            if i == self.ixsep:
                continue
            # if i < self.ixsep:
                
            # self.dl[i,self.j_pfr][-1]/2 aligns the end of the gridcell in the divertor region with the end of the gridcell in BOUT++
            sdl_max_pfr = np.max(self.dl2d[i,self.j_pfr]) #+ self.dl[i,self.j_pfr][-1]/2
            bdl_max_pfr = np.max(bgrid.dl2d[i,bgrid.j_pfr]) #+ bgrid.dl[i,bgrid.j_pfr][-1]/2
            self.dl2d[i,self.j_pfr] = self.dl2d[i,self.j_pfr] * bdl_max_pfr / sdl_max_pfr
            
            # self.dl[i,self.j_cfs][-1]/2 aligns the end of the gridcell in the CFS region with the end of the gridcell in BOUT++
            sdl_max_cfs = self.dl2d[i,self.j_cfs][-1] + self.dl[i,self.j_cfs][-1]
            bdl_max_cfs = bgrid.dl2d[i,bgrid.j_cfs][-1] + bgrid.dl[i,bgrid.j_cfs][-1]
            self.dl2d[i,self.j_cfs] = self.dl2d[i,self.j_cfs] * bdl_max_cfs / sdl_max_cfs
            # else:
            #     sdl_max = np.max(self.dl2d[i,:]) + self.dl[i,:][-1]/2
            #     bdl_max = np.max(bgrid.dl2d[i,:]) + bgrid.dl[i,:][-1]/2
            #     self.dl2d[i,:] = self.dl2d[i,:] * bdl_max / sdl_max
        
    def _define_grid(self,name,bgrid,sgrid):
        self.grid_name = name
        
        self.nx = bgrid.nx
        self.ny = sgrid.ny
        self.nxy = self.nx * self.ny
        
        self.dl2d=np.zeros((self.nx,self.ny)) 
        
        self.dl = sgrid.radial_spline_interpolation(sgrid.dl, bgrid, slinear_cutoff_ind = 0)
        self.theta = sgrid.radial_spline_interpolation(sgrid.theta, bgrid, threshold=-1000)
        # FINISH DEFINING ALL VARIABLES LATER
        
        self.ni = None
        self.ne = None
        self.Ti = None
        self.Te = None
        
        # self.nx = None
        # self.ny = None
        # self.nxy = None
        
        # self.rxy = None
        # self.zxy = None
        # self.zomp = None
        # self.romp = None

        # self.ixsep = None
        # self.jyseps11 = None
        # self.jyseps22 = None
        self.jomp = sgrid.jomp
        
        self.j_pfr = sgrid.j_pfr
        self.j_cfs = sgrid.j_cfs
        
        self.ixsep = bgrid.ixsep
        self.jyseps11 = sgrid.jyseps11
        self.jyseps22 = sgrid.jyseps22
        # self.jomp = bgrid.jomp

        self.psi_axis = bgrid.psi_axis
        self.psixy = bgrid.psixy
        
        ###### CLEAN THIS UP LATER! #######
        self.psinxy = np.zeros((self.nx, self.ny))
        for j in range(self.ny):
            if j in self.j_pfr:
                self.psinxy[:,j] = bgrid.psinxy[:,0]
            elif j in self.j_cfs:
                self.psinxy[:,j] = bgrid.psinxy[:,bgrid.jomp]
        
        # self.psin_max_pfr = None
        # self.psin_min_pfr = None
        
    def plot_radial_profiles(self,choord,solps,bout):
        fig, axs = plt.subplots()
        
        axs.scatter(self.psinxy[choord,:],self.ni[choord,:],marker='s', facecolors='none', edgecolors='b',label='SOLPS')
        axs.scatter(bout.psinxy[:,int(choord/98*64)],self.ni[:,choord],marker='o', facecolors='none', edgecolors='r',label='BOUT++ (x only)')
        axs.axis('tight')
        axs.set_ylim(bottom=0)
        axs.set_xlabel(r'$\psi_n$')
        axs.set_ylabel(rf'$n_i$')
        axs.legend()
        
    def plot_extrapolation(self,plotted_vals,tplotted_vals,choords,bout,names=[],units=[],dpi=300):
        '''
        Parameters
        ----------
        plotted_vals : list of NP ARRAYS
            VARIABLE TO BE PLOTTED.
        tplotted_vals : list of NP ARRAYS
            VARIABLE FOR COMPARISON WITH SOLPS VALUES ALONG SPLINE
        choords : TYPE, optional
            CHOORD TO BE PLOTTED IN THE BOUT++ Y AXIS. The default is [bout.ixsep].
        name : TYPE, optional
            NAME FOR THE Y AXIS OF THE SUBPLOT. NOT USED IF EMPTY. The default is [].
        units : TYPE, optional
            UNIT TO BE INCLUDED IN THE Y LABEL. NOT USED IF EMPTY. The default is [].
        dpi : INT
            THE DPI USED FOR THE GRAPH. The default is 300.

        Returns
        -------
        None.

        '''    
        n_rows = len(choords)
        height_per_row = 2
        n_cols = len(plotted_vals)
        width_per_col = 4
        fig, axs = plt.subplots(n_rows,n_cols,figsize=(n_cols * width_per_col, n_rows * height_per_row), dpi=dpi)
        
        # Force axs to always be 2D
        if n_rows == 1 and n_cols == 1:
            axs = np.array([[axs]])
        elif n_rows == 1:
            axs = axs[np.newaxis, :]
        elif n_cols == 1:
            axs = axs[:, np.newaxis]
            
        if not (len(names) != len(units) or len(names) == 0):
            for i,(var,tvar,name,unit) in enumerate(zip(plotted_vals,tplotted_vals,names,units)):
                for j,choord in enumerate(choords):
                    if choord < bout.ixsep:
                        # Including both PFR and CFS regions
                        axs[j,i].plot(self.dl2d[choord,self.j_pfr],tvar[choord,self.j_pfr],marker='s', c='b', markersize=5,markerfacecolor='none', label="SOLPS")
                        axs[j,i].plot(bout.dl2d[choord,bout.j_pfr],var[choord,bout.j_pfr],marker='o',c='r', markersize=5,markerfacecolor='none', label="BOUT++")
                        
                        ylow  = 0.9*min(np.min(tvar[choord,bout.j_pfr]), np.min(var[choord,bout.j_pfr]))
                        #yhigh = max(np.max(tvar[choord,bout.j_pfr]), np.max(var[choord,bout.j_pfr]))
                        axs[j,i].set_ylim(bottom=ylow, top = 1.2*np.max(tvar[choord,self.j_pfr]))
                        # axs[j,i].set_ylim(bottom=np.min(tvar[choord,self.j_pfr]), top = 1.2*np.max(tvar[choord,self.j_pfr]))
                        # axs[j,i].plot(self.dl2d[choord,self.j_cfs],tvar[choord,self.j_cfs],marker='s',c='b',markersize=5,markerfacecolor='none', label="SOLPS")
                        # axs[j,i].plot(bout.dl2d[choord,bout.j_cfs],var[choord,bout.j_cfs],marker='o',c='r',markersize=5,markerfacecolor='none', label="BOUT++")
                    else:
                        axs[j,i].plot(self.dl2d[choord,:],tvar[choord,:],marker='s',c='b', markersize=5,markerfacecolor='none', label="SOLPS")
                        axs[j,i].plot(bout.dl2d[choord,:],var[choord,:],marker='o',c='r', markersize=5,markerfacecolor='none', label="BOUT++")
                        
                    axs[j,i].set_xlabel(r"$s(m)$")
                    axs[j,i].set_ylabel(rf"${name} @ {choord} ~ {unit}$")
                    axs[j,i].legend()
            plt.show()
        else:
            
            raise ValueError("Len(names) and len(units) needs to match len(vals)")
            
    def plot_extrapolation(self,plotted_vals,tplotted_vals,choords,bout,names=[],units=[],dpi=300):
        '''
        Parameters
        ----------
        plotted_vals : list of NP ARRAYS
            VARIABLE TO BE PLOTTED.
        tplotted_vals : list of NP ARRAYS
            VARIABLE FOR COMPARISON WITH SOLPS VALUES ALONG SPLINE
        choords : TYPE, optional
            CHOORD TO BE PLOTTED IN THE BOUT++ Y AXIS. The default is [bout.ixsep].
        name : TYPE, optional
            NAME FOR THE Y AXIS OF THE SUBPLOT. NOT USED IF EMPTY. The default is [].
        units : TYPE, optional
            UNIT TO BE INCLUDED IN THE Y LABEL. NOT USED IF EMPTY. The default is [].
        dpi : INT
            THE DPI USED FOR THE GRAPH. The default is 300.

        Returns
        -------
        None.

        '''    
        n_rows = len(choords)
        height_per_row = 2
        n_cols = len(plotted_vals)
        width_per_col = 4
        fig, axs = plt.subplots(n_rows,n_cols,figsize=(n_cols * width_per_col, n_rows * height_per_row), dpi=dpi)
        
        # Force axs to always be 2D
        if n_rows == 1 and n_cols == 1:
            axs = np.array([[axs]])
        elif n_rows == 1:
            axs = axs[np.newaxis, :]
        elif n_cols == 1:
            axs = axs[:, np.newaxis]
            
        if not (len(names) != len(units) or len(names) == 0):
            for i,(var,tvar,name,unit) in enumerate(zip(plotted_vals,tplotted_vals,names,units)):
                for j,choord in enumerate(choords):
                    if choord < bout.ixsep:
                        # Including both PFR and CFS regions
                        axs[j,i].plot(self.dl2d[choord,self.j_pfr],tvar[choord,self.j_pfr],marker='s', c='b', markersize=5,markerfacecolor='none', label="SOLPS")
                        axs[j,i].plot(bout.dl2d[choord,bout.j_pfr],var[choord,bout.j_pfr],marker='o',c='r', markersize=5,markerfacecolor='none', label="BOUT++")
                        
                        ylow  = 0.9*min(np.min(tvar[choord,bout.j_pfr]), np.min(var[choord,bout.j_pfr]))
                        #yhigh = max(np.max(tvar[choord,bout.j_pfr]), np.max(var[choord,bout.j_pfr]))
                        axs[j,i].set_ylim(bottom=ylow, top = 1.2*np.max(tvar[choord,self.j_pfr]))
                        # axs[j,i].set_ylim(bottom=np.min(tvar[choord,self.j_pfr]), top = 1.2*np.max(tvar[choord,self.j_pfr]))
                        # axs[j,i].plot(self.dl2d[choord,self.j_cfs],tvar[choord,self.j_cfs],marker='s',c='b',markersize=5,markerfacecolor='none', label="SOLPS")
                        # axs[j,i].plot(bout.dl2d[choord,bout.j_cfs],var[choord,bout.j_cfs],marker='o',c='r',markersize=5,markerfacecolor='none', label="BOUT++")
                    else:
                        axs[j,i].plot(self.dl2d[choord,:],tvar[choord,:],marker='s',c='b', markersize=5,markerfacecolor='none', label="SOLPS")
                        axs[j,i].plot(bout.dl2d[choord,:],var[choord,:],marker='o',c='r', markersize=5,markerfacecolor='none', label="BOUT++")
                        
                    axs[j,i].set_xlabel(r"$s(m)$")
                    axs[j,i].set_ylabel(rf"${name} @ {choord} ~ {unit}$")
                    axs[j,i].legend()
            plt.show()
        else:
            
            raise ValueError("Len(names) and len(units) needs to match len(vals)")
        
    def plot_multi_BOUT_midplane_profiles(self, choords, names_bplotting, bout):
        bvars = [bout.ni, bout.Ti, bout.Te]
        tvars = [self.ni, self.Ti, self.Te]
        var_names = ["ni", "Ti", "Te"]
        fig, axs = plt.subplots(3,3,figsize=(9,9),dpi = 300)
        for j, (choord, name) in enumerate(zip(choords, names_bplotting)):
            for i, (name_bplotting, bvar, tvar, var_name) in enumerate(zip(names_bplotting,bvars, tvars, var_names)):
                
                # Plot BOUT++ and SOLPS++ values
                # axs[i,j].scatter(bout.psinxy[:, choord], bvar[:, choord], marker='s',s=10, facecolors='salmon', edgecolors='r', label=f"BOUT++")
                axs[i,j].scatter(bout.psinxy[:, choord], tvar[:, choord], marker='s',s=10, facecolors='blue', edgecolors='b', label=f"Map")
                # axs[i,j].plot(bout.psinxy[:, choord], bvar[:, choord], color='darkred', linewidth=0.5)
                
                
                # # Example j-index groups
                # jx = bout.j_pfr
                # jjx = bout.j_cfs
                
                # # Select columns for each group from both datasets
                # A_jx = bout.dl2d[:, jx]  # shape (N, len(jx))
                # A_jjx = bout.dl2d[:, jjx]  # shape (N, len(jjx))
                # B = self.dl2d  # shape (N, Q)
                
                # # Compute differences via broadcasting
                # diff_jx = np.abs(A_jx[:, :, None] - B[:, None, :])    # shape (N, len(jx), Q)
                # diff_jjx = np.abs(A_jjx[:, :, None] - B[:, None, :])  # shape (N, len(jjx), Q)
        
                # # Find the indices of minimum difference along B's axis
                # closest_jx = np.argmin(diff_jx, axis=2)     # shape (N, len(jx))
                # closest_jjx = np.argmin(diff_jjx, axis=2)   # shape (N, len(jjx))
                
                # # Initialize matched values array
                # matched_B_values = np.zeros_like(bout.dl2d)
                
                # # Advanced indexing to fetch matched B values
                # rows = np.arange(bout.dl2d.shape[0])[:, None]  # shape (N, 1)
                
                # matched_B_values[:, jx] = tvar[rows, closest_jx]
                # matched_B_values[:, jjx] = tvar[rows, closest_jjx]
                
                # # Optional: reassemble closest_indices array like before
                # closest_indices = np.full(bout.dl2d.shape, -1, dtype=int)
                # closest_indices[:, jx] = closest_jx
                # closest_indices[:, jjx] = closest_jjx
                
                # # Plot SOLPS++ matched values with color based on closest tvar index
                # colorbar_spread = np.mean(closest_indices[:, choord])
                # # ax.plot(bout.psinxy[:, choord], matched_B_values[:, choord], color='blue', linewidth=0.5)
                # sc = axs[i,j].scatter(bout.psinxy[:, choord], matched_B_values[:, choord],
                #                 c=closest_indices[:, choord], cmap='viridis',vmin=colorbar_spread-3, vmax=colorbar_spread+3, marker='s',
                #                 edgecolors='none', s=10, facecolors='none', label='SOLPS++')
                
                
                # # Add colorbar to indicate which index from tvar was chosen
                # cbar = plt.colorbar(sc, ax=axs[i,j])
                # # cbar.set_label('SOLPS Index')
                
                # ax.set_ylim(bottom=0)  # Y-axis from 0 up
                axs[i,j].set_xlabel(r'$\psi_n$')
                axs[i,j].set_ylabel(f"{var_name}, j={choord}")
                axs[i,j].legend()
            
        fig.suptitle(f"BOUT++ radial profiles")
        plt.show()

def softmax(x, x_min, scale):
    return x_min + scale * np.logaddexp(0, (x - x_min) / scale)

