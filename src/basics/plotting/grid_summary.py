#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 13:53:10 2025

@author: ttaczak
"""

import basics
import matplotlib.pyplot as plt
import numpy as np

def plot_matrix(matrix, plot_name, masked = False):
    """Plots the matrix using a color plot with a square aspect ratio, handling extreme values."""
    # Mask extreme outliers for better visualization
    percent = 99
    
    # Flatten the array to determine the 10% closest to zero
    flattened = matrix.flatten()
    num_to_keep = int(0.1 * flattened.size)  # 10% of total elements
    
    # Sort indices based on value (smallest first)
    sorted_indices = np.argsort(flattened)  # Sort in ascending order
    threshold_value = flattened[sorted_indices[num_to_keep - 1]]  # Get the cutoff value
    
    # Mask all values except the lowest 10%
    masked_matrix = np.ma.masked_where(matrix > threshold_value, matrix)
    
    # # Define color limits to improve contrast
    # vmin = np.percentile(masked_matrix.compressed(), 100-percent)
    # vmax = np.percentile(masked_matrix.compressed(), percent)
    
    fig = plt.figure(dpi=800)
    ax = fig.add_subplot()
    if masked:
        plt.imshow(masked_matrix.T, cmap='viridis', interpolation='nearest')#, vmin=vmin, vmax=vmax)
    else:
        plt.imshow(matrix.T, cmap='viridis', interpolation='nearest')
    plt.colorbar(label='Value')
    plt.title(f"{plot_name}")
    plt.xlabel("x_ind")
    plt.ylabel("y_ind")
     
    # square plot
    ax.set_aspect(1.0/ax.get_data_ratio(), adjustable='box')
    plt.show()

def plot_radial_profiles(grid, 
                         var_name, 
                         choords = None):
    
    if var_name == "ni":
        var = grid.ni
        plot_name = "n_i"
        var_units = "m^{-3}"
        
    elif var_name == "Ti":
        var = grid.Ti
        plot_name = "T_i"
        var_units = "eV"
        
    elif var_name == "Te":
        var = grid.Te
        plot_name = "T_e"
        var_units = "eV"
        
    if grid.grid_name == "SOLPS":
        color = "r"
    elif grid.grid_name == "BOUT++":
        color = "b"
    else:
        color = "g"
        
    if choords is None:
        choords = [grid.jomp]
    elif type(choords) is not list or np.ndarray:
        choords = np.array(choords).tolist()
    
    return_figs = []
    for choord in choords:
        
        fig, ax = plt.subplots()
        ax.plot(grid.psinxy[:,choord], var[:,choord],marker='o',c=color, markersize=5,markerfacecolor='none', label=f"{grid.grid_name}")
        ax.set_xlabel(rf"$\rho$")
        ax.set_ylabel(rf"${plot_name} \ \ (jy={choord}) \ [{var_units}]$")
        ax.legend()
        return_figs.append(fig)
        plt.show()
        

def plot_poloidal_profiles(grid, 
                           var_name, 
                           psin_inner=0.01, 
                           psin_outer=0.90, 
                           regions = ["CFS", "SOL", "PFR"]):
    
    return_figs = []
    
    if var_name == "ni":
        var = grid.ni
        plot_name = "n_i"
        var_units = "m^{-3}"
        
    elif var_name == "Ti":
        var = grid.Ti
        plot_name = "T_i"
        var_units = "eV"
        
    elif var_name == "Te":
        var = grid.Te
        plot_name = "T_e"
        var_units = "eV"
        
    if grid.grid_name == "SOLPS":
        color = "r"
    elif grid.grid_name == "BOUT++":
        color = "b"
    else:
        color = "g"
    
    
    psin_cfs_min = np.min(grid.psinxy[:grid.ixsep,grid.j_cfs])
    psin_cfs_max = np.max(grid.psinxy[:grid.ixsep,grid.j_cfs])
    psin_sol_min = np.min(grid.psinxy[grid.ixsep+1:,:])
    psin_sol_max = np.max(grid.psinxy[grid.ixsep+1:,:])
    psin_pfr_min = np.min(grid.psinxy[:grid.ixsep,grid.j_pfr])
    psin_pfr_max = np.max(grid.psinxy[:grid.ixsep,grid.j_pfr])
    
    psin_cfs_diff = psin_cfs_max-psin_cfs_min
    psin_sol_diff = psin_sol_max-psin_sol_min
    psin_pfr_diff = psin_pfr_max-psin_pfr_min
    
    psin_cfs_p25 = psin_cfs_max - psin_inner*psin_cfs_diff
    psin_cfs_p75 = psin_cfs_max - psin_outer*psin_cfs_diff
    psin_sol_p25 = psin_sol_min + psin_inner*psin_sol_diff
    psin_sol_p75 = psin_sol_min + psin_outer*psin_sol_diff
    psin_pfr_p25 = psin_pfr_max - psin_inner*psin_pfr_diff
    psin_pfr_p75 = psin_pfr_max - psin_outer*psin_pfr_diff
    
    ix_cfs_p25 = grid.jyseps11+np.argmin(np.abs(grid.psinxy[:grid.ixsep,grid.jyseps22]-psin_cfs_p25))
    ix_cfs_p75 = grid.jyseps11+np.argmin(np.abs(grid.psinxy[:grid.ixsep,grid.jyseps22]-psin_cfs_p75))
    ix_sol_p25 = grid.ixsep+np.argmin(np.abs(grid.psinxy[grid.ixsep+1:,grid.jyseps11]-psin_sol_p25))
    ix_sol_p75 = grid.ixsep+np.argmin(np.abs(grid.psinxy[grid.ixsep+1:,grid.jyseps11]-psin_sol_p75))
    ix_pfr_p25 = np.argmin(np.abs(grid.psinxy[:grid.ixsep,grid.jyseps11]-psin_pfr_p25))
    ix_pfr_p75 = np.argmin(np.abs(grid.psinxy[:grid.ixsep,grid.jyseps11]-psin_pfr_p75))
    
    if "CFS" in regions:
        fig, axs = plt.subplots(2,1)
        fig.suptitle(r"CFR")
        
        # print(f"{psin_sol_p25:.3f}",f"{psin_sol_p75:.3f}",f"{psin_cfs_p25:.3f}",f"{psin_cfs_p25:.3f}",f"{psin_pfr_p25:.3f}",f"{psin_pfr_p25:.3f}")
        # print(ix_sol_p25,ix_sol_p75,ix_cfs_p25,ix_cfs_p75,ix_pfr_p25,ix_pfr_p75)
        
        axs[0].plot(grid.dl2d[ix_cfs_p25,grid.j_cfs],var[ix_cfs_p25,grid.j_cfs],marker='o',c=color, markersize=5,markerfacecolor='none', label=f"{grid.grid_name}")
        axs[0].set_xlabel(rf"$s(m)$")
        axs[0].set_ylabel(rf"${plot_name} \ \ (\psi_n={psin_cfs_p25:0.3}) \ [{var_units}]$")
        axs[0].legend()
        
        axs[1].plot(grid.dl2d[ix_cfs_p75,grid.j_cfs],var[ix_cfs_p75,grid.j_cfs],marker='o',c=color, markersize=5,markerfacecolor='none', label=f"{grid.grid_name}")
        axs[1].set_xlabel(rf"$s(m)$")
        axs[1].set_ylabel(rf"${plot_name} \ \ (\psi_n={psin_cfs_p75:0.3}) \ [{var_units}]$")
        axs[1].legend()
        
        return_figs.append(fig)
        plt.show()
        plt.close()
    
    
    if "SOL" in regions:
        fig, axs = plt.subplots(2,1)
        fig.suptitle(r"SOL")
        
        axs[0].plot(grid.dl2d[ix_sol_p25,:],var[ix_sol_p25,:],marker='o',c=color, markersize=5,markerfacecolor='none', label=f"{grid.grid_name}")
        axs[0].set_xlabel(rf"$s(m)$")
        axs[0].set_ylabel(rf"${plot_name} \ \ (\psi_n={psin_sol_p25:0.3}) \ [{var_units}]$")
        axs[0].legend()
        
        axs[1].plot(grid.dl2d[ix_sol_p75,:],var[ix_sol_p75,:],marker='o',c=color, markersize=5,markerfacecolor='none', label=f"{grid.grid_name}")
        axs[1].set_xlabel(rf"$s(m)$")
        axs[1].set_ylabel(rf"${plot_name} \ \ (\psi_n={psin_sol_p75:0.3}) \ [{var_units}]$")
        axs[1].legend()
        
        return_figs.append(fig)
        plt.show()
        plt.close()
    
    
    if "PFR" in regions:
        fig, axs = plt.subplots(2,1)
        fig.suptitle(r"PFR")
    
        axs[0].plot(grid.dl2d[ix_pfr_p25,grid.j_pfr],var[ix_pfr_p25,grid.j_pfr],marker='o',c=color, markersize=5,markerfacecolor='none', label=f"{grid.grid_name}")
        axs[0].set_xlabel(r"$s(m)$")
        axs[0].set_ylabel(rf"${plot_name} \ \ (\psi_n={psin_pfr_p25:0.3}) \ [{var_units}]$")
        axs[0].legend()
    
        axs[1].plot(grid.dl2d[ix_pfr_p75,grid.j_pfr],var[ix_pfr_p75,grid.j_pfr],marker='o',c=color, markersize=5,markerfacecolor='none', label=f"{grid.grid_name}")
        axs[1].set_xlabel(r"$s(m)$")
        axs[1].set_ylabel(rf"${plot_name} \ \ (\psi_n={psin_pfr_p75:0.3}) \ [{var_units}]$")
        axs[1].legend()
        
        return_figs.append(fig)
        plt.show()
        plt.close()
    
    # return return_figs

