#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 18:18:03 2025

@author: ttaczak
"""
import matplotlib.pyplot as plt
from Readers.BOUT_Reader import BOUT_data
import numpy as np
import matplotlib.colors as mcolors
import os
from pathlib import Path
import warnings
import time
from PlotPDF import to_pdf
def bout_plot(bgrid_file_loc,val, 
              title="title", 
              x_axis = r'$R(m)$', 
              y_axis = r'$Z(m)$',
              dpi=800,
              scale="linear", 
              linthresh=1e18,
              show_plot=False,
              output_type="PDF",
              suppress_warnings=False):
    '''
    Recommended viewing method is to download the plots into your scratch directory,
    then scp the plots to your personal computer for viewing using 
    
    scp -i ~/.ssh/nersc -r $USER@perlmutter.nersc.gov:$SCRATCH/plots desired/plotting/destination

    Parameters
    ----------
    bgrid_file_loc : TYPE
        DESCRIPTION.
    val : TYPE
        DESCRIPTION.
    title : TYPE, optional
        DESCRIPTION. The default is "title".
    x_axis : TYPE, optional
        DESCRIPTION. The default is r'$R(m)$'.
    y_axis : TYPE, optional
        DESCRIPTION. The default is r'$Z(m)$'.
    dpi : TYPE, optional
        DESCRIPTION. The default is 300.
    scale : TYPE, optional
        Type of scale for the graph. Currently accepted options are
        "log" or "symlog." The default is "linear".

    Returns
    -------
    None.

    '''
    scratch = os.environ.get("$SCRATCH")
    plotting_dir = Path(f"{scratch}/plots")
    if not plotting_dir:
        if not suppress_warnings: warnings.warn("Could not find {plotting_dir}! Using plt.show() instead -> can be VERY slow on NERSC!")
        
    
    bout=BOUT_data(bgrid_file_loc)
    rxy = bout.Rxy 
    zxy = bout.Zxy
    nx,ny=bout.Rxy.shape # here (x,y)=(radial,poloidal)
    
    fig,ax = plt.subplots(figsize=[7,7])
    
    # Apply log scale if requested
    if scale == "log":
        min_val = np.nanmin(val)
        max_val = np.nanmax(val)
        # Shift everything so the minimum is slightly above zero
        shift = np.abs(min_val) + 10**(np.floor(np.log10(np.abs(max_val))) - 6)  # Dynamic shift
        val_shifted = val + shift

        norm = mcolors.LogNorm(vmin=np.nanmin(val_shifted), vmax=np.nanmax(val_shifted))
        c1 = ax.pcolor(rxy, zxy, val_shifted, norm=norm, cmap='viridis')
    elif scale == "symlog":
        norm = mcolors.SymLogNorm(linthresh=linthresh, linscale=1, vmin=np.nanmin(val), vmax=np.nanmax(val))
        c1 = ax.pcolor(rxy, zxy, val, norm=norm, cmap='viridis')
    else:
        c1 = ax.pcolor(rxy, zxy, val, cmap='viridis')
    
    ax.axis('tight')
    fig.colorbar(c1, ax=ax)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(f"{x_axis}")
    ax.set_ylabel(f"{y_axis}")
    ax.set_title(title)
    
    if show_plot:
        plt.show()
        
    if output_type == "PNG":
        if not plotting_dir:
            if not suppress_warnings: warnings.warn(f"Could not find plotting_dir:{plotting_dir} and output_type = {output_type}! Plotting using plt.show() -> can be VERY slow on NERSC!")
        else:
            plot_name = 'bout_plot_{:.4f}.png'.format(time.time())
            plot_loc = plotting_dir + plot_name
            plt.savefig(plot_loc, dpi=dpi, bbox_inches='tight')
            plt.close()
            print("Successfully plotted bout_plot_{:.4f}.png at {plotting_dir}")
    elif output_type == "PDF":
        return fig
    else:
        raise Exception(f"{output_type} not recognized! Not saving the requested figured.")

