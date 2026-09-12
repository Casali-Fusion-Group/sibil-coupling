#!/usr/bin/env python3
"""
sibil_map_to_bout.py

Loads SIBIL_config.yaml, prints a summary of the resolved configuration,
asks for confirmation, and (if confirmed) updates b2.transport.inputfile.
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
import re
import numpy as np
import netCDF4 as nc
import yaml
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from boutdata.data import BoutData
from boututils.datafile import DataFile
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter
from boutdata.data import BoutData
sys.path.append("../../helper_scripts")
from sibil_io_functions import load_config, sibil_config_summary, confirm, parse_tdata2, parse_block_sizes, write_updated_namelist, _format_value, extract_tdata_first_index, interpolate_dataset, get_midplane_distances_from_psin, vec_summary, set_tdata2_block, get_all_midplane_distances
from converged_profile_functions import unbiased_convergence

DEFAULT_CONFIG_PATH = "SIBIL_config.yaml"

def main():
    config_path = DEFAULT_CONFIG_PATH
    
    if not os.path.exists(config_path):
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    sibil_config_summary(config)
    
    if not confirm():
        print("Aborted. No changes made.")
        sys.exit(0)
    
    import sibil  # imported here so summary/confirmation works without dependency installed
    
    # Matches any tdata(i, j, k, l) = value — tolerates arbitrary whitespace.
    _TDATA_RE = re.compile(
        r"(tdata\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*)"
        r"([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )
    
    # Matches ndata(1, k, l) = N  — used to discover block sizes and identifiers.
    _NDATA_RE = re.compile(
        r"ndata\(\s*1\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(\d+)",
        re.IGNORECASE,
    )
    
    INPUT_FILE  = f"{config["base_path"]}/{config["iteration"]}_iteration/b2.transport.inputfile"
    OUTPUT_FILE = f"{config["base_path"]}/{int(config["iteration"])+1}_iteration/b2.transport.inputfile"
    bout_dir  = Path(f"{config["base_path"]}/{config["iteration"]}_iteration/Shelf_new_grid2_sinks2_gradfix")
    data_dir  = bout_dir / "numpy"
    grid_fname= Path(f"{config["base_path"]/config["bgrid"]}")
    grid_fname = str(grid_fname)
    d = BoutData(path=bout_dir)
    
    grid = DataFile(grid_fname)
    try:
        _ = grid["psi_bdry"]
        has_psi_bdry = True
    except Exception:
        has_psi_bdry = False
        
    if has_psi_bdry:
        psi_bndry = grid["psi_bdry"]
    else:
        psi_bndry = grid["psi_bndry"]
    
    jomp = 44
    psi_midplane = grid["psixy"][:,jomp]
    psin_midplane = ((grid["psixy"] - grid["psi_axis"]) / (psi_bndry - grid["psi_axis"]))[:,jomp]
    
    
    vars_to_plot_list = ["Da_t","Xe_t","Xi_t"]
    vars_to_plot_averaged = {}
    N = 100
    l = 130
    for var_name in vars_to_plot_list:
        var = np.load(data_dir / f"{var_name}.npy")
        vars_to_plot_averaged[var_name] = converged_variable(var, N, l)
    
    values = extract_tdata_first_index(INPUT_FILE)
    psin_mid, distances = get_all_midplane_distances(grid_fname)
    
    outputs = interpolate_dataset(distances,vars_to_plot_averaged['Da_t'][:,jomp,-1],values)
    nan_mask = np.isnan(outputs)
    outputs[nan_mask]=0
    
    # PLOTTING
    rows = len(vars_to_plot_averaged)
    fig, axs = plt.subplots(rows,1, figsize=(8,6))
    for i, var in enumerate(vars_to_plot_list):
        profile = interpolate_dataset(distances,vars_to_plot_averaged[f'{var}'][:,jomp,-1],values)
        nan_mask = np.isnan(profile)
        profile[nan_mask]=0
        transport_coeff = var.split("_")[0]
        ylabel = f"{transport_coeff[:-1]}_{transport_coeff[-1]}"
        axs[i].scatter(values,profile/np.linalg.norm(profile),label=f"{var}")
        axs[i].set_ylabel(f"${ylabel}$")
        axs[i].legend(fontsize="xx-small", handlelength=1.5)
        if i == rows-1:
            axs[i].set_xlabel(r"${\psi}_n$")
    
    fig.tight_layout()
    
    outpath = f"{config["base_path"]}/{config["iteration"]}_iteration/{config["to_bout_plot"]}"
    fig.savefig(outpath, dpi=100)
    print(f"Saved plot to {outpath}")
    plt.close()

    # # ------------------------------------------------------------------
    # # Step 2: supply new values for each block as a 1-D array
    # #
    # # Replace the lists below with your actual arrays.
    # # Each list must have exactly N entries where N = ndata(1,1,k,l).
    # # ------------------------------------------------------------------
        
    # percentage = 0.005 # percentage of the cross field transport that has a turbulent contribution
    
    # # set particle density-driven diffusivity
    # set_tdata2_block(tdata2,k=1,l=1,turbulent_profile=outputs,percentage=percentage)
    # set_tdata2_block(tdata2,k=1,l=3,turbulent_profile=outputs,percentage=percentage)
    # set_tdata2_block(tdata2,k=1,l=4,turbulent_profile=outputs,percentage=percentage)
    # set_tdata2_block(tdata2,k=1,l=5,turbulent_profile=outputs,percentage=percentage)
    # set_tdata2_block(tdata2,k=1,l=6,turbulent_profile=outputs,percentage=percentage)
    # set_tdata2_block(tdata2,k=1,l=7,turbulent_profile=outputs,percentage=percentage)
    # set_tdata2_block(tdata2,k=1,l=8,turbulent_profile=outputs,percentage=percentage)
    
    # # set ion thermal anomalous diffusivity
    # set_tdata2_block(tdata2,k=3,l=1,turbulent_profile=outputs,percentage=percentage)
    # set_tdata2_block(tdata2,k=3,l=3,turbulent_profile=outputs,percentage=percentage)
    # set_tdata2_block(tdata2,k=3,l=4,turbulent_profile=outputs,percentage=percentage)
    # set_tdata2_block(tdata2,k=3,l=5,turbulent_profile=outputs,percentage=percentage)
    # set_tdata2_block(tdata2,k=3,l=6,turbulent_profile=outputs,percentage=percentage)
    # set_tdata2_block(tdata2,k=3,l=7,turbulent_profile=outputs,percentage=percentage)
    # set_tdata2_block(tdata2,k=3,l=8,turbulent_profile=outputs,percentage=percentage)
    
    # # set electron thermal anomalous diffusivity
    # set_tdata2_block(tdata2,k=4,l=1,turbulent_profile=outputs,percentage=percentage)
    
    # # ------------------------------------------------------------------
    # # Step 3: write
    # # ------------------------------------------------------------------
    # write_updated_namelist(INPUT_FILE, OUTPUT_FILE, tdata2)

if __name__ == "__main__":
    main()
