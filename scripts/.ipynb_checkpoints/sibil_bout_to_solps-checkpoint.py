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
from scipy.interpolate import interp1d
from boutdata.data import BoutData
from boututils.datafile import DataFile
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter
from boutdata.data import BoutData
import matplotlib.pyplot as plt
from ipywidgets import interact, IntSlider, fixed
from sibil_io_functions import load_config, sibil_config_summary, confirm, parse_tdata2, parse_block_sizes, write_updated_namelist, _format_value, extract_tdata_first_index, interpolate_dataset, get_midplane_distances_from_psin, vec_summary, set_tdata2_block, get_bout_midplane_distances
from converged_profile_functions import unbiased_convergence, end_biased_convergence

DEFAULT_CONFIG_PATH = "SIBIL_config.yaml"

def main():
    config_path = DEFAULT_CONFIG_PATH
    
    if not os.path.exists(config_path):
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)
    
    config = load_config(config_path)
    sibil_config_summary(config)

    next_iteration_folder = f"{config["base_path"]}/{int(config["iteration"])+1}_iteration"

    if not os.path.exists(next_iteration_folder):
        print(f"Must set up folder for iteration {int(config["iteration"])+1} first!")
        raise FileNotFoundError(f"Next iteration folder doesn't yet exist: {next_iteration_folder}")
    
    if not confirm():
        print("Aborted. No changes made.")
        sys.exit(0)
    
    import sibil  # imported here so summary/confirmation works without dependency installed
    
    INPUT_FILE  = f"{config["base_path"]}/{config["iteration"]}_iteration/b2.transport.inputfile"
    OUTPUT_FILE = f"{config["base_path"]}/{int(config["iteration"])+1}_iteration/b2.transport.inputfile"
    bout_dir  = Path(f"{config["base_path"]}/{config["iteration"]}_iteration/{config["files"]["bout_dir"]}")
    data_dir  = bout_dir / "numpy"
    grid_fname= Path(f"{config["base_path"]}/{config["iteration"]}_iteration/{config["files"]["bgrid"]}")
    grid_fname = str(grid_fname)
    transport_coeffs = config["bout_to_solps_pass"]["passed_coeffs"]
    profile_sample_timestep = config["bout_to_solps_pass"]["profile_sample_timestep"]
    jomp = config["bout_to_solps_pass"]["jomp"]
    N, l = config["bout_to_solps_pass"]["N"], config["bout_to_solps_pass"]["l"]
    #d = BoutData(path=bout_dir)
    
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
        
    psin = ((grid["psixy"] - grid["psi_axis"]) / (psi_bndry - grid["psi_axis"]))[:,jomp]
    
    sepx_dists_solps = extract_tdata_first_index(INPUT_FILE)
    psin_mid, sepx_dists_bout = get_bout_midplane_distances(grid_fname)
    
    convergent_transport_coeffs = {}
    interplted_transport_coeffs = {}
    
    for var_name in transport_coeffs:
        var = np.load(f"{data_dir}/{var_name}.npy")
        var_conv = unbiased_convergence(var, N, l)
        var_conv_midplane = var_conv[profile_sample_timestep,:,jomp]
        var_conv_interpolated = interpolate_dataset(sepx_dists_bout,var_conv_midplane,sepx_dists_solps)
        nan_mask = np.isnan(var_conv_interpolated)
        var_conv_interpolated[nan_mask]=0
        
        convergent_transport_coeffs[var_name] = var_conv
        interplted_transport_coeffs[var_name] = var_conv_interpolated

    # -----------------------------------------------------------------------------
    # 2) AUTO-DETECT SHAPES
    # -----------------------------------------------------------------------------
    # Poloidal size = last axis of any array
    n_pol = convergent_transport_coeffs["Da_t"].shape[-1]
    
    # Does anything (including psin) carry a time axis?
    has_time = any(a.ndim == 3 for a in list(convergent_transport_coeffs.values()) + [psin])
    n_time = None
    if has_time:
        time_arrays = [a for a in list(convergent_transport_coeffs.values()) + [psin] if a.ndim == 3]
        n_time = time_arrays[0].shape[0]
        for a in time_arrays:
            assert a.shape[0] == n_time, "Mismatched n_time across arrays"
    
    def get_slice(arr, j, t):
        """Return the 1D radial profile at poloidal index j (and time t if present)."""
        if arr.ndim == 3:
            return arr[t, :, j]
        else:
            return arr[:, j]
    
    print(f"Timesteps: {n_time}")
    
    # -----------------------------------------------------------------------------
    # 3) PLOT FUNCTION -- creates a brand-new figure each call (inline-friendly)
    # -----------------------------------------------------------------------------
    colors = {"Da_t": "C0", "Xe_t": "C1", "Xi_t": "C2"}
    labels = {"Da_t": r"$D$", "Xe_t": r"$\chi_e$", "Xi_t": r"$\chi_i$"}
    
    j = jomp
    x = sepx_dists_bout
    x_interp = sepx_dists_solps
    
    fig, axs = plt.subplots(3, 1, figsize=(4, 5), sharex=True)

    for ax, name in zip(axs, convergent_transport_coeffs.keys()):
        
        print(convergent_transport_coeffs[name].shape)
        y = get_slice(convergent_transport_coeffs[name], j, profile_sample_timestep)
        y_interp = interplted_transport_coeffs[name]
            
        ax.plot(x, y, color=colors[name], label=labels[name])
        ax.scatter(x_interp, y_interp, color=colors[name], label=labels[name])
        
        ax.set_ylabel(labels[name])
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.yaxis.get_offset_text().set_fontsize(8)

    axs[2].set_xlabel(r"$\psi_n$")
    title = f"Poloidal index = {j}"
    if has_time:
        title += f"   |   Time index = {profile_sample_timestep}"

    outpath = f"{config["base_path"]}/{config["iteration"]}_iteration/{config["plotting"]["pass_from_bout"]}"
    fig.savefig(outpath, dpi=100)
    print(f"Saved plot to {outpath}")
    plt.close()

    # ------------------------------------------------------------------
    # Step 2: supply new values for each block as a 1-D array
    #
    # Replace the lists below with your actual arrays.
    # Each list must have exactly N entries where N = ndata(1,1,k,l).
    # ------------------------------------------------------------------
    
    percentage = config["bout_to_solps_pass"]["percentage"] # percentage of the cross field transport that has a turbulent contribution
    
    ls = config["bout_to_solps_pass"]["ls"]

    tdata2 = parse_tdata2(INPUT_FILE)
    block_sizes = parse_block_sizes(INPUT_FILE)

    alpha=config["bout_to_solps_pass"]["alpha"]

    for l in ls:
        if "Da_t" in transport_coeffs:
            # set particle density-driven diffusivity
            set_tdata2_block(tdata2,block_sizes,alpha,k=1,l=l, turbulent_profile=interplted_transport_coeffs["Da_t"],percentage=percentage)
        
        if "Xi_t" in transport_coeffs:
            # set ion thermal anomalous diffusivity
            set_tdata2_block(tdata2,block_sizes,alpha,k=3,l=l,turbulent_profile=interplted_transport_coeffs["Xi_t"],percentage=percentage)
    
    if "Xi_t" in transport_coeffs:
        # set electron thermal anomalous diffusivity
        set_tdata2_block(tdata2,block_sizes,alpha,k=4,l=1,turbulent_profile=interplted_transport_coeffs["Xe_t"],percentage=percentage)
    
    # ------------------------------------------------------------------
    # Step 3: write
    # ------------------------------------------------------------------
    write_updated_namelist(INPUT_FILE, OUTPUT_FILE, tdata2)

    
if __name__ == "__main__":
    main()
