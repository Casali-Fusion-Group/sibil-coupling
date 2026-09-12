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
from sibil_io_functions import load_config, print_summary, parse_tdata2, parse_block_sizes, write_updated_namelist, _format_value, extract_tdata_first_index
from converged_profile_functions.py import converged_variable

DEFAULT_CONFIG_PATH = "SIBIL_config.yaml"

def main():
    config_path = DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        print(f"Error: config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    print_summary(config)

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

    INPUT_FILE  = f"{config[base_path]}/{config[iteration]}_iteration/b2.transport.inputfile"
    OUTPUT_FILE = f"{config[base_path]}/{int(config[iteration]+1)}_iteration/b2.transport.inputfile"
    bout_dir  = Path(f"{config[base_path]}/{config[iteration]}_iteration/Shelf_new_grid2_sinks2_gradfix")
    data_dir  = bout_dir / "numpy"
    grid_fname= bout_dir / "d3d_shelf_184016.03200_nc3.grd.nc"
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

    vars_to_plot_list = ["Da_t"]
    vars_to_plot_averaged = {}
    N = 100
    l = 130
    for var_name in vars_to_plot_list:
        var = np.load(data_dir / f"{var_name}.npy")
        if len(var.shape) == 4:
            var = np.sum(var, axis=3) / var.shape[3]
        var = np.transpose(var, (1, 2, 0))
        (ix,jy,t_all) = var.shape
        var_tilde = np.zeros((ix,jy,l))
        for t in range(l):
            if t == 0:
                var_tilde[:,:,0] = var[:,:,0]
            elif l < N:
                if t == 0:
                    var_tilde[:,:,t] = var[:,:,0]
                else:
                    if var_name == "Da_t":
                        var_tilde[:,:,t] = np.maximum(0,(1/N) * np.sum(var[:,:,:t],axis=2)+((N-l)/N)*var[:,:,(t-1)])
                    else:
                        var_tilde[:,:,t] = (1/N) * np.sum(var[:,:,:t],axis=2)+((N-l)/N)*var[:,:,(t-1)]
            elif l >= N:
                if var_name == "Da_t":
                    var_tilde[:,:,t] = np.maximum(0,(1/l) * np.sum(var[:,:,:t],axis=2))
                else:
                    var_tilde[:,:,t] = (1/l) * np.sum(var[:,:,:t],axis=2)
                
        vars_to_plot_averaged[var_name] = var_tilde
        
        
    # ------------------------------------------------------------------
    # Step 1: parse
    # ------------------------------------------------------------------
    tdata2 = parse_tdata2(INPUT_FILE)

    block_sizes = parse_block_sizes(INPUT_FILE)
    print("\nBlock sizes (ndata(1,k,l)):")
    for (k, l), n in sorted(block_sizes.items()):
        print(f"  (k={k}, l={l})  ->  N={n}")

if __name__ == "__main__":
    main()
