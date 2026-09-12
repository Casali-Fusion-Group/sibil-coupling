#!/usr/bin/env python3

import os
import sys
import yaml
import shutil
from pathlib import Path
from sibil_io_functions import load_config

DEFAULT_CONFIG_PATH = "SIBIL_config.yaml"
DEFAULT_JOB_SCRIPT  = "job_submit.sh"
DEFAULT_SAVE_DATA   = "save_data.sh"

def main():
    config_path = DEFAULT_CONFIG_PATH

    if not os.path.exists(config_path):
        print(f"Error: config file not found {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    sibil_iteration = int(config["iteration"])

    if not sibil_iteration > 0:
        print(f"Error: cannot copy previous bout dir for iteration {sibil_iteration-1}")
        sys.exit(1)
        
    current_iteration_folder = Path(f"{config["base_path"]}/{sibil_iteration}_iteration")
    previous_iteration_folder = Path(f"{config["base_path"]}/{sibil_iteration-1}_iteration")

    if not os.path.exists(current_iteration_folder):
        print(f"Error: run folder not found {current_iteration_folder}")
        sys.exit(1)

    if not os.path.exists(previous_iteration_folder):
        print(f"Error: previous run folder not found {previous_iteration_folder}")
        sys.exit(1)

    print(current_iteration_folder)
    bout_dir = current_iteration_folder / config["files"]["bout_dir"]
    bout_dir.mkdir(parents=True, exist_ok=True)

    prev_bout_dir = previous_iteration_folder / config["files"]["bout_dir"]

    print("Finding required files in previous iteration directory")
    # REQUIRED FILES
    bout_inp_fname    = prev_bout_dir / "BOUT.inp"
    bout_grid_fname   = previous_iteration_folder / config["files"]["bgrid"]
    bout_save_fname   = prev_bout_dir / DEFAULT_SAVE_DATA
    bout_submit_fname = prev_bout_dir / DEFAULT_JOB_SCRIPT
    
    shutil.copy(bout_inp_fname, bout_dir)
    shutil.copy(bout_grid_fname, bout_dir)
    shutil.copy(bout_save_fname, bout_dir)
    shutil.copy(bout_submit_fname, bout_dir)
    
    print("New bout direcotry successfully built")
    
if __name__ == "__main__":
    main()
