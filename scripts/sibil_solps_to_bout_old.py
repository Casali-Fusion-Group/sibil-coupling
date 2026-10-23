#!/usr/bin/env python3
"""
sibil_map_to_bout.py

Loads SIBIL_config.yaml, prints a summary of the resolved configuration,
asks for confirmation, and (if confirmed) runs sibil.SIBIL_Midplane_Mapper.

Usage:
    python run_sibil.py [path/to/SIBIL_config.yaml]
"""

import os
import sys
import yaml
from sibil_io_functions import load_config, sibil_config_summary, confirm

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

    if config["mapping"] == "midplane":
        print("\nRunning SIBIL_Midplane_Mapper...")
        mapper = sibil.SIBIL_Midplane_Mapper(config)
        print("Done.")
        return mapper
    else:
        print("Aborted. No recognized mapping procedure.")
        sys.exit(0)


if __name__ == "__main__":
    main()
