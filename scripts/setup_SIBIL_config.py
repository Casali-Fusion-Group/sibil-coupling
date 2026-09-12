#!/usr/bin/env python3
"""
setup_sibil_config.py

Creates a default SIBIL_config.yaml populated with placeholder values.
Edit the generated file (base_path, file names, softmax_scales) before
running run_sibil.py.

Usage:
    python setup_sibil_config.py [output_path]

If output_path is omitted, "SIBIL_config.yaml" is created in the
current directory. Existing files are not overwritten unless --force
is passed.
"""

import os
import sys

DEFAULT_OUTPUT = "SIBIL_config.yaml"

TEMPLATE = """\
# SIBIL_config.yaml
# Configuration for SIBIL_Midplane_Mapper
# Paths that start with "../" are resolved relative to base_path.
#
# >>> EDIT ALL VALUES BELOW BEFORE RUNNING run_sibil.py <<<

base_path: "/path/to/SOLPS/directory"
iteration: "0"
mapping:   "midplane"

plotting:
    pass_from_solps: "midplane_profiles.png"
    pass_from_bout: "transport_midplane_profiles.png"
    
files:
    gfile:     "../g_file_name"
    geofile:   "../geometry_file_name.geo"
    bout_dir:  "grid_subdirectory"
    bgrid:     "grid_subdirectory/bgrid_file_name.grd.nc"
    b2fstate:  "b2fstate"
    b2fplasmf: "b2fplasmf"
    b2fgmtry:  "b2fgmtry"

bout_to_solps_pass:
    passed_coeffs: ["Da_t","Xe_t","Xi_t"]
    N: 100
    l: 200
    jomp: 44
    ls: [1, 3, 4, 5, 6, 7, 8]       # Ionization states of all elements found in run (eg is D and C)
    percentage: 0.050               # percentage of the cross field transport that has a turbulent contribution
    profile_sample_timestep: 200    # Where in the convergence is the profile take from (-1 is last timestep)
                                    # NOTE: profile_sample_timestep >= N and often profile_sample_timestep = l

# softmax_scales: {variable: [max_scale, min_scale]}
softmax_scales:
    ni: [1.0e+16, 1.0e+15]
    ne: [1.0e+16, 1.0e+15]
    Ti: [1.0e+00, 1.0e-04]
    Te: [1.0e+00, 1.0e-04]
"""


def main():
    args = sys.argv[1:]
    force = "--force" in args

    output_path = DEFAULT_OUTPUT

    if os.path.exists(output_path) and not force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        sys.exit(1)

    with open(output_path, "w") as f:
        f.write(TEMPLATE)

    print(f"Created default config: {output_path}")
    print("Edit base_path, file names, and softmax_scales before running run_sibil.py.")


if __name__ == "__main__":
    main()
