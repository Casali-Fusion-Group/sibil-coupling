# SOLPS-ITER and BOUT++ Iterative Linkage (SIBIL)

SIBIL is an iterative, self-consistent coupling between SOLPS-ITER and BOUT++.. The package  includes python wrapper/I:O scripts to read SOLPS-ITER simulations, create BOUT++ simulations, and check for convergence.

The package includes reader files for EQDSKs, b2fstate, b2fplasmf, and b2fgmtry.

This new capability enables turbulence-informed cross field transport at the plasma edge including effects arising from transients and the interplay between the pedestal structure, edge turbulence, and divertor-plasma solutions.<img width="468" height="61" alt="image" src="https://github.com/user-attachments/assets/7071dda7-2474-4825-84a8-deaadff60f76" />


This code was developed under Prof. Casali's U.S. Department of Energy, Office of Science and Office of Advanced Scientific Computing Research through the Advanced Computing (SciDAC) program under Award Number R011382908


---

## Installation

Install with pip from the current directory:

```bash
pip install -e .
```

---

## Scripts Style Guide

We highly encourage new users to add to the user functionality included in this package. However, we would greatly appreciate if any and all comments to follow PEP 8 python standards such that any scripts be named according to the following rules.

```bash
b_script_name.py

[bout]_[short_description_of_script_function].[python_script]
```

For scripts that require user input, please use python parser and include -h flag to print help on what the parser function serves.

For scripts that do not require user input, the package creators would greatly appreciate the users using python. If the scripts are already written using different standards, that is totally ok! However, please understand if the standards are modified according to python standards. 
