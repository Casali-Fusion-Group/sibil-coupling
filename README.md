# SOLPS-ITER and BOUT++ Iterative Linkage (SIBIL)

SIBIL is a package that couples SOLPS-ITER and the BOUT++ elm-6f model. The package primarily includes python wrapper/I:O scripts to read SOLPS-ITER simulations, create BOUT++ simulations, and check for convergence.

The package includes reader files for EQDSKs, b2fstate, b2fplasmf, and b2fgmtry.

---

## Installation

Install with pip from the current directory:

```bash
pip install -e .
