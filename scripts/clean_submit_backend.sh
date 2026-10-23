#!/bin/bash
#SBATCH -A m4491
#SBATCH -C cpu
#SBATCH -q debug
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -o bclean_%j.out

set -euo pipefail

module load python

python3 "$BOUT_CLEAN_PY" . "$@"
