#!/bin/bash
#SBATCH -N 1
#SBATCH -A m4491
#SBATCH -C cpu
#SBATCH -J bout_save_data
#SBATCH -q regular
#SBATCH -t 02:00:00
#SBATCH -o output_%j.log
#SBATCH -e output_%j.log

# ---------------------------------------------------------------------------
# Backend job script for the `bout_save` python script.
#
# This file is NOT meant to be run or sbatch'd directly by a user. It is
# submitted by `save_submit.sh`, which resolves all paths on the login node
# and passes them in via --export. SLURM then executes this file on a
# compute node (cwd = DATA_DIR, via --chdir).
#
# Install: this file lives alongside `save_submit.sh` and `bout_save` in
#          your scripts directory. Only `save_submit.sh` needs to be on
#          your PATH / invoked by the user.
#
# NOTE: `bout_save` has no .py extension. That is fine for execution --
# `python /path/to/bout_save` runs it regardless of extension -- but it
# does mean the file cannot be `import`ed as a module. It is always
# invoked through the interpreter here rather than relying on a shebang,
# so the `module load python` interpreter is the one that gets used.
# ---------------------------------------------------------------------------

set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
    echo "Error: this script must be submitted via save_submit.sh, not run or sbatch'd directly." >&2
    exit 1
fi

if [ -z "${BOUT_SAVE_PY:-}" ]; then
    echo "Error: BOUT_SAVE_PY is not set." >&2
    echo "This script must be submitted via save_submit.sh, not with a bare 'sbatch'." >&2
    exit 1
fi

if [ ! -f "$BOUT_SAVE_PY" ]; then
    echo "Error: payload not found: $BOUT_SAVE_PY" >&2
    exit 1
fi

# OpenMP settings (single core)
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=true

# `module` and `conda activate` are not -e/-u clean: they routinely return
# nonzero and reference unbound variables internally.  Relaxing the shell
# options around them stops the job dying silently before srun.
set +euo pipefail
module load conda
conda activate bout-data
CONDA_RC=$?
set -euo pipefail

if [ "$CONDA_RC" -ne 0 ]; then
    echo "Warning: 'conda activate bout-data' returned $CONDA_RC" >&2
fi

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:+$LD_LIBRARY_PATH:}/global/cfs/projectdirs/m4491/ABOUND/local_libs/usr/netcdf-4.1.3/lib

# Let `bout_save` import helper modules that live next to it, without
# needing an installed package or a .py extension on the entry point.
# export PYTHONPATH="$(dirname "$BOUT_SAVE_PY")${PYTHONPATH:+:$PYTHONPATH}"

echo "Workdir:  $(pwd)"
echo "Started:  $(date)"
echo

# Sanity check: the job should be sitting in a BOUT++ data directory.
if [ ! -f BOUT.inp ]; then
    echo "Warning: no BOUT.inp in $(pwd) -- is this the right data directory?" >&2
fi

# cwd IS the data directory (--chdir), so "." is the path argument.
# "$@" carries the user's extra flags verbatim; the trailing "--" then "."
# guarantees the positional is never swallowed by a greedy nargs list.
RC=0
# one process, SLURM_CPUS_PER_TASK cores: bout_save forks that many workers,
# each collecting one variable (--workers defaults to SLURM_CPUS_PER_TASK)
srun -n 1 -c "${SLURM_CPUS_PER_TASK:-1}" --cpu-bind=none \
     python -u "$BOUT_SAVE_PY" "$@" -- . || RC=$?

echo
echo "Exit code: $RC"
echo "Finished:  $(date)"
exit $RC
