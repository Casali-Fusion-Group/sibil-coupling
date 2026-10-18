#!/bin/bash
#SBATCH -N 1
#SBATCH -A m4491
#SBATCH -C cpu
#SBATCH -J bout_save_data
#SBATCH -q debug
#SBATCH -t 00:30:00
#SBATCH -o output_%j.log
#SBATCH -e output_%j.log

# ---------------------------------------------------------------------------
# Self-submitting launcher for the `bout_save` python script.
#
# Install: put this file (and `bout_save`) in your scripts directory,
#          chmod +x it, and make sure that directory is on your PATH.
#          Optionally symlink without the extension:
#              ln -s bout_save_submit.sh bout_save_submit
#
# Usage:   bout_save_submit /path/to/data [extra args for bout_save]
#          bout_save_submit .
#
# Run directly, it validates arguments and sbatch's ITSELF. SLURM then
# re-executes this same file on a compute node, where SLURM_JOB_ID is set
# and the payload at the bottom runs.
#
# NOTE: sbatch copies the batch script into SLURM's spool directory, so
# $0 / BASH_SOURCE inside the job does NOT point back at your scripts
# directory. All paths are therefore resolved during the submit phase and
# passed to the job through --export.
#
# NOTE: `bout_save` has no .py extension. That is fine for execution --
# `python /path/to/bout_save` runs it regardless of extension -- but it
# does mean the file cannot be `import`ed as a module. It is always
# invoked through the interpreter here rather than relying on a shebang,
# so the `module load python` interpreter is the one that gets used.
# ---------------------------------------------------------------------------

set -euo pipefail

PAYLOAD_NAME="bout_save"

if [ -z "${SLURM_JOB_ID:-}" ]; then
    # =====================================================================
    # SUBMIT PHASE  (login node)
    # =====================================================================

    # Default to the current directory.  $1 is consumed as DATA_DIR only if
    # it is a non-flag that names an existing directory; otherwise it is
    # left in "$@" and forwarded to bout_save.
    #
    #   [ "$#" -gt 0 ]           -> bare `shift` returns nonzero and would
    #                               kill the script under `set -e`.
    #   [ "${1#-}" = "$1" ]      -> true only when $1 does NOT start with '-'.
    #   [ -d "$1" ]              -> the actual "is this a path?" test.
    DATA_DIR="."
    if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ] && [ -d "$1" ]; then
        DATA_DIR="$1"
        shift
    fi

    # A leading non-flag that survived the block above is a typo'd path (or
    # a stray positional).  Fail loudly rather than silently using $PWD.
    if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
        echo "Error: '$1' is not a directory and not a flag." >&2
        echo "Usage: $(basename "$0") [DATA_DIR] [bout_save args...]" >&2
        exit 1
    fi

    # Absolute path, so "." works from anywhere.
    DATA_DIR="$(cd "$DATA_DIR" && pwd)"

    # Resolve this script's real location (follows symlinks, PATH-safe)
    SELF="$(readlink -f "${BASH_SOURCE[0]}")"
    SCRIPT_DIR="$(dirname "$SELF")"
    PY_SCRIPT="$SCRIPT_DIR/$PAYLOAD_NAME"

    if [ ! -f "$PY_SCRIPT" ]; then
        echo "Error: $PAYLOAD_NAME not found in $SCRIPT_DIR" >&2
        exit 1
    fi

    # Only BOUT_SAVE_PY goes through --export: a single value with no commas
    # and no spaces (assuming a sane path).  Everything else rides in "$@".
    #
    # --chdir makes the JOB start in DATA_DIR, so output_%j.log and
    # error_%j.log land alongside the data instead of in $PWD.
    sbatch --chdir="$DATA_DIR" \
           --export=ALL,BOUT_SAVE_PY="$PY_SCRIPT" \
           "$SELF" "$@"

    echo "Submitted: $PY_SCRIPT"
    echo "Workdir:   $DATA_DIR"
    exit 0
fi

# =====================================================================
# JOB PHASE  (compute node, cwd = DATA_DIR)
# =====================================================================
if [ -z "${BOUT_SAVE_PY:-}" ]; then
    echo "Error: BOUT_SAVE_PY is not set." >&2
    echo "This script must be submitted via itself, not with a bare 'sbatch'." >&2
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

export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/global/cfs/projectdirs/m4491/ABOUND/local_libs/usr/netcdf-4.1.3/lib

# Let `bout_save` import helper modules that live next to it, without
# needing an installed package or a .py extension on the entry point.
# export PYTHONPATH="$(dirname "$BOUT_SAVE_PY")${PYTHONPATH:+:$PYTHONPATH}"

echo "Host:     $(hostname)"
echo "Workdir:  $(pwd)"
echo "Script:   $BOUT_SAVE_PY"
echo "Python:   $(command -v python)"
echo "Env:      ${CONDA_DEFAULT_ENV:-<none>}"
echo "Args:     $*"
echo "Started:  $(date)"
echo

# Sanity check: the job should be sitting in a BOUT++ data directory.
if [ ! -f BOUT.inp ]; then
    echo "Warning: no BOUT.inp in $(pwd) -- is this the right data directory?" >&2
fi

# cwd IS the data directory (--chdir), so "." is the path argument.
# "$@" carries the user's extra flags verbatim; the trailing "--" then "."
# guarantees the positional is never swallowed by a greedy nargs list.
srun -n 1 python "$BOUT_SAVE_PY" "$@" -- .
RC=$?

echo
echo "Exit code: $RC"
echo "Finished:  $(date)"
exit $RC
