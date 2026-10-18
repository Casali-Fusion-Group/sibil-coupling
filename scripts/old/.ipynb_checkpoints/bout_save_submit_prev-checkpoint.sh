#!/bin/bash
# ---------------------------------------------------------------------------
# Login-node launcher for the `bout_save` python script.
#
# This is a thin wrapper: it validates arguments, resolves paths, and
# submits `save_submit_backend.sh` (which carries the #SBATCH directives
# and the actual job logic) via sbatch. It does NOT run on a compute node
# itself.
#
# Install: put this file, `save_submit_backend.sh`, and `bout_save` in your
#          scripts directory, chmod +x this file, and make sure that
#          directory is on your PATH.
#          Optionally symlink without the extension:
#              ln -s save_submit.sh save_submit
#
# Usage:   save_submit.sh /path/to/data [extra args for bout_save]
#          save_submit.sh .
#
# NOTE: sbatch copies the batch script into SLURM's spool directory, so
# $0 / BASH_SOURCE inside the job does NOT point back at your scripts
# directory. All paths are therefore resolved here, on the login node, and
# passed to the job through --export.
# ---------------------------------------------------------------------------

set -euo pipefail

PAYLOAD_NAME="bout_save"
BACKEND_NAME="save_submit_backend.sh"

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
BACKEND_SCRIPT="$SCRIPT_DIR/$BACKEND_NAME"

if [ ! -f "$PY_SCRIPT" ]; then
    echo "Error: $PAYLOAD_NAME not found in $SCRIPT_DIR" >&2
    exit 1
fi

if [ ! -f "$BACKEND_SCRIPT" ]; then
    echo "Error: $BACKEND_NAME not found in $SCRIPT_DIR" >&2
    exit 1
fi

# ---------------------------------------------------------------------
# Build a short, informative job name from the data directory, e.g.
#   this_is_my_directory_testing_ABC  ->  bsave_testing_ABC
#
# Take the last MAX_SUFFIX_LEN characters of the basename; if that cut
# lands mid-word, trim forward to the next '_' so the name isn't a
# mangled word fragment. Then strip any characters SLURM job names
# shouldn't contain.
# ---------------------------------------------------------------------
MAX_SUFFIX_LEN=20
BASE_NAME="$(basename "$DATA_DIR")"
if [ "${#BASE_NAME}" -gt "$MAX_SUFFIX_LEN" ]; then
    SUFFIX="${BASE_NAME: -${MAX_SUFFIX_LEN}}"
    case "$SUFFIX" in
        *_*) SUFFIX="${SUFFIX#*_}" ;;
    esac
else
    SUFFIX="$BASE_NAME"
fi
# Sanitize: keep alnum, dot, dash, underscore only.
SUFFIX="$(echo "$SUFFIX" | tr -c 'A-Za-z0-9._-' '_')"
JOB_NAME="bsave_${SUFFIX}"

# Only BOUT_SAVE_PY goes through --export: a single value with no commas
# and no spaces (assuming a sane path).  Everything else rides in "$@".
#
# --chdir makes the JOB start in DATA_DIR, so output_%j.log and
# error_%j.log land alongside the data instead of in $PWD.
# --job-name overrides the backend's default #SBATCH -J so the queue
# shows which data directory each job belongs to.
sbatch --chdir="$DATA_DIR" \
       --job-name="$JOB_NAME" \
       --export=ALL,BOUT_SAVE_PY="$PY_SCRIPT" \
       "$BACKEND_SCRIPT" "$@"

echo "Submitted: $PY_SCRIPT"
echo "Workdir:   $DATA_DIR"
echo "Job name:  $JOB_NAME"
