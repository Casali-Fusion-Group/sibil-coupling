#!/bin/bash
# ---------------------------------------------------------------------------
# Login-node launcher for the `bout_save` python script.
#
# Submits `save_submit_backend.sh` as a chain of up to SAVE_TRIES attempts:
#
#   attempt 1 : runs after $SAVE_AFTER (e.g. afterany:<bout_jobid>), or now
#   attempt 2 : afternotok:<a1>          -> only if attempt 1 failed/timed out
#   attempt 3 : afternotok:<a1>:<a2>     -> only if attempts 1 AND 2 failed
#
# Once an attempt succeeds, every later attempt's dependency can never be
# satisfied and --kill-on-invalid-dep removes it from the queue.  Attempt N
# depends on ALL earlier attempts (not just N-1) so that a cancelled
# never-satisfied job is not mistaken for a failure.
#
# Environment overrides:
#   SAVE_AFTER   dependency for attempt 1      (default: none)
#   SAVE_QOS     QOS for every attempt         (default: premium)
#   SAVE_TIME    walltime for every attempt    (default: 00:30:00)
#   SAVE_TRIES   number of attempts            (default: 3)
#
# Usage:   bout_save_submit.sh [DATA_DIR] [bout_save args...]
# ---------------------------------------------------------------------------

set -euo pipefail

PAYLOAD_NAME="bout_save"
BACKEND_NAME="save_submit_backend.sh"

SAVE_AFTER="${SAVE_AFTER:-}"
SAVE_QOS="${SAVE_QOS:-debug}"
SAVE_TIME="${SAVE_TIME:-00:30:00}"
SAVE_TRIES="${SAVE_TRIES:-1}"

DATA_DIR="."
if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ] && [ -d "$1" ]; then
    DATA_DIR="$1"
    shift
fi

if [ "$#" -gt 0 ] && [ "${1#-}" = "$1" ]; then
    echo "Error: '$1' is not a directory and not a flag." >&2
    echo "Usage: $(basename "$0") [DATA_DIR] [bout_save args...]" >&2
    exit 1
fi

DATA_DIR="$(cd "$DATA_DIR" && pwd)"

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

# Job name: bsave_<tail of data dir basename>
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
SUFFIX="$(printf '%s' "$SUFFIX" | tr -c 'A-Za-z0-9._-' '_')"
JOB_NAME="bsave_${SUFFIX}"

echo "Workdir:   $DATA_DIR"
echo "QOS/time:  $SAVE_QOS / $SAVE_TIME   (up to $SAVE_TRIES attempts)"

# Command-line --qos/--time override any #SBATCH -q/-t in the backend.
PREV=""
for i in $(seq 1 "$SAVE_TRIES"); do
    if [ "$i" -eq 1 ]; then
        DEP="$SAVE_AFTER"
    else
        DEP="afternotok${PREV}"
    fi

    JID=$(sbatch --parsable \
                 --chdir="$DATA_DIR" \
                 --job-name="${JOB_NAME}_${i}" \
                 --qos="$SAVE_QOS" \
                 --time="$SAVE_TIME" \
                 ${DEP:+--dependency="$DEP" --kill-on-invalid-dep=yes} \
                 --export=ALL,BOUT_SAVE_PY="$PY_SCRIPT" \
                 "$BACKEND_SCRIPT" "$@")
    JID="${JID%%;*}"
    PREV="${PREV}:${JID}"

    echo "Attempt $i: job $JID  ${DEP:+[$DEP]}"
done
