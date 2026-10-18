#!/usr/bin/env python3
"""
bout_run -- submit a BOUT++ run directory, then chain `bout_save` to run
automatically when the BOUT job finishes.

1. Validates the run directory (job_submit.sh, BOUT.inp, $BGRIDS).
2. Expands $BGRIDS / ${BGRIDS} in BOUT.inp, keeping the un-substituted
   template in BOUT.inp_orig.
3. Submits job_submit.sh with `sbatch --parsable` and captures the job ID.
4. Calls `bout_save_submit` with SAVE_AFTER=<after>:<jobid>.  The wrapper
   passes that to sbatch as an explicit --dependency and queues the
   retry chain (default: 3 x 30 min on debug).

Default dependency is afterany, since BOUT runs usually end non-zero
(crash / walltime) even when the dumps are usable.

Usage
-----
    bout_run <dir>                        # submit run + chained bout_save
    bout_run <dir> --no-save              # submit run only
    bout_run <dir> --save-args "-c 1"     # extra flags for bout_save
    bout_run <dir> --after ok             # bout_save only if BOUT exits 0
    bout_run <dir> --save-qos regular --save-time 01:00:00 --save-tries 2
    bout_run <dir> --dry-run              # show the plan, submit nothing
    bout_run <dir> --save-only            # skip BOUT, just queue bout_save
"""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

SCRIPT_NAME = "job_submit.sh"
BOUT_INP = "BOUT.inp"
BOUT_INP_ORIG = "BOUT.inp_orig"
SAVE_WRAPPER = "bout_save_submit"


def die(msg: str) -> None:
    print(f"Error: {msg}", file=sys.stderr)
    sys.exit(1)


def print_cmd(cmd: list[str], cwd: Path | None = None) -> None:
    if cwd is not None:
        print(f"  (cwd: {cwd})")
    print(f"  $ {' '.join(shlex.quote(c) for c in cmd)}")


def prepare_bout_inp(run_dir: Path, bgrids: str, dry_run: bool) -> None:
    """Substitute $BGRIDS into BOUT.inp; BOUT.inp_orig is the template."""
    inp = run_dir / BOUT_INP
    orig = run_dir / BOUT_INP_ORIG

    if orig.is_file():
        source = orig
        print(f"  using existing template : {orig.name}")
    elif inp.is_file():
        if not dry_run:
            shutil.copy2(inp, orig)
        print(f"  saved template          : {inp.name} -> {orig.name}")
        source = orig
    else:
        die(f"'{BOUT_INP}' not found in '{run_dir}'")
        return

    text = source.read_text() if source.is_file() else ""
    new_text = text.replace("${BGRIDS}", bgrids).replace("$BGRIDS", bgrids)

    n = text.count("${BGRIDS}") + text.count("$BGRIDS")
    print(f"  substituted $BGRIDS     : {n} occurrence(s) -> {bgrids}")
    if n == 0:
        print(f"  ! warning: no $BGRIDS placeholder found in {source.name}")

    if not dry_run:
        (run_dir / BOUT_INP).write_text(new_text)


def sbatch_parsable(args: list[str], cwd: Path) -> str:
    cmd = ["sbatch", "--parsable", *args]
    print_cmd(cmd, cwd=cwd)
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        die(f"sbatch failed (exit {proc.returncode})")
    job_id = proc.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        sys.stderr.write(proc.stdout + proc.stderr)
        die(f"could not parse job ID from sbatch output: {proc.stdout!r}")
    return job_id


def submit_bout_save(run_dir: Path, save_args: list[str],
                     dep_job: str | None, after: str,
                     qos: str, walltime: str, tries: int,
                     dry_run: bool) -> None:
    wrapper = shutil.which(SAVE_WRAPPER)
    if wrapper is None:
        print(f"  ! {SAVE_WRAPPER} not found on $PATH -- skipping post-processing.",
              file=sys.stderr)
        print(f"    Queue it manually with:  {SAVE_WRAPPER} {run_dir}",
              file=sys.stderr)
        return

    env = os.environ.copy()
    env["SAVE_QOS"] = qos
    env["SAVE_TIME"] = walltime
    env["SAVE_TRIES"] = str(tries)
    env.pop("SAVE_AFTER", None)
    if dep_job:
        env["SAVE_AFTER"] = f"{after}:{dep_job}"
        print(f"  dependency              : {after}:{dep_job}")
    print(f"  qos / time / tries      : {qos} / {walltime} / {tries}")

    cmd = [wrapper, str(run_dir), *save_args]
    print_cmd(cmd)
    if dry_run:
        return

    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    sys.stdout.write("".join(f"    | {ln}\n"
                             for ln in proc.stdout.splitlines() if ln.strip()))
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        print(f"  ! {SAVE_WRAPPER} returned {proc.returncode}; "
              f"the BOUT job is still queued.", file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="bout_run",
        description="Submit a BOUT++ run directory and chain bout_save afterwards.",
    )
    p.add_argument("dir", type=Path, nargs="?", default=Path("."),
                   help="BOUT++ run directory (default: cwd)")
    p.add_argument("--no-save", action="store_true",
                   help="Submit the BOUT run only; do not chain bout_save.")
    p.add_argument("--save-only", action="store_true",
                   help="Skip the BOUT run; queue bout_save immediately.")
    p.add_argument("--save-args", default="",
                   help="Extra arguments forwarded to bout_save, as one quoted "
                        "string (e.g. --save-args '-c 1 -v N0 Ni').")
    p.add_argument("--after", default="afterany",
                   choices=["afterok", "afterany", "any", "ok"],
                   help="Slurm dependency type for the first bout_save attempt.")
    p.add_argument("--save-qos", default="debug",
                   help="QOS for bout_save attempts (default: debug).")
    p.add_argument("--save-time", default="00:30:00",
                   help="Walltime per bout_save attempt (default: 00:30:00).")
    p.add_argument("--save-tries", type=int, default=1,
                   help="Max bout_save attempts (default: 1).")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="Print the plan without submitting anything.")
    args = p.parse_args()

    after = {"ok": "afterok", "any": "afterany"}.get(args.after, args.after)

    run_dir = args.dir.resolve()
    if not run_dir.is_dir():
        die(f"directory '{args.dir}' does not exist")

    save_args = shlex.split(args.save_args)
    save_opts = dict(qos=args.save_qos, walltime=args.save_time,
                     tries=args.save_tries, dry_run=args.dry_run)

    print(f"\nrun directory : {run_dir}")

    if args.save_only:
        print("mode          : bout_save only (no BOUT submission)\n")
        submit_bout_save(run_dir, save_args, None, after, **save_opts)
        return

    if not (run_dir / SCRIPT_NAME).is_file():
        die(f"'{SCRIPT_NAME}' not found in '{run_dir}'")

    bgrids = os.environ.get("BGRIDS")
    if not bgrids:
        die("environment variable BGRIDS is not set")

    print("\n[1/3] preparing BOUT.inp")
    prepare_bout_inp(run_dir, bgrids, args.dry_run)

    print("\n[2/3] submitting BOUT run")
    if args.dry_run:
        print_cmd(["sbatch", "--parsable", SCRIPT_NAME], cwd=run_dir)
        job_id = "<jobid>"
    else:
        job_id = sbatch_parsable([SCRIPT_NAME], cwd=run_dir)
    print(f"  BOUT job ID             : {job_id}")

    if args.no_save:
        print("\n[3/3] --no-save given; bout_save not queued")
    else:
        print("\n[3/3] queueing bout_save (dependent)")
        submit_bout_save(run_dir, save_args, job_id, after, **save_opts)

    print(f"\nSubmitted job {run_dir.name}  (BOUT job {job_id})")
    if not args.dry_run:
        print(f"Watch with:  squeue -u $USER --start")
    print()


if __name__ == "__main__":
    main()
