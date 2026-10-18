#!/usr/bin/env python3
"""
next_iteration.py -- set up the next iteration directory for a SIBIL <-> BOUT++
code-coupling run.

Replaces the manual dance of:

    mv 1_iteration/Shelf_new_grid2_sinks2_gradfix/ .
    cp -r 1_iteration/ 2_iteration
    mv Shelf_new_grid2_sinks2_gradfix/ 1_iteration/
    # then hand-edit 2_iteration/SIBIL_config.yaml: iteration: "1" -> "2"
    # then hand-build 2_iteration/<bout_dir>/ with BOUT.inp, grid, job_submit.sh

What it does
------------
1. Finds the highest-numbered ``<N>_iteration`` directory in the working dir.
2. Reads ``<N>_iteration/SIBIL_config.yaml`` for ``bout_dir`` and ``bgrid``.
3. Copies ``<N>_iteration`` -> ``<N+1>_iteration``, *excluding* the BOUT run
   directory (which holds the large dump files) -- no move/move-back needed.
4. Rewrites the ``iteration:`` key in the new config to ``N+1`` (formatting and
   comments preserved; only that one line is touched).
5. Creates ``<N+1>_iteration/<bout_dir>/`` containing only BOUT.inp, the grid
   file, and job_submit.sh copied from the previous iteration's BOUT dir.

Usage
-----
    python next_iteration.py                 # operate in $PWD
    python next_iteration.py -d /path/to/run
    python next_iteration.py --dry-run       # print the plan, change nothing
    python next_iteration.py --from 3        # force source iteration number
    python next_iteration.py --extra data.nc --extra restart/
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path

try:
    import yaml
except ImportError:  # pragma: no cover
    sys.exit("PyYAML is required:  pip install pyyaml")

CONFIG_NAME = "SIBIL_config.yaml"
ITER_RE = re.compile(r"^(\d+)_iteration$")
# Files seeded into the new BOUT directory (job script name is configurable).
DEFAULT_JOB_SCRIPT = "job_submit.sh"
DEFAULT_PLOTTING_SCRIPT = "sibil_interactive_plotter.ipynb"
BOUT_INPUT = "BOUT.inp"


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def find_iterations(root: Path) -> dict[int, Path]:
    """Return {iteration_number: path} for every <N>_iteration dir under root."""
    found: dict[int, Path] = {}
    for entry in root.iterdir():
        if not entry.is_dir():
            continue
        m = ITER_RE.match(entry.name)
        if m:
            found[int(m.group(1))] = entry
    return found


def latest_iteration(root: Path, forced: int | None = None) -> tuple[int, Path]:
    iters = find_iterations(root)
    if not iters:
        sys.exit(f"No '<N>_iteration' directories found in {root}")
    if forced is not None:
        if forced not in iters:
            sys.exit(f"Requested iteration {forced} not found in {root} "
                     f"(have: {sorted(iters)})")
        return forced, iters[forced]
    n = max(iters)
    return n, iters[n]


# --------------------------------------------------------------------------- #
# config handling
# --------------------------------------------------------------------------- #
def load_config(cfg_path: Path) -> dict:
    if not cfg_path.is_file():
        sys.exit(f"Missing config file: {cfg_path}")
    with cfg_path.open() as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        sys.exit(f"{cfg_path} did not parse into a mapping.")
    return data


def config_get(cfg: dict, key: str) -> str | None:
    """Fetch `key` from the top level, or from any one-level-nested mapping."""
    if key in cfg:
        return cfg[key]
    for value in cfg.values():
        if isinstance(value, dict) and key in value:
            return value[key]
    return None


def bump_iteration_line(cfg_path: Path, new_value: int, dry_run: bool) -> None:
    """Rewrite only the `iteration:` line, preserving quoting style/comments."""
    text = cfg_path.read_text()
    pattern = re.compile(
        r"""^(?P<indent>\s*)(?P<key>iteration)(?P<sep>\s*:\s*)
            (?P<q>["']?)(?P<val>[^"'\s#]*)(?P=q)
            (?P<tail>.*)$""",
        re.MULTILINE | re.VERBOSE,
    )

    matches = list(pattern.finditer(text))
    if not matches:
        sys.exit(f"Could not find an 'iteration:' key in {cfg_path}")
    if len(matches) > 1:
        print(f"  ! {len(matches)} 'iteration:' keys found; updating the first one.")

    m = matches[0]
    quote = m.group("q") or ""
    replacement = (f"{m.group('indent')}{m.group('key')}{m.group('sep')}"
                   f"{quote}{new_value}{quote}{m.group('tail')}")
    new_text = text[:m.start()] + replacement + text[m.end():]

    print(f"  iteration: {m.group('q')}{m.group('val')}{m.group('q')}"
          f"  ->  {quote}{new_value}{quote}")
    if not dry_run:
        cfg_path.write_text(new_text)


# --------------------------------------------------------------------------- #
# copying
# --------------------------------------------------------------------------- #
def copy_iteration(src: Path, dst: Path, exclude: set[str], dry_run: bool) -> None:
    """Copy the iteration tree, skipping top-level names in `exclude`."""
    print(f"  copy tree {src}  ->  {dst}   (excluding: {', '.join(sorted(exclude)) or 'nothing'})")
    if dry_run:
        return
    ignore = shutil.ignore_patterns(*exclude) if exclude else None
    shutil.copytree(src, dst, ignore=ignore, symlinks=True)


def copy_item(src: Path, dst: Path, dry_run: bool, required: bool = True) -> bool:
    """Copy a file or directory; returns True if copied."""
    if not src.exists():
        msg = f"  ! missing: {src}"
        if required:
            sys.exit(msg + "   (required)")
        print(msg + "   (skipped)")
        return False
    print(f"  copy {src}  ->  {dst}")
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, symlinks=True, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)
    return True

def link_item(src: Path, dst: Path, dry_run: bool) -> bool:
    """Links a file or directory; returns True if linked."""
    if not src.exists():
        msg = f"  ! missing: {src}"
        print(msg + "   (skipped)")
        return False
    print(f"  ln -s {src}  ->  {dst}")
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not src.is_file():
        msg = f"  ! not a file: {src}"
    else:
        if src.exists() or src.is_symlink():
            print(f"Skipping existing: {src}")
            return
        src.symlink_to(dst)
    return True

# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser(
        description="Set up the next <N>_iteration directory for a SIBIL/BOUT++ coupling run.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("-d", "--dir", default=".", type=Path,
                   help="Directory containing the <N>_iteration folders.")
    p.add_argument("--from", dest="from_iter", type=int, default=None,
                   help="Source iteration number (default: highest found).")
    p.add_argument("--job-script", default=DEFAULT_JOB_SCRIPT,
                   help="Name of the batch submission script inside the BOUT dir.")
    p.add_argument("-l", "--link-plotter", type=bool, default=True,
                   help="Link interactive jupyter plotting script from scripts dir.")
    p.add_argument("--extra", action="append", default=[],
                   metavar="PATH",
                   help="Additional path (relative to the old BOUT dir) to seed "
                        "into the new BOUT dir. Repeatable.")
    p.add_argument("-n", "--dry-run", action="store_true",
                   help="Show what would happen without touching the filesystem.")
    p.add_argument("-f", "--force", action="store_true",
                   help="Delete the destination iteration directory if it exists.")
    args = p.parse_args()

    root = args.dir.resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    # 1. locate current iteration -------------------------------------------
    cur_n, cur_dir = latest_iteration(root, args.from_iter)
    new_n = cur_n + 1
    new_dir = root / f"{new_n}_iteration"
    print(f"\nroot            : {root}")
    print(f"current         : {cur_dir.name}")
    print(f"next            : {new_dir.name}")

    if new_dir.exists():
        if not args.force:
            sys.exit(f"\nDestination already exists: {new_dir}\n"
                     f"Re-run with --force to replace it.")
        print(f"  ! removing existing {new_dir}")
        if not args.dry_run:
            shutil.rmtree(new_dir)

    # 2. read config ---------------------------------------------------------
    cfg_path = cur_dir / CONFIG_NAME
    cfg = load_config(cfg_path)

    bout_dir_rel = config_get(cfg, "bout_dir")
    bgrid_rel = config_get(cfg, "bgrid")
    if not bout_dir_rel:
        sys.exit(f"No 'bout_dir' key found in {cfg_path}")
    if not bgrid_rel:
        sys.exit(f"No 'bgrid' key found in {cfg_path}")

    bout_dir_rel = str(bout_dir_rel).strip().rstrip("/")
    bgrid_rel = str(bgrid_rel).strip()
    grid_name = os.path.basename(bgrid_rel)

    old_bout = cur_dir / bout_dir_rel
    new_bout = new_dir / bout_dir_rel

    print(f"bout_dir        : {bout_dir_rel}")
    print(f"bgrid           : {bgrid_rel}")
    for key in ("gfile", "geofile"):
        val = config_get(cfg, key)
        if val is not None:
            print(f"{key:<16}: {val}")

    if not old_bout.is_dir():
        sys.exit(f"\nBOUT directory not found: {old_bout}")

    # 3. copy the iteration tree minus the heavy BOUT run dir ---------------
    print("\n[1] duplicating iteration directory")
    copy_iteration(cur_dir, new_dir, exclude={"b2plasmf", os.path.basename(bout_dir_rel)},
                   dry_run=args.dry_run)

    # 4. copy the interactive plotter into sibil dir if desired -------------
    scripts_dir = Path(os.getenv('SCRIPTS'))
    interactive_plotter = scripts_dir / DEFAULT_PLOTTING_SCRIPT
    link_item(interactive_plotter,new_dir,args.dry_run)

    # 5. bump the iteration counter -----------------------------------------
    print("\n[2] updating SIBIL_config.yaml")
    new_cfg = new_dir / CONFIG_NAME
    if args.dry_run and not new_cfg.exists():
        new_cfg = cfg_path  # read the source copy just to report the change
    bump_iteration_line(new_cfg, new_n, args.dry_run)

    # 6. seed the fresh BOUT directory --------------------------------------
    print(f"\n[3] seeding new BOUT directory: {new_bout}")
    if not args.dry_run:
        new_bout.mkdir(parents=True, exist_ok=True)

    # grid path in the config may be given relative to the iteration dir
    # (e.g. "<bout_dir>/xxx.grd.nc") or as a bare filename inside the BOUT dir.
    grid_src = cur_dir / bgrid_rel
    if not grid_src.exists():
        grid_src = old_bout / grid_name

    copy_item(old_bout / BOUT_INPUT, new_bout / BOUT_INPUT, args.dry_run)
    copy_item(grid_src, new_bout / grid_name, args.dry_run)
    copy_item(old_bout / args.job_script, new_bout / args.job_script,
              args.dry_run, required=False)

    for extra in args.extra:
        copy_item(old_bout / extra, new_bout / extra, args.dry_run, required=False)
        
    print(f"\n{'[dry run] ' if args.dry_run else ''}"
          f"{new_dir.name} is ready.\n")


if __name__ == "__main__":
    main()
