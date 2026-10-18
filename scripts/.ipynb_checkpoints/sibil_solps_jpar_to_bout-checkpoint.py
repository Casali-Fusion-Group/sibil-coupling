#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sibil_solps_jpar_to_bout.py   (v6)

Put a physically sensible equilibrium parallel current density ``Jpar0``
into an existing BOUT++ (elm-6f) grid file, combining

  * hypnotoad's g-file current -- the *real* equilibrium current (Ohmic +
    bootstrap), correct in the core but truncated to exactly zero outside
    the LCFS, and
  * the SOLPS-ITER edge current closure from ``b2fstate`` -- the only thing
    that knows about thermoelectric / sheath / Pfirsch-Schlueter current in
    the SOL, but which contains *no* equilibrium current at all.

Nothing else in the grid file is touched.

    python sibil_solps_jpar_to_bout.py SIBIL_config.yaml

---------------------------------------------------------------------------
v6 -- what changed
---------------------------------------------------------------------------
**Every option now comes from the config file.**  The command line takes
exactly one argument, the path to the YAML, and nothing else that affects
physics.  All the flags that used to live on argparse -- ``--mode``,
``--lam-j-mm``, ``--blend-width``, ``--plot``, ``--dry-run``, ... -- are
read from a dedicated section::

    solps_to_bout_pass:
      mode:         add
      lam_j_mm:     5.0
      match_band:   0.015
      blend_width:  0.040
      plot:         true
      png:          testing-jpar-map.png
      dry_run:      true

Why this is better than flags for this particular script: the output is a
*modified grid file*, and a grid file with no record of how it was made is
a liability six months later.  With the settings in the YAML, the config
that produced the grid sits next to the grid, in version control, and a
re-run is reproducible by construction rather than by shell history.

Three support features come with it:

1. ``--print-template`` dumps a fully commented ``solps_to_bout_pass``
   block with every key at its default value.  Paste it into your YAML and
   edit.  (This is the only remaining non-config command-line option, plus
   ``--section`` for the rare case of keeping several named presets in one
   file.)

2. **Unknown keys are errors, not silent no-ops.**  A typo like
   ``blend_with: 0.04`` used to be ignored; now it stops the run and
   suggests the nearest valid key.  The whole reason you hit the
   ``--blend-width`` confusion in v5 was a flag that looked applied and
   was not.

3. **The resolved settings are echoed** at the top of every run, with a
   marker showing which values came from the file and which are defaults.
   Nothing is applied invisibly.

Paths are still read from wherever they already live in your config
(top level, ``rundir``, or whatever nesting you use) -- the resolver
flattens the whole document -- but an explicit path inside
``solps_to_bout_pass`` wins if you set one.

---------------------------------------------------------------------------
v5 -- inline C1 separatrix repair (unchanged, still in force)
---------------------------------------------------------------------------
v4 extended Jpar0 outside the separatrix with a one-sided exponential
anchored on the last interior value::

    J(p) = J0 * exp(-(p - pm) / lam)

That is continuous in J but not in dJ/dpsi_n: hypnotoad's interior profile
is essentially flat (J' ~ 0) while the exponential starts with slope
-J0/lam.  For J0 = -0.335 MA/m^2 and lam = 0.010 in psi_n that is a step of
~33 MA/m^2 per unit psi_n across one cell face.  Current gradient is the
peeling drive, so you traded a step in J for a step in J'.

v5 matches value AND slope on the core side, extrapolates linearly, and
smoothsteps into the exponential tail over a finite width::

    fit over [pm - match_band, pm]  ->  J0 = J(pm), s = dJ/dpsi_n(pm)

    J_lin(p) = J0 + s * (p - pm)
    J_exp(p) = J0 * exp(-(p - pm) / lam)
    t        = (p - pm) / blend_width
    w        = 3t^2 - 2t^3                   (clipped to [0, 1])
    J(p)     = (1 - w) * J_lin(p) + w * J_exp(p)

Because w(0) = w'(0) = 0 and w(1) = 1, w'(1) = 0, the result matches the
core in both value and slope at p = pm, becomes the pure exponential at
p = pm + blend_width, and is C1 in between.  ``blend_width: 0``
reproduces v4 exactly.

Measured on a synthetic reproduction of d3d_shelf_184016.03200
(J0 = -0.335 MA/m^2, flat interior, lam = 0.010), reporting
max|dJ/dpsi_n| over 0.95 < psi_n < 1.10::

    original hypnotoad cliff        255.7     1.00x
    one-sided exponential (v4)       30.2     0.12x
    C1, blend_width = 0.005          41.9     0.16x   <-- WORSE
    C1, blend_width = 0.020          23.4     0.09x
    C1, blend_width = 0.040          13.2     0.05x
    C1, blend_width = 0.060           8.8     0.03x

Note the 0.005 row.  A narrow handoff out of a near-zero-slope linear
segment creates its own kink; you need blend_width >~ 2*lam before it
helps at all.  The code warns when you violate that.

---------------------------------------------------------------------------
Why additive
---------------------------------------------------------------------------
SOLPS takes B as fixed input.  No toroidal loop voltage, no bootstrap
model, no Ampere constraint, so its current is the *edge closure* only.
Replacing ``Jpar0`` with it throws away the plasma current.  For this run

    max |J_par| (SOLPS, fch_p)   = 0.0223 MA/m^2   (0.0140 at the OMP)
    core  J_par (hypnotoad)      = 0.335  MA/m^2   -> SOLPS is ~4.2% at OMP

and ``check_fch_conventions.py`` returned a median relative ``|div j|`` of
2.0e-14 for the ``normal`` convention, i.e. round-off, with a 5e10
separation from the runner-up.  The extraction is correct; the current
really is small.  ``fchdia`` and ``fch_32`` are identically zero here
because the drift terms are off, so the Pfirsch-Schlueter contribution is
*absent* -- enabling drifts would roughly double the edge current but would
not close the gap, because that gap is Ohmic + bootstrap.

So::

    Jpar0 = Jpar_hyp_extended  +  w(psi_n) * Jpar_SOLPS

What this does NOT fix
----------------------
Hypnotoad builds Jpar0 from the g-file's pprime / TTprime, defined only
inside the LCFS and carrying no pedestal-resolved bootstrap term.  The
interior profile is therefore flat, the fitted slope s is ~0, and no amount
of blending can invent the H-mode bootstrap peak that should sit near
psi_n ~ 0.97-0.99.  lam_J stays a free parameter your growth rate depends
on: quote it and scan 3-10 mm.  A quantitatively correct pedestal current
needs an equilibrium re-solved with the bootstrap current as a constraint
(CHEASE / HELENA / FreeGS, or a kinetic EFIT) -- see
``edge_current_diagnostic.py`` for whether that is necessary in your case.

Modes
-----
  add          Jpar_hyp_extended + w * Jpar_SOLPS            (default)
  repair-only  Jpar_hyp_extended                             (no SOLPS)
  solps-blend  SOLPS everywhere, blended to hypnotoad in core (v1 behaviour)
  replace      SOLPS everywhere, no blend                     (debug)

Requires ``gqdsk_compat.py`` and ``boutgrid_compat.py`` on PYTHONPATH.

@author: ttaczak
"""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import sys

import numpy as np
import scipy.interpolate as si

MU0 = 4.0e-7 * np.pi
QE = 1.602176634e-19

#: the config section this script reads
SECTION = "solps_jpar_pass"


# ===========================================================================
# ===========================================================================
#  SETTINGS  --  the single source of truth for every option
# ===========================================================================
# ===========================================================================
#
# Each entry is:
#     key: (default, type, help)
#
# type is one of "str", "float", "float_or_none", "int", "bool", or a tuple
# of allowed strings (a choice).  Everything the script does is driven by
# this table: the defaults, the validation, the echo at the top of the run,
# and the --print-template output.  Add an option here and it works
# everywhere; there is no second place to update.
#
_SPEC = {
    # ---- paths -----------------------------------------------------------
    "b2fstate": (None, "str",
                 "SOLPS state file.  Defaults to the top-level config entry, "
                 "or <rundir>/b2fstate, or ./b2fstate."),
    "b2fgmtry": (None, "str",
                 "SOLPS geometry file.  Same fallback chain as b2fstate."),
    "geofile": (None, "str",
                "Optional SOLPS .geo file for cell-centre R/Z.  If it is "
                "absent or unreadable the code falls back to b2fgmtry "
                "crx/cry, which is if anything more reliable."),
    "gfile": (None, "str",
              "EFIT g-file.  Supplies psi(R,Z) so every SOLPS cell can be "
              "tagged with psi_n."),
    "bgrid": (None, "str",
              "The hypnotoad BOUT++ grid to modify."),

    # ---- what to build ---------------------------------------------------
    "mode": ("add", ("add", "repair-only", "solps-blend", "replace"),
             "add          Jpar_hyp_extended + w * Jpar_SOLPS\n"
             "repair-only  Jpar_hyp_extended only (isolates the separatrix "
             "repair -- run this alongside 'add' to see how much of the "
             "change is SOLPS and how much is the cliff)\n"
             "solps-blend  SOLPS everywhere, blended to hypnotoad in core\n"
             "replace      SOLPS everywhere, no blend (debug)"),
    "sign": (1.0, "float",
             "Flip the SOLPS parallel direction (+1 or -1).  The B2 poloidal "
             "direction need not match BOUT++'s parallel direction.  Matters "
             "little in 'add' mode, where SOLPS is ~4% of the total."),
    "fch_convention": ("normal", ("normal", "swapped"),
                       "Which fch component is poloidal.  Only consulted if "
                       "b2fstate has no fch_p.  check_fch_conventions.py "
                       "decides this by charge conservation; for "
                       "d3d_shelf_184016.03200 'normal' wins by 5e10."),

    # ---- where SOLPS is allowed to act ------------------------------------
    "w_on": (0.98, "float",
             "psi_n where the SOLPS window starts opening.  Keep this above "
             "the innermost SOLPS ring: that ring sits on a Dirichlet "
             "potential boundary, so its current is set by the BC rather "
             "than by physics."),
    "w_full": (1.00, "float",
               "psi_n where the SOLPS window is fully open."),
    "core_blend": (0.05, "float",
                   "Blend width for mode 'solps-blend'.  Ignored otherwise."),

    # ---- the C1 separatrix repair -----------------------------------------
    "lam_j": (0.010, "float",
              "SOL current decay length in psi_n units.  Superseded by "
              "lam_j_mm if that is set."),
    "lam_j_mm": (None, "float_or_none",
                 "Decay length in millimetres; converted to psi_n using the "
                 "grid's own dpsi_n/dr at the separatrix.  Easier to reason "
                 "about than lam_j.  5 mm ~ 0.010 in psi_n for this case."),
    "match_band": (0.015, "float",
                   "psi_n width of the core-side window used to fit value "
                   "AND slope at the separatrix.  0.015 is ~3-4 radial "
                   "cells here."),
    "blend_width": (0.040, "float",
                    "psi_n width over which the linear extension hands off "
                    "to the exponential tail.  This is the knob that removes "
                    "the kink in dJ/dpsi_n.  Use >= 2*lam_j; 0 reproduces "
                    "the v4 one-sided exponential exactly."),
    "psin_match": (1.000, "float",
                   "psi_n at which the repair anchors onto hypnotoad.  1.0 "
                   "is the separatrix, which is what the C1 construction "
                   "assumes."),
    "pfr_zero": (True, "bool",
                 "Zero hypnotoad's private-flux-region current.  In the PFR "
                 "hypnotoad evaluates J(psi_n) at psi_n < 1 even though the "
                 "region is not core, so that current is meaningless."),
    "trunc_frac": (0.25, "float",
                   "Guard rail: a column is only repaired if its SOL side "
                   "currently satisfies max|J| < trunc_frac * |J0|, i.e. "
                   "hypnotoad really did truncate it.  Makes the repair "
                   "idempotent and safe to re-run."),
    "cap_frac": (1.5, "float",
                 "The linear extension is magnitude-capped at "
                 "cap_frac * |J0| so a noisy fitted slope cannot overshoot "
                 "before the smoothstep damps it."),

    # ---- SOLPS profile extraction -----------------------------------------
    "psin_max": (None, "float_or_none",
                 "Trim SOLPS cells above this psi_n.  Divertor-leg cells can "
                 "carry psi_n values that collide with core ones and corrupt "
                 "the monotonic sort; 1.05-1.10 is usually right."),
    "smooth": (0.0, "float",
               "Smoothing-spline factor on the OMP profile.  SOLPS currents "
               "are radially noisy because they are differences of large "
               "fluxes; 5e-3 is a reasonable starting point if needed."),
    "extrap": ("flat", ("flat", "zero", "exp"),
               "Behaviour outside the SOLPS radial domain.  'flat' is "
               "safest for a current."),

    # ---- output -----------------------------------------------------------
    "out": (None, "str",
            "Write to this file instead of modifying bgrid in place.  When "
            "writing in place a .bak is made on first run."),
    "write_apar": (False, "bool",
                   "Also write a diagnostic Apar0 from radial integration of "
                   "mu0 * J_par.  Plotting aid only -- elm-6f wants Jpar0, "
                   "and Apar0 is gauge-dependent."),
    "plot": (False, "bool",
             "Draw the three-panel summary."),
    "png": (None, "str",
            "Save the plot here instead of showing it.  Implies plot."),
    "dry_run": (False, "bool",
                "Compute and report, but do not touch the grid file.  Always "
                "worth doing once before a real write."),
    "quiet": (False, "bool",
              "Suppress the running commentary."),
}

#: keys that name a filesystem path, and the aliases they may appear under
#: elsewhere in the config document
_PATH_FALLBACKS = {
    "b2fstate": ("b2fstate", "statefile", "b2fstate_file"),
    "b2fgmtry": ("b2fgmtry", "gmtryfile", "b2fgmtry_file"),
    "geofile": ("geofile", "geo", "solps_geo"),
    "gfile": ("gfile", "geqdsk", "efit", "eqdsk"),
    "bgrid": ("bgrid", "gridfile", "bout_grid", "grid"),
}


class Settings(object):
    """Resolved options, exposed as plain attributes.

    Deliberately shaped like the old argparse Namespace so the rest of the
    script did not need rewriting when the options moved into the config.
    ``origin`` records where each value came from, which is what the echo
    at the top of the run prints.
    """

    def __init__(self, values, origin):
        for k, v in values.items():
            setattr(self, k, v)
        self._origin = origin

    # the repair takes the positive form; the old code used the negative
    @property
    def no_pfr_zero(self):
        return not self.pfr_zero

    def echo(self, stream=sys.stdout):
        print("", file=stream)
        print("=" * 74, file=stream)
        print("settings  [{}]".format(SECTION), file=stream)
        print("=" * 74, file=stream)
        width = max(len(k) for k in _SPEC)
        for k in _SPEC:
            v = getattr(self, k)
            mark = {"config": "<-", "default": "  ",
                    "derived": "~ "}[self._origin.get(k, "default")]
            print("  {} {:<{w}}  {}".format(mark, k, _fmt_value(v), w=width),
                  file=stream)
        print("", file=stream)
        print("  '<-' from config,  '~' derived,  blank = default",
              file=stream)


def _fmt_value(v):
    if v is None:
        return "(none)"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return "{:g}".format(v)
    return str(v)


# ===========================================================================
# config reading
# ===========================================================================
def _flatten(d, out=None):
    """Flatten a nested document so path lookups work regardless of nesting."""
    if out is None:
        out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (dict, list)):
                _flatten(v, out)
            else:
                out[str(k).lower()] = v
    elif isinstance(d, list):
        for v in d:
            _flatten(v, out)
    return out


def _coerce(key, raw, kind):
    """Turn a YAML scalar into the type the spec demands, or explain why not."""
    if isinstance(kind, tuple):
        s = str(raw).strip()
        if s not in kind:
            raise ValueError(
                "{}: '{}' is not one of {}".format(key, s, ", ".join(kind)))
        return s

    if kind == "str":
        if raw is None:
            return None
        s = str(raw).strip()
        return s if s and s.lower() not in ("none", "null", "~") else None

    if kind == "bool":
        if isinstance(raw, bool):
            return raw
        s = str(raw).strip().lower()
        if s in ("true", "yes", "on", "1"):
            return True
        if s in ("false", "no", "off", "0"):
            return False
        raise ValueError("{}: '{}' is not a boolean".format(key, raw))

    if kind in ("float", "float_or_none"):
        if raw is None or str(raw).strip().lower() in ("none", "null", "~", ""):
            if kind == "float_or_none":
                return None
            raise ValueError("{}: needs a number, got nothing".format(key))
        try:
            return float(raw)
        except (TypeError, ValueError):
            raise ValueError("{}: '{}' is not a number".format(key, raw))

    if kind == "int":
        try:
            return int(raw)
        except (TypeError, ValueError):
            raise ValueError("{}: '{}' is not an integer".format(key, raw))

    raise AssertionError("bad spec kind {}".format(kind))


def load_settings(path, section=SECTION):
    """Read ``section`` out of the YAML and resolve every option.

    Unknown keys are a hard error with a spelling suggestion.  That is
    deliberate: a mistyped key that silently does nothing is exactly the
    failure mode this rewrite exists to remove.
    """
    try:
        import yaml
    except ImportError:
        raise ImportError("pyyaml is required to read the config file")

    if not os.path.isfile(path):
        raise SystemExit("no such config file: {}".format(path))
    with open(path, "r") as fh:
        doc = yaml.safe_load(fh) or {}

    if not isinstance(doc, dict):
        raise SystemExit("config file does not parse to a mapping")

    block = doc.get(section)
    if block is None:
        # tolerate the section being nested one level down
        for v in doc.values():
            if isinstance(v, dict) and section in v:
                block = v[section]
                break
    if block is None:
        block = {}
        print("warning: config has no '{}' section -- running entirely on "
              "defaults.  Use --print-template to generate one."
              .format(section), file=sys.stderr)
    if not isinstance(block, dict):
        raise SystemExit("'{}' must be a mapping, not {}"
                         .format(section, type(block).__name__))

    # normalise keys: hyphens -> underscores, case-insensitive
    given = {}
    for k, v in block.items():
        given[str(k).strip().lower().replace("-", "_")] = v

    # reject typos loudly
    unknown = [k for k in given if k not in _SPEC]
    if unknown:
        msg = ["unrecognised key(s) in '{}':".format(section)]
        for k in unknown:
            near = difflib.get_close_matches(k, list(_SPEC), n=1, cutoff=0.6)
            msg.append("    {}{}".format(
                k, "   (did you mean '{}'?)".format(near[0]) if near else ""))
        msg.append("")
        msg.append("run with --print-template to see every valid key")
        raise SystemExit("\n".join(msg))

    # legacy spelling: no_pfr_zero was the v4/v5 flag
    if "no_pfr_zero" in given:
        raise SystemExit(
            "'no_pfr_zero' was replaced by its positive form.  Use\n"
            "    pfr_zero: false\n"
            "instead -- double negatives in config files age badly.")

    values, origin = {}, {}
    for key, (default, kind, _help) in _SPEC.items():
        if key in given:
            values[key] = _coerce(key, given[key], kind)
            origin[key] = "config"
        else:
            values[key] = default
            origin[key] = "default"

    # ---- paths: fall back to anywhere else in the document ------------
    flat = _flatten(doc)
    rundir = None
    for n in ("rundir", "run_dir", "solps_dir", "solpsdir", "path"):
        if flat.get(n):
            rundir = str(flat[n])
            break

    for key, aliases in _PATH_FALLBACKS.items():
        if values[key]:
            continue
        for n in aliases:
            if flat.get(n):
                values[key] = str(flat[n])
                origin[key] = "derived"
                break

    if rundir and os.path.isdir(rundir):
        for key, fname in (("b2fstate", "b2fstate"),
                           ("b2fgmtry", "b2fgmtry"),
                           ("geofile", "solps.geo")):
            if not values[key]:
                cand = os.path.join(rundir, fname)
                if os.path.isfile(cand):
                    values[key] = cand
                    origin[key] = "derived"

    for key, fname in (("b2fstate", "b2fstate"), ("b2fgmtry", "b2fgmtry")):
        if not values[key] and os.path.isfile(fname):
            values[key] = fname
            origin[key] = "derived"

    # ---- light consistency work ---------------------------------------
    if values["png"] and not values["plot"]:
        values["plot"] = True
        origin["plot"] = "derived"

    if values["w_full"] < values["w_on"]:
        raise SystemExit("w_full ({}) must be >= w_on ({})"
                         .format(values["w_full"], values["w_on"]))

    return Settings(values, origin)


def print_template(stream=sys.stdout):
    """Emit a fully commented section the user can paste into their YAML."""
    print("# ----------------------------------------------------------------",
          file=stream)
    print("# {}  --  read by sibil_solps_jpar_to_bout.py".format(SECTION),
          file=stream)
    print("# Every key below is optional; the value shown is the default.",
          file=stream)
    print("# ----------------------------------------------------------------",
          file=stream)
    print("{}:".format(SECTION), file=stream)
    for key, (default, kind, help_text) in _SPEC.items():
        print("", file=stream)
        for line in help_text.split("\n"):
            # wrap at a sensible width without pulling in textwrap gymnastics
            words, cur = line.split(), ""
            for w in words:
                if len(cur) + len(w) + 1 > 68:
                    print("  # {}".format(cur), file=stream)
                    cur = w
                else:
                    cur = (cur + " " + w).strip()
            if cur:
                print("  # {}".format(cur), file=stream)
        if isinstance(kind, tuple):
            print("  #   one of: {}".format(", ".join(kind)), file=stream)
        val = _fmt_value(default)
        if default is None:
            print("  # {}:".format(key), file=stream)
        else:
            print("  {}: {}".format(key, val), file=stream)


def _hdr(title):
    print("")
    print("=" * 74)
    print(title)
    print("=" * 74)


# ===========================================================================
# flexible imports
# ===========================================================================
def _import_readers():
    """Return (B2fstate, B2fgmtry, GQDSK) however they happen to be installed."""
    try:
        import sibil
        have = [getattr(sibil, n, None)
                for n in ("B2fstate", "B2fgmtry", "GQDSK")]
        if all(h is not None for h in have):
            return tuple(have)
    except ImportError:
        pass

    patterns = ("sibil.readers.{}_Reader", "readers.{}_Reader",
                "{}_Reader", "{}")
    out = {}
    for stem in ("B2fstate", "B2fgmtry", "GQDSK"):
        mod = None
        for pat in patterns:
            try:
                mod = __import__(pat.format(stem), fromlist=[stem])
                break
            except ImportError:
                continue
        if mod is None:
            raise ImportError(
                "cannot import {}_Reader -- put it on PYTHONPATH or run from "
                "the directory containing the readers".format(stem))
        out[stem] = getattr(mod, stem)
    return out["B2fstate"], out["B2fgmtry"], out["GQDSK"]


def _import_datafile():
    try:
        from boututils.datafile import DataFile
        return DataFile
    except ImportError:
        raise ImportError("boututils is required to read/write the grid file")


def _import_eq_resolver():
    try:
        from gqdsk_compat import resolve_equilibrium
        return resolve_equilibrium
    except ImportError:
        raise ImportError(
            "gqdsk_compat.py not found on PYTHONPATH.  It resolves SIBIL's "
            "GQDSK attribute names (psiaxis/psiedge/psi/R_grid/Z_grid) to the "
            "classic geqdsk spellings and verifies the psi array orientation.")


def _import_grid_resolver():
    try:
        from boutgrid_compat import resolve_grid_psi
        return resolve_grid_psi
    except ImportError:
        raise ImportError(
            "boutgrid_compat.py not found on PYTHONPATH.  It resolves the "
            "grid's psi normalisation (psi_bdry vs psi_bndry vs *_gfile) and "
            "validates it against the ixseps1 separatrix location.")


# ===========================================================================
# small helpers
# ===========================================================================
def _get(obj, *names):
    for n in names:
        v = getattr(obj, n, None)
        if v is not None:
            return v
    if isinstance(obj, dict):
        for n in names:
            if n in obj and obj[n] is not None:
                return obj[n]
    return None


def smoothstep(x, lo, hi):
    """C1 smoothstep: 0 for x <= lo, 1 for x >= hi."""
    if hi <= lo:
        return (np.asarray(x) >= hi).astype(float)
    t = np.clip((np.asarray(x, dtype=float) - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smoothstep(t):
    """Cubic Hermite smoothstep on a pre-normalised argument.

    Zero slope at BOTH ends is the whole point -- it is what makes the
    handoff from the linear extension to the exponential tail C1 rather
    than merely continuous.
    """
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# ===========================================================================
# SOLPS side
# ===========================================================================
class SolpsJpar(object):
    """Extract J_par [A/m^2] on the SOLPS mesh and tag every cell with psi_n."""

    def __init__(self, statefile, gmtryfile, gfile, geofile=None,
                 convention="normal", sign=1.0, verbose=True):
        self.verbose = verbose
        self.sign = float(sign)
        self.convention = convention
        self.eq = None                      # filled by _assign_psin

        B2fstate, B2fgmtry, GQDSK = _import_readers()

        self._log("reading {}".format(os.path.basename(gmtryfile)))
        self.gm = B2fgmtry(gmtryfile)
        self._log("reading {}".format(os.path.basename(statefile)))
        self.st = B2fstate(statefile)

        self._shape()
        self._geometry(geofile)
        self._compute_jpar()
        self._assign_psin(gfile)
        self._locate_midplane()

    # ------------------------------------------------------------------
    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _shape(self):
        fch = _get(self.st, "fch")
        if fch is None:
            raise KeyError("b2fstate has no 'fch' array -- currents were not "
                           "solved in this run")
        self.fch = np.asarray(fch, dtype=float)
        self.npol, self.nrad = self.fch.shape[0], self.fch.shape[1]
        self._log("    SOLPS mesh (incl. guards): npol={} nrad={}"
                  .format(self.npol, self.nrad))

    # ------------------------------------------------------------------
    def _geometry(self, geofile):
        gm = self.gm
        self.hx = np.asarray(_get(gm, "hx"), dtype=float)
        self.hy = np.asarray(_get(gm, "hy"), dtype=float)
        hz = _get(gm, "hz")
        vol = _get(gm, "vol")
        self.vol = None if vol is None else np.asarray(vol, dtype=float)

        # ---- poloidal face area ---------------------------------------
        if self.vol is not None and np.nanmedian(np.abs(self.hx)) > 0:
            with np.errstate(divide="ignore", invalid="ignore"):
                self.Apol = self.vol / self.hx
            self._log("    poloidal face area from vol/hx")
        else:
            hz = np.asarray(hz, dtype=float)
            self.Apol = self.hy * hz
            self._log("    poloidal face area from hy*hz")

        # ---- R, Z at cell centres -------------------------------------
        self.Rc = self.Zc = None
        if geofile and os.path.isfile(geofile):
            try:
                self.Rc, self.Zc = self._read_geo(geofile)
                self._log("    R/Z from .geo file")
            except Exception as exc:                        # noqa: BLE001
                self._log("    .geo read failed ({}), using b2fgmtry"
                          .format(exc))
                self.Rc = self.Zc = None
        if self.Rc is None:
            crx = _get(gm, "crx")
            cry = _get(gm, "cry")
            if crx is None or cry is None:
                raise KeyError("no crx/cry in b2fgmtry and no usable .geo")
            crx = np.asarray(crx, dtype=float)
            cry = np.asarray(cry, dtype=float)
            self.Rc = np.mean(crx.reshape(self.npol, self.nrad, -1), axis=2)
            self.Zc = np.mean(cry.reshape(self.npol, self.nrad, -1), axis=2)
            self._log("    R/Z from b2fgmtry crx/cry")

        # ---- B components ---------------------------------------------
        self.bb = _get(gm, "bb")
        self.Btot = self.Bpol = None
        if self.bb is not None:
            bb = np.asarray(self.bb, dtype=float)
            if bb.ndim == 3 and bb.shape[-1] >= 4:
                med = np.array([np.nanmedian(np.abs(bb[..., k]))
                                for k in range(bb.shape[-1])])
                itot = int(np.nanargmax(med))
                # |B_pol| = smallest *non-negligible* component.  index 1 is
                # b_rad and is identically zero when drifts are off, so a
                # plain argmin picks it and every division becomes 0/0.
                thresh = 1e-3 * med[itot]
                cand = [k for k in range(len(med))
                        if k != itot and med[k] > thresh]
                ipol = int(min(cand, key=lambda k: med[k])) if cand else 0
                self.Btot = np.abs(bb[..., itot])
                self.Bpol = np.abs(bb[..., ipol])
                self._log("    bb: |B| = index {}, |B_pol| = index {} "
                          "(medians {})".format(itot, ipol, np.round(med, 3)))

    @staticmethod
    def _read_geo(path):
        """Pull cell-centre R, Z out of a SOLPS .geo file if present."""
        import re
        with open(path, "r") as fh:
            txt = fh.read()
        got = {}
        for m in re.finditer(r"\*cf:\s*\S+\s+(\d+)\s+(\S+)", txt):
            label = m.group(2)
            if label not in ("cr", "cz", "crx", "cry"):
                continue
            n = int(m.group(1))
            vals = txt[m.end():].split()[:n]
            got[label] = np.array([v.replace("D", "E") for v in vals],
                                  dtype=float)
        if "cr" in got and "cz" in got:
            return got["cr"], got["cz"]
        raise ValueError("geo file lacks cell-centre R/Z")

    # ------------------------------------------------------------------
    def _compute_jpar(self):
        """J_par [A/m^2] at cell centres."""
        fch_p = _get(self.st, "fch_p")

        if fch_p is not None and np.nanmax(np.abs(np.asarray(fch_p,
                                                             float))) > 0:
            # SOLPS's own parallel projection -- sidesteps the bb index
            # ambiguity entirely.
            self._log("    using b2fstate parallel-current array directly")
            f = np.asarray(fch_p, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                jpar = f / self.Apol
            src = "fch_p"
        else:
            fchx = self.fch[..., 0]
            if self.convention.startswith("swapped"):
                fchx = self.fch[..., 1]
            # value at ix is the flux through the LOWER-ix face, so the two
            # poloidal faces of cell ix are fchx[ix] and fchx[ix+1]
            favg = np.empty_like(fchx)
            favg[:-1, :] = 0.5 * (fchx[:-1, :] + fchx[1:, :])
            favg[-1, :] = fchx[-1, :]
            with np.errstate(divide="ignore", invalid="ignore"):
                jpol = favg / self.Apol
            if self.Btot is None or self.Bpol is None:
                raise KeyError("need bb to project J_pol -> J_par, and "
                               "b2fstate has no usable fch_p")
            with np.errstate(divide="ignore", invalid="ignore"):
                jpar = jpol * self.Btot / self.Bpol
            src = "fchx * |B|/|B_pol|"

        jpar = np.where(np.isfinite(jpar), jpar, 0.0) * self.sign
        self.jpar = jpar
        self._log("    J_par source: {}".format(src))
        self._log("    max |J_par| = {:.4e} A/m^2 = {:.4f} MA/m^2"
                  .format(np.nanmax(np.abs(jpar)),
                          np.nanmax(np.abs(jpar)) * 1e-6))

    # ------------------------------------------------------------------
    def _assign_psin(self, gfile):
        """psi_n of every SOLPS cell centre, via the verified g-file map."""
        _, _, GQDSK = _import_readers()
        resolve_equilibrium = _import_eq_resolver()

        self._log("reading {}".format(gfile))
        g = GQDSK(gfile)

        # all name resolution + psi orientation verification lives here
        self.eq = resolve_equilibrium(g, verbose=self.verbose)
        self.psin = np.asarray(self.eq["psin_of_RZ"](self.Rc, self.Zc),
                               dtype=float)

        self._log("    SOLPS psi_n range: {:.4f} .. {:.4f}"
                  .format(np.nanmin(self.psin), np.nanmax(self.psin)))

    # ------------------------------------------------------------------
    def _locate_midplane(self):
        """Outer midplane poloidal index: max R on the innermost core ring."""
        zax = float(self.eq.get("zmaxis", 0.0)) if self.eq else 0.0
        iy = 1                                   # first non-guard radial ring
        R = self.Rc[:, iy]
        Z = self.Zc[:, iy]
        ok = np.isfinite(R) & (np.abs(Z - zax) < 0.5)
        if not ok.any():
            ok = np.isfinite(R)
        idx = np.arange(self.npol)[ok]
        self.jomp = int(idx[np.nanargmax(R[ok])])
        self._log("    outer midplane at ix = {}  (R = {:.4f}, Z = {:.4f})"
                  .format(self.jomp, self.Rc[self.jomp, iy],
                          self.Zc[self.jomp, iy]))

    # ------------------------------------------------------------------
    def omp_profile(self, psin_max=None, smooth=0.0):
        """1-D (psi_n, J_par) at the outer midplane, sorted and unique."""
        j = self.jomp
        psin = self.psin[j, 1:-1].copy()
        jpar = self.jpar[j, 1:-1].copy()

        good = np.isfinite(psin) & np.isfinite(jpar)
        if psin_max is not None:
            good &= psin <= psin_max
        psin, jpar = psin[good], jpar[good]

        order = np.argsort(psin)
        psin, jpar = psin[order], jpar[order]
        psin, uniq = np.unique(psin, return_index=True)
        jpar = jpar[uniq]

        if psin.size < 3:
            raise ValueError(
                "only {} usable SOLPS midplane points after trimming -- "
                "check psin_max".format(psin.size))

        if smooth > 0.0 and psin.size > 5:
            s = smooth * np.nanvar(jpar) * psin.size
            jpar = si.UnivariateSpline(psin, jpar, s=s)(psin)

        self._log("    OMP profile: {} points, psi_n {:.4f} .. {:.4f}"
                  .format(psin.size, psin[0], psin[-1]))
        self._log("    OMP max |J_par| = {:.4f} MA/m^2"
                  .format(np.nanmax(np.abs(jpar)) * 1e-6))
        return psin, jpar


# ===========================================================================
# BOUT++ grid side
# ===========================================================================
class BoutGrid(object):
    """Read a hypnotoad grid, resolve psi_n, hold Jpar0, write it back."""

    def __init__(self, path, geq=None, verbose=True):
        self.path = path
        self.verbose = verbose
        DataFile = _import_datafile()
        resolve_grid_psi = _import_grid_resolver()

        self._log("reading BOUT++ grid {}".format(path))
        with DataFile(path) as f:
            self.keys = list(f.list())
            self.data = {k: f.read(k) for k in self.keys}

        # ---- psi normalisation: psi_bdry vs psi_bndry vs *_gfile -------
        info = resolve_grid_psi(self.data, geq=geq, verbose=verbose)
        self.psi_axis = info["psi_axis"]
        self.psi_bdry = info["psi_bdry"]
        self.psinxy = np.asarray(info["psinxy"], dtype=float)
        self.ixseps1 = int(info["ixseps1"])

        self.psixy = np.asarray(self.data["psixy"], dtype=float)
        self.nx, self.ny = self.psixy.shape

        self.ixseps2 = self._int("ixseps2", self.nx)
        self.jyseps1_1 = self._int("jyseps1_1", -1)
        self.jyseps2_2 = self._int("jyseps2_2", self.ny - 1)

        jp = self.data.get("Jpar0")
        self.jpar_orig = (None if jp is None
                          else np.asarray(jp, dtype=float).copy())
        if self.jpar_orig is not None:
            self._log("    existing Jpar0: max |.| = {:.4f} MA/m^2"
                      .format(np.nanmax(np.abs(self.jpar_orig)) * 1e-6))
        else:
            self._log("    grid has no Jpar0 -- it will be created")
            self.jpar_orig = np.zeros_like(self.psixy)

        self._log("    nx = {}  ny = {}  ixseps1 = {}  ixseps2 = {}"
                  .format(self.nx, self.ny, self.ixseps1, self.ixseps2))

    def _log(self, *a):
        if self.verbose:
            print(*a)

    def _int(self, key, default):
        v = self.data.get(key)
        if v is None:
            return default
        try:
            return int(np.asarray(v).ravel()[0])
        except (TypeError, ValueError, IndexError):
            return default

    # ------------------------------------------------------------------
    def core_mask(self):
        """True on poloidal columns that belong to the closed-field region."""
        m = np.zeros((self.nx, self.ny), dtype=bool)
        j0 = max(self.jyseps1_1 + 1, 0)
        j1 = min(self.jyseps2_2 + 1, self.ny)
        m[:, j0:j1] = True
        return m

    def pfr_mask(self):
        """True in the private flux region: inboard of ixseps1, in the legs."""
        m = np.zeros((self.nx, self.ny), dtype=bool)
        ix = max(int(self.ixseps1), 0)
        j0 = max(self.jyseps1_1 + 1, 0)
        j1 = min(self.jyseps2_2 + 1, self.ny)
        m[:ix, :j0] = True
        m[:ix, j1:] = True
        return m

    def core_column(self):
        """A poloidal index guaranteed to be on a closed flux surface."""
        j0 = max(self.jyseps1_1 + 1, 0)
        j1 = min(self.jyseps2_2 + 1, self.ny)
        if j1 <= j0:
            return self.ny // 2
        return int((j0 + j1) // 2)

    # ------------------------------------------------------------------
    def write(self, jpar, outpath=None, apar=None, keep_original=True):
        DataFile = _import_datafile()
        target = outpath or self.path

        if outpath is None:
            bak = self.path + ".bak"
            if not os.path.exists(bak):
                shutil.copy2(self.path, bak)
                self._log("    backup written: {}".format(bak))
        else:
            if os.path.abspath(outpath) != os.path.abspath(self.path):
                shutil.copy2(self.path, outpath)

        with DataFile(target, write=True) as f:
            if keep_original and "Jpar0_hypnotoad" not in self.keys:
                f.write("Jpar0_hypnotoad", self.jpar_orig)
            f.write("Jpar0", np.asarray(jpar, dtype=np.float64))
            if apar is not None:
                f.write("Apar0", np.asarray(apar, dtype=np.float64))
        self._log("    wrote Jpar0 to {}".format(target))


# ===========================================================================
# separatrix repair  (C1)
# ===========================================================================
def _fit_core_edge(p, j, pm, match_band, min_pts=3):
    """Linear fit of ``j(p)`` on the core side of ``pm``.

    Returns ``(J0, slope)`` evaluated AT ``pm``, or ``(None, None)`` if the
    column has nothing usable.  Falls back to (nearest interior value, 0.0)
    when the match band is too sparse to fit, which reproduces the v4
    value-only anchor rather than failing.
    """
    good = np.isfinite(p) & np.isfinite(j)
    band = good & (p <= pm) & (p >= pm - match_band)
    n = int(band.sum())

    if n >= max(2, min_pts):
        # polyfit in (p - pm) so the intercept IS the value at pm
        coef = np.polyfit(p[band] - pm, j[band], 1)
        return float(coef[1]), float(coef[0])

    inner = good & (p <= pm)
    if not inner.any():
        return None, None
    k = int(np.argmin(pm - p[inner]))
    return float(j[inner][k]), 0.0


def repair_separatrix(psinxy, jpar, lam, psin_match=1.0,
                      match_band=0.015, blend_width=0.040,
                      trunc_frac=0.25, cap_frac=1.5, min_pts=3,
                      pfr_mask=None, verbose=False):
    """C1-continuous continuation of ``jpar`` outside ``psin_match``.

    Fit value AND slope on the core side, extrapolate linearly, smoothstep
    into the exponential tail over ``blend_width``.  See the module
    docstring for the construction and the measured gradient reduction.
    """
    p_all = np.asarray(psinxy, dtype=float)
    j_all = np.asarray(jpar, dtype=float)
    if p_all.shape != j_all.shape:
        raise ValueError("psinxy {} and jpar {} shape mismatch"
                         .format(p_all.shape, j_all.shape))

    jnew = j_all.copy()
    lam = float(lam)
    if not np.isfinite(lam) or lam <= 0.0:
        if verbose:
            print("    repair_separatrix: lam = {} is not usable, nothing done"
                  .format(lam))
        return jnew, 0

    bw = max(0.0, float(blend_width))
    pm = float(psin_match)
    ny = p_all.shape[1] if p_all.ndim == 2 else 1
    nfixed = 0

    for jy in range(ny):
        if p_all.ndim == 2:
            p, j = p_all[:, jy], j_all[:, jy]
        else:
            p, j = p_all, j_all

        mcol = None
        if pfr_mask is not None:
            mcol = pfr_mask[:, jy] if np.ndim(pfr_mask) == 2 else pfr_mask
            if np.all(mcol):
                continue

        out = np.isfinite(p) & np.isfinite(j) & (p > pm)
        if not out.any():
            continue

        J0, s = _fit_core_edge(p, j, pm, match_band, min_pts)
        if J0 is None or abs(J0) < 1e-30:
            continue

        # guard rail: only touch columns hypnotoad actually truncated
        peak_sol = float(np.nanmax(np.abs(j[out])))
        if peak_sol >= trunc_frac * abs(J0):
            continue

        d = p[out] - pm
        cap = cap_frac * abs(J0)
        j_lin = np.clip(J0 + s * d, -cap, cap)
        j_exp = J0 * np.exp(-d / lam)
        w = _smoothstep(d / bw) if bw > 0.0 else np.ones_like(d)
        jcol = (1.0 - w) * j_lin + w * j_exp

        idx = np.where(out)[0]
        if mcol is not None:
            keep = ~mcol[idx]
            idx, jcol = idx[keep], jcol[keep]
        if idx.size == 0:
            continue

        if p_all.ndim == 2:
            jnew[idx, jy] = jcol
        else:
            jnew[idx] = jcol
        nfixed += 1

    if verbose:
        print("    separatrix repair: C1, lam = {:.4f} psi_n, "
              "match_band = {:.3f}, blend_width = {:.3f}"
              .format(lam, match_band, bw))
        print("    repaired {} of {} poloidal columns".format(nfixed, ny))
        if 0.0 < bw < 2.0 * lam:
            print("    WARNING: blend_width {:.3f} < 2*lam {:.3f} -- a narrow "
                  "handoff can be WORSE than one-sided.  Try blend_width: "
                  "{:.3f}".format(bw, 2.0 * lam, 4.0 * lam))

    return jnew, nfixed


def lam_j_from_mm(grid, lam_mm, verbose=True):
    """Convert a SOL current decay length in mm to psi_n units.

    dpsi_n/dr is estimated from the psi_n step between the two columns
    straddling ixseps1 and the corresponding real-space distance.
    """
    ix = int(grid.ixseps1)
    if ix < 1 or ix >= grid.nx:
        return None
    Rxy = grid.data.get("Rxy")
    if Rxy is None:
        return None
    Rxy = np.asarray(Rxy, dtype=float)
    Zxy = grid.data.get("Zxy")
    Zxy = np.asarray(Zxy, dtype=float) if Zxy is not None else None

    dpsin = grid.psinxy[ix, :] - grid.psinxy[ix - 1, :]
    dR = Rxy[ix, :] - Rxy[ix - 1, :]
    dZ = (Zxy[ix, :] - Zxy[ix - 1, :]) if Zxy is not None else 0.0
    dr = np.sqrt(dR ** 2 + dZ ** 2)
    good = np.isfinite(dpsin) & np.isfinite(dr) & (dr > 0)
    if not good.any():
        return None
    grad = float(np.nanmedian(np.abs(dpsin[good] / dr[good])))      # 1/m
    lam = grad * (lam_mm * 1e-3)
    if verbose:
        print("    dpsi_n/dr at separatrix = {:.2f} 1/m  ->  lam_j({:.1f} mm)"
              " = {:.5f} in psi_n".format(grad, lam_mm, lam))
    return lam


# ===========================================================================
# mapping + assembly
# ===========================================================================
def map_profile(psin_src, prof_src, psin_xy, extrapolate="flat"):
    """PCHIP-interpolate a 1-D psi_n profile onto the 2-D mesh."""
    f = si.PchipInterpolator(psin_src, prof_src, extrapolate=False)
    out = np.asarray(f(psin_xy), dtype=float)

    lo, hi = psin_src[0], psin_src[-1]
    below, above = psin_xy < lo, psin_xy > hi
    if extrapolate == "flat":
        out[below] = prof_src[0]
        out[above] = prof_src[-1]
    elif extrapolate == "zero":
        out[below] = 0.0
        out[above] = 0.0
    elif extrapolate == "exp":
        def _efold(p0, p1, v0, v1):
            if v0 == 0 or v1 == 0 or np.sign(v0) != np.sign(v1):
                return None
            return (p1 - p0) / np.log(abs(v1 / v0))
        lh = _efold(psin_src[-2], psin_src[-1], prof_src[-2], prof_src[-1])
        out[below] = prof_src[0]
        out[above] = (prof_src[-1] * np.exp((psin_xy[above] - hi) / lh)
                      if lh else prof_src[-1])
    out[~np.isfinite(out)] = 0.0
    return out


def grad_report(psinxy, jpar, lo=0.95, hi=1.10):
    """max |dJ/dpsi_n| over a psi_n window -- the peeling-drive proxy."""
    p = np.asarray(psinxy, dtype=float)
    j = np.asarray(jpar, dtype=float)
    worst = 0.0
    ny = p.shape[1] if p.ndim == 2 else 1
    for jy in range(ny):
        pc = p[:, jy] if p.ndim == 2 else p
        jc = j[:, jy] if j.ndim == 2 else j
        g = np.isfinite(pc) & np.isfinite(jc)
        if g.sum() < 3:
            continue
        o = np.argsort(pc[g])
        ps, js = pc[g][o], jc[g][o]
        sel = (ps >= lo) & (ps <= hi)
        if sel.sum() < 3:
            continue
        d = np.abs(np.diff(js[sel]) / np.diff(ps[sel]))
        if d.size:
            worst = max(worst, float(np.nanmax(d)))
    return worst


def max_dj_dpsin(J, psin):
    """Global max |dJ/dpsi_n| -- dominated by the core, reported separately."""
    dJ = np.diff(J, axis=0)
    dp = np.diff(psin, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.abs(dJ / dp)
    g = g[np.isfinite(g)]
    return float(np.nanmax(g)) if g.size else np.nan


def build_jpar(solps, grid, cfg, verbose=True):
    """Assemble the new Jpar0 according to ``cfg.mode``."""
    psin_src, jpar_src = solps.omp_profile(psin_max=cfg.psin_max,
                                           smooth=cfg.smooth)
    psin_xy = grid.psinxy

    # ---- SOLPS mapped onto the mesh -----------------------------------
    j_solps = map_profile(psin_src, jpar_src, psin_xy, extrapolate=cfg.extrap)

    # ---- decay length --------------------------------------------------
    lam_j = cfg.lam_j
    if cfg.lam_j_mm is not None:
        lam2 = lam_j_from_mm(grid, cfg.lam_j_mm, verbose=verbose)
        if lam2 is not None:
            lam_j = lam2
        elif verbose:
            print("    lam_j_mm requested but the grid metric is unusable; "
                  "falling back to lam_j = {:.4f}".format(lam_j))

    # ---- hypnotoad, cliff repaired -------------------------------------
    jhyp = grid.jpar_orig.copy()
    pfr = grid.pfr_mask()
    if cfg.pfr_zero:
        jhyp[pfr] = 0.0
        if verbose:
            print("    PFR current zeroed ({} cells)".format(int(pfr.sum())))

    j_hyp, nfix = repair_separatrix(
        psin_xy, jhyp, lam_j,
        psin_match=cfg.psin_match,
        match_band=cfg.match_band,
        blend_width=cfg.blend_width,
        trunc_frac=cfg.trunc_frac,
        cap_frac=cfg.cap_frac,
        pfr_mask=(pfr if cfg.pfr_zero else None),
        verbose=verbose)

    # ---- blend weight --------------------------------------------------
    w = smoothstep(psin_xy, cfg.w_on, cfg.w_full)
    # never let SOLPS act outside its own radial domain, in either direction
    w = np.where((psin_xy > psin_src[-1]) | (psin_xy < psin_src[0]), 0.0, w)

    mode = cfg.mode
    if mode == "add":
        J = j_hyp + w * j_solps
    elif mode == "repair-only":
        J = j_hyp
    elif mode == "solps-blend":
        wc = smoothstep(psin_xy, psin_src[0], psin_src[0] + cfg.core_blend)
        J = (1.0 - wc) * grid.jpar_orig + wc * j_solps
    elif mode == "replace":
        J = j_solps
    else:
        raise ValueError("unknown mode '{}'".format(mode))

    J = np.where(np.isfinite(J), J, 0.0)

    # ---- diagnostics ---------------------------------------------------
    if verbose:
        core = grid.core_mask() & (psin_xy < 0.95)
        jcore = (np.nanmean(np.abs(grid.jpar_orig[core]))
                 if core.any() else np.nan)
        jsol = np.nanmax(np.abs(j_solps))

        _hdr("diagnostics")
        print("  core |J_par| (hypnotoad, psi_n < 0.95) = {:.4f} MA/m^2"
              .format(jcore * 1e-6))
        print("  peak |J_par| (SOLPS, mapped)           = {:.4f} MA/m^2"
              .format(jsol * 1e-6))
        if np.isfinite(jcore) and jcore > 0:
            ratio = jsol / jcore
            print("  ratio SOLPS / equilibrium              = {:.3f}"
                  .format(ratio))
            if ratio > 0.30:
                print("  *** ratio > 0.30 -- suspiciously large.  Re-run "
                      "check_fch_conventions.py before trusting this. ***")
            elif ratio < 0.005:
                print("  note: SOLPS contribution < 0.5% -- negligible; the "
                      "separatrix repair is doing all the work.")

        g0 = grad_report(psin_xy, grid.jpar_orig)
        g1 = grad_report(psin_xy, J)
        print("")
        print("  max |dJ/dpsi_n|, 0.95 < psi_n < 1.10   (the peeling drive)")
        print("    before = {:.4e} A/m^2".format(g0))
        print("    after  = {:.4e} A/m^2".format(g1))
        if np.isfinite(g0) and g0 > 0:
            print("    ratio  = {:.3f}x   ({:+.1f}%)"
                  .format(g1 / g0, 100.0 * (g1 - g0) / g0))
            if g1 > g0:
                print("  *** gradient INCREASED -- the repair introduced a "
                      "new discontinuity.")
                if 0.0 < cfg.blend_width < 2.0 * lam_j:
                    print("      blend_width {:.3f} < 2*lam {:.3f} is the "
                          "likely cause; try {:.3f}."
                          .format(cfg.blend_width, 2.0 * lam_j, 4.0 * lam_j))
                else:
                    print("      Try a larger lam_j_mm or blend_width.")
        print("")
        print("  global max |dJ/dpsi_n| (core-dominated, for reference)")
        print("    before = {:.4e}   after = {:.4e}"
              .format(max_dj_dpsin(grid.jpar_orig, psin_xy),
                      max_dj_dpsin(J, psin_xy)))

        # sign sanity
        if np.isfinite(jcore) and jcore > 0:
            s_hyp = np.sign(np.nanmean(grid.jpar_orig[core]))
            sepband = (psin_xy > 0.99) & (psin_xy < 1.01)
            if sepband.any():
                s_sol = np.sign(np.nanmean(j_solps[sepband]))
                if s_hyp != 0 and s_sol != 0 and s_hyp != s_sol:
                    print("")
                    print("  *** SIGN NOTE: SOLPS edge current is opposite in "
                          "sign to the equilibrium current.")
                    print("      Physically possible (thermoelectric current "
                          "can oppose Ip), but if you")
                    print("      expect alignment, set  sign: -1.0")

    return J, j_solps, j_hyp, w, (psin_src, jpar_src)


def build_apar(grid, J, verbose=True):
    """Diagnostic A_par from radial integration of mu0 * J_par."""
    try:
        Rxy = np.asarray(grid.data["Rxy"], dtype=float)
        Bpxy = np.asarray(grid.data["Bpxy"], dtype=float)
        dx = np.asarray(grid.data["dx"], dtype=float)
    except KeyError:
        if verbose:
            print("    Apar0 skipped: grid lacks Rxy/Bpxy/dx")
        return None
    with np.errstate(divide="ignore", invalid="ignore"):
        dr = dx / (Rxy * Bpxy)
    dr = np.where(np.isfinite(dr), dr, 0.0)
    A = -MU0 * np.cumsum(np.cumsum(J * dr, axis=0) * dr, axis=0)
    A -= A[0, :]
    return A


# ===========================================================================
# plotting
# ===========================================================================
def make_plot(grid, J, j_solps, j_hyp, w, omp, cfg, outpng=None):
    import matplotlib
    if outpng:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    psin_src, jpar_src = omp
    psin_xy = grid.psinxy
    jc = grid.core_column()
    pp = psin_xy[:, jc]
    mode = cfg.mode

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    xlim = (max(0.85, np.nanmin(psin_xy)), min(1.12, np.nanmax(psin_xy)))

    # ---- panel 1: profiles --------------------------------------------
    ax = axes[0]
    ax.plot(psin_src, jpar_src * 1e-6, "o-", ms=3, lw=1,
            label="SOLPS midplane", color="tab:blue")
    ax.plot(pp, grid.jpar_orig[:, jc] * 1e-6, "--", lw=1.2,
            label="original grid Jpar0", color="tab:green")
    ax.plot(pp, j_hyp[:, jc] * 1e-6, "-", lw=1.4,
            label="hypnotoad, repaired", color="tab:orange")
    ax.plot(pp, J[:, jc] * 1e-6, "-", lw=2.0, label="final Jpar0",
            color="k", alpha=0.8)
    ax.axvline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel(r"$\psi_N$")
    ax.set_ylabel(r"$J_\parallel$  [MA m$^{-2}$]")
    ax.set_title("radial profile")
    ax.legend(fontsize=7)
    ax.set_xlim(*xlim)

    # ---- panel 2: the quantity the repair exists to reduce -------------
    ax = axes[1]
    with np.errstate(divide="ignore", invalid="ignore"):
        g0 = np.abs(np.diff(grid.jpar_orig[:, jc]) / np.diff(pp))
        g1 = np.abs(np.diff(J[:, jc]) / np.diff(pp))
    pmid = 0.5 * (pp[1:] + pp[:-1])
    ax.semilogy(pmid, g0 * 1e-6, "--", lw=1.2, color="tab:green",
                label="before")
    ax.semilogy(pmid, g1 * 1e-6, "-", lw=1.8, color="k", label="after")
    ax.axvline(1.0, color="0.4", lw=0.8, ls=":")
    ax.set_xlabel(r"$\psi_N$")
    ax.set_ylabel(r"$|dJ_\parallel/d\psi_N|$  [MA m$^{-2}$]")
    ax.set_title("peeling drive")
    ax.set_xlim(*xlim)
    ax.legend(fontsize=7, loc="upper left")

    if mode in ("add", "solps-blend", "replace"):
        ax2 = ax.twinx()
        ax2.plot(pp, w[:, jc], "-", lw=1.2, color="tab:red", alpha=0.6)
        ax2.set_ylabel("SOLPS blend weight", color="tab:red", fontsize=8)
        ax2.set_ylim(-0.05, 1.05)
        ax2.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
    else:
        ax.text(0.97, 0.05, "mode = {}\nno SOLPS term".format(mode),
                ha="right", va="bottom", transform=ax.transAxes,
                fontsize=8, color="0.35")

    # ---- panel 3: the 2-D map -----------------------------------------
    ax = axes[2]
    v = np.nanmax(np.abs(J)) * 1e-6
    im = ax.pcolormesh(np.arange(grid.nx), np.arange(grid.ny), (J * 1e-6).T,
                       cmap="RdBu_r", vmin=-v, vmax=v, shading="auto")
    ax.axvline(grid.ixseps1, color="k", lw=0.8)
    ax.axhline(grid.jyseps1_1 + 0.5, color="k", lw=0.8, ls="--")
    ax.axhline(grid.jyseps2_2 + 0.5, color="k", lw=0.8, ls="--")
    ax.set_xlabel("radial index")
    ax.set_ylabel("poloidal index")
    ax.set_title(r"$J_{\parallel 0}$ on grid  [MA m$^{-2}$]")
    fig.colorbar(im, ax=ax)

    fig.tight_layout()
    if outpng:
        fig.savefig(outpng, dpi=130)
        print("    plot written: {}".format(outpng))
    else:
        plt.show()


# ===========================================================================
# CLI  --  config path only
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Write SOLPS-informed Jpar0 into a BOUT++ elm-6f grid.  "
                    "All options are read from the '{}' section of the "
                    "config file.".format(SECTION),
        epilog="There are deliberately no physics flags.  Edit the config.")
    p.add_argument("config", nargs="?", help="SIBIL YAML config")
    p.add_argument("--section", default=SECTION,
                   help="read a different section name (for keeping several "
                        "named presets in one file).  Default: {}"
                        .format(SECTION))
    p.add_argument("--print-template", action="store_true",
                   help="print a fully commented config section and exit")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.print_template:
        print_template()
        return 0

    if not args.config:
        print("usage: sibil_solps_jpar_to_bout.py SIBIL_config.yaml\n"
              "       sibil_solps_jpar_to_bout.py --print-template",
              file=sys.stderr)
        return 2

    cfg = load_settings(args.config, section=args.section)
    verbose = not cfg.quiet

    missing = [k for k in ("b2fstate", "b2fgmtry", "gfile", "bgrid")
               if not getattr(cfg, k)]
    if missing:
        print("missing required path(s): {}\n"
              "set them under '{}:' in {} (or anywhere in the document)"
              .format(", ".join(missing), args.section, args.config),
              file=sys.stderr)
        return 2

    _hdr("SOLPS -> BOUT++  Jpar0   (mode = {})".format(cfg.mode))
    print("  config : {}  [{}]".format(args.config, args.section))
    if verbose:
        cfg.echo()

    solps = SolpsJpar(cfg.b2fstate, cfg.b2fgmtry, cfg.gfile,
                      geofile=cfg.geofile,
                      convention=cfg.fch_convention,
                      sign=cfg.sign, verbose=verbose)

    print("")
    grid = BoutGrid(cfg.bgrid, geq=solps.eq, verbose=verbose)

    J, j_solps, j_hyp, w, omp = build_jpar(solps, grid, cfg, verbose=verbose)

    apar = build_apar(grid, J, verbose=verbose) if cfg.write_apar else None

    if cfg.plot:
        make_plot(grid, J, j_solps, j_hyp, w, omp, cfg, outpng=cfg.png)

    _hdr("output")
    if cfg.dry_run:
        print("  dry_run: true -- grid file NOT modified")
        print("  final Jpar0: max |.| = {:.4f} MA/m^2,  mean |.| = {:.4f}"
              .format(np.nanmax(np.abs(J)) * 1e-6,
                      np.nanmean(np.abs(J)) * 1e-6))
        print("  set  dry_run: false  in the config to write it.")
    else:
        grid.write(J, outpath=cfg.out, apar=apar)
        print("  original hypnotoad current preserved as 'Jpar0_hypnotoad'")
    return 0


if __name__ == "__main__":
    sys.exit(main())
