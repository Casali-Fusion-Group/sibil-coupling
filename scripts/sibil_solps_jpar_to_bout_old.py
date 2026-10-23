#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sibil_solps_jpar_to_bout.py   (v7)

Put a smooth, field-line-consistent equilibrium parallel current ``Jpar0``
into an existing BOUT++ (elm-6f) grid file, combining

  * hypnotoad's g-file current (Ohmic + bootstrap) -- correct in the core,
    truncated to zero outside the LCFS, and
  * the SOLPS-ITER edge current closure from ``b2fstate`` (thermoelectric /
    sheath), which contains no equilibrium current.

Nothing else in the grid file is touched.

    python sibil_solps_jpar_to_bout.py SIBIL_config.yaml

---------------------------------------------------------------------------
v7 -- why v6 had to go
---------------------------------------------------------------------------
v6 repaired the separatrix cliff column by column, anchoring each column's
SOL extension on that column's own core-side current.  Two consequences:

1. Leg columns (j <= jyseps1_1, j > jyseps2_2) have no core side -- their
   ix < ixseps1 cells are private flux, zeroed by pfr_zero -- so their SOL
   was never extended.  Along every SOL field line J0 therefore dropped from
   ~0.33 MA/m^2 to ~0 in one cell at the X-point rows (j 7->8, 59->60).
   elm-6f drives with Grad_par(J0); that step is a delta function on exactly
   the cells where every crash so far has concentrated.
2. The value+slope fit at psi_n = 1 with a cubic (C1) smoothstep leaves a
   corner in d2J/dpsi at the separatrix (radial kink at ix ~ 129) and at
   the end of the blend.

v7 builds the edge current as  J0 = c(psi) * B  outside the core:

    core cells   J = (1 - S(psi; a, b)) * J_hyp  +  S(psi; a, b) * c_edge * B
    SOL cells    J = c_edge * B * E(psi)     E: 1 at psi=1 -> 0 at 1 + sol_taper
    PF cells     J = c_edge * B * P(psi)     P: 1 at psi=1 -> 0 at 1 - pf_taper
    (+ SOLPS, SOL only)  Ws(psi) * c_solps(psi) * B

c_edge is ONE number: the flux-surface average of J_hyp/B at psi_n = b.
Because J/B is then a flux function outside the handoff, Grad_par(J0/B) = 0
along every SOL and PF field line, including through the X-point rows, and
the current tapers to exactly zero at a finite distance on both sides.
S, E, P and Ws are quintic smoothsteps (zero 1st AND 2nd derivative at both
ends), so J0 is C2 in psi everywhere -- across ix ~ 128-131 in particular.

The hypnotoad current is always read from ``Jpar0_hypnotoad`` when the grid
has it (i.e. after any earlier run of this script), so re-running is
idempotent and never builds on a previously modified Jpar0.

What this does NOT fix
----------------------
Hypnotoad's interior current has no pedestal-resolved bootstrap peak; the
taper widths are free parameters (quote them, scan them).  A quantitatively
correct pedestal current needs a re-solved equilibrium with bootstrap.

Modes
-----
  add          smooth hypnotoad extension + SOLPS edge current (SOL only)
  repair-only  smooth hypnotoad extension only

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
    "mode": ("add", ("add", "repair-only"),
             "add          smooth hypnotoad extension + SOLPS edge current "
             "(SOL cells only)\n"
             "repair-only  smooth hypnotoad extension only"),
    "sign": (1.0, "float",
             "Flip the SOLPS parallel direction (+1 or -1)."),
    "fch_convention": ("normal", ("normal", "swapped"),
                       "Which fch component is poloidal.  Only consulted if "
                       "b2fstate has no fch_p."),

    # ---- core-side handoff (C2) -------------------------------------------
    "psin_blend_start": (0.970, "float",
                         "psi_n where the core current starts handing off "
                         "from hypnotoad's J to the flux-function form "
                         "c_edge * B.  Keep it inside the clean interior."),
    "psin_blend_end": (0.995, "float",
                       "psi_n where the handoff is complete.  c_edge is the "
                       "flux-surface average of J_hyp/B on this surface, so "
                       "it must sit inside hypnotoad's cliff (the last few "
                       "cells before psi_n = 1 are smeared by it)."),

    # ---- tapers to zero (C2) ----------------------------------------------
    "sol_taper": (0.030, "float",
                  "psi_n width over which the SOL current goes from c_edge*B "
                  "at the separatrix to exactly zero.  Superseded by "
                  "sol_taper_mm."),
    "sol_taper_mm": (None, "float_or_none",
                     "SOL taper width in mm at the outboard midplane, "
                     "converted with the grid's dpsi_n/dr at the separatrix."),
    "pf_taper": (0.010, "float",
                 "psi_n width over which the private-flux current goes from "
                 "c_edge*B at the separatrix to zero.  Superseded by "
                 "pf_taper_mm."),
    "pf_taper_mm": (None, "float_or_none",
                    "PF taper width in mm."),
    "leg_taper": (False, "bool",
                  "Additionally taper the SOL and PF current to zero along "
                  "the divertor legs, from the X-point row to the target. "
                  "Off by default: it re-introduces Grad_par(J0/B) in the "
                  "legs (only 4-8 cells long here)."),

    # ---- SOLPS window (SOL only) ------------------------------------------
    "solps_on": (0.005, "float",
                 "psi_n width outside the separatrix over which the SOLPS "
                 "current is switched on (quintic).  Zero inside psi_n = 1."),
    "solps_off": (0.010, "float",
                  "psi_n width over which the SOLPS current is switched off "
                  "before the edge of the SOLPS radial domain (no step at the "
                  "domain edge)."),

    # ---- SOLPS profile extraction -----------------------------------------
    "psin_max": (None, "float_or_none",
                 "Trim SOLPS cells above this psi_n.  Divertor-leg cells can "
                 "carry psi_n values that collide with core ones and corrupt "
                 "the monotonic sort; 1.05-1.10 is usually right."),
    "smooth": (5.0e-3, "float",
               "Smoothing factor of the quintic spline fitted to the SOLPS "
               "OMP profile (relative to its variance).  The spline is what "
               "makes the SOLPS term C2; 0 interpolates the raw points."),

    # ---- output -----------------------------------------------------------
    "out": (None, "str",
            "Write to this file instead of modifying bgrid in place.  When "
            "writing in place a .bak is made on first run."),
    "write_apar": (False, "bool",
                   "Also write a diagnostic Apar0 from radial integration of "
                   "mu0 * J_par.  Plotting aid only."),
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


#: v6 keys that no longer do anything.  Accepted (so old configs still run)
#: but reported, never silently honoured.
_DEPRECATED = {
    "w_on": "SOLPS now acts in the SOL only; see solps_on",
    "w_full": "SOLPS now acts in the SOL only; see solps_on",
    "core_blend": "mode solps-blend was removed",
    "lam_j": "replaced by sol_taper (compact C2 taper, not an exponential)",
    "lam_j_mm": "replaced by sol_taper_mm",
    "match_band": "v7 does not fit a slope; see psin_blend_start/end",
    "blend_width": "replaced by psin_blend_start/end and sol_taper",
    "psin_match": "the handoff is now psin_blend_start..psin_blend_end",
    "pfr_zero": "PF current now tapers smoothly to zero; see pf_taper",
    "trunc_frac": "v7 always starts from the original hypnotoad current",
    "cap_frac": "no linear extension in v7",
    "extrap": "the SOLPS term is switched off inside its domain (solps_off)",
    "plot": "plots live in sibil_solps_jpar_to_bout_diagnostic.py",
    "png": "plots live in sibil_solps_jpar_to_bout_diagnostic.py",
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

    # v6 keys: accepted, reported, ignored
    for k in [k for k in given if k in _DEPRECATED and k not in _SPEC]:
        print("note: '{}' is a v6 key and is ignored -- {}"
              .format(k, _DEPRECATED[k]), file=sys.stderr)
        given.pop(k)

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
    if not values["psin_blend_start"] < values["psin_blend_end"] < 1.0:
        raise SystemExit("need psin_blend_start < psin_blend_end < 1.0, got "
                         "{} and {}".format(values["psin_blend_start"],
                                            values["psin_blend_end"]))

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
        self.jyseps2_1 = self._int("jyseps2_1", self.ny // 2)
        self.jyseps1_2 = self._int("jyseps1_2", self.ny // 2)
        self.jyseps2_2 = self._int("jyseps2_2", self.ny - 1)

        # the ORIGINAL hypnotoad current: Jpar0_hypnotoad if an earlier run
        # of this script already replaced Jpar0, otherwise Jpar0 itself
        self.jpar_current = self.data.get("Jpar0")
        src = "Jpar0_hypnotoad" if "Jpar0_hypnotoad" in self.data else "Jpar0"
        jp = self.data.get(src)
        if jp is not None:
            self.jpar_orig = np.asarray(jp, dtype=float).copy()
            self._log("    hypnotoad current from '{}': max |.| = {:.4f} MA/m^2"
                      .format(src, np.nanmax(np.abs(self.jpar_orig)) * 1e-6))
        else:
            self._log("    grid has no Jpar0 -- it will be created")
            self.jpar_orig = np.zeros_like(self.psixy)
        if self.jpar_current is None:
            self.jpar_current = self.jpar_orig.copy()
        self.jpar_current = np.asarray(self.jpar_current, dtype=float)

        self.Bxy = np.asarray(self.data["Bxy"], dtype=float)

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
    def leg_columns(self):
        """True for poloidal indices in a divertor leg (incl. upper legs)."""
        j = np.arange(self.ny)
        leg = (j <= self.jyseps1_1) | (j > self.jyseps2_2)
        if self.jyseps1_2 > self.jyseps2_1:                  # double null
            leg |= (j > self.jyseps2_1) & (j <= self.jyseps1_2)
        return leg

    def regions(self):
        """Boolean (nx, ny) masks: core, sol, pf."""
        ix = np.arange(self.nx)[:, None]
        leg = self.leg_columns()[None, :]
        inside = ix < self.ixseps1
        core = inside & ~leg
        pf = inside & leg
        sol = ~inside
        return np.broadcast_to(core, (self.nx, self.ny)).copy(), \
            np.broadcast_to(sol, (self.nx, self.ny)).copy(), \
            np.broadcast_to(pf, (self.nx, self.ny)).copy()

    def core_column(self):
        """A poloidal index on a closed flux surface (mid-core)."""
        core_j = np.flatnonzero(~self.leg_columns())
        return int(core_j[len(core_j) // 2]) if core_j.size else self.ny // 2

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
# C2 building blocks
# ===========================================================================
def smoothstep5(t):
    """Quintic smoothstep 6t^5 - 15t^4 + 10t^3 on a pre-normalised argument.

    0 for t <= 0, 1 for t >= 1, and zero first AND second derivative at both
    ends -- the property that makes every join in v7 C2.
    """
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def psin_per_metre(grid):
    """|dpsi_n/dr| at the separatrix [1/m], median over poloidal columns."""
    ix = int(grid.ixseps1)
    if ix < 1 or ix >= grid.nx:
        return None
    Rxy = grid.data.get("Rxy")
    if Rxy is None:
        return None
    Rxy = np.asarray(Rxy, dtype=float)
    Zxy = grid.data.get("Zxy")
    Zxy = np.asarray(Zxy, dtype=float) if Zxy is not None else np.zeros_like(Rxy)
    dpsin = grid.psinxy[ix, :] - grid.psinxy[ix - 1, :]
    dr = np.hypot(Rxy[ix, :] - Rxy[ix - 1, :], Zxy[ix, :] - Zxy[ix - 1, :])
    good = np.isfinite(dpsin) & np.isfinite(dr) & (dr > 0)
    if not good.any():
        return None
    return float(np.nanmedian(np.abs(dpsin[good] / dr[good])))


def resolve_widths(grid, cfg, verbose=True):
    """SOL / PF taper widths in psi_n, converting the *_mm options if set."""
    sol, pf = float(cfg.sol_taper), float(cfg.pf_taper)
    if cfg.sol_taper_mm is not None or cfg.pf_taper_mm is not None:
        g = psin_per_metre(grid)
        if g is None:
            if verbose:
                print("    *_mm widths requested but the grid metric is "
                      "unusable; using the psi_n values")
        else:
            if cfg.sol_taper_mm is not None:
                sol = g * cfg.sol_taper_mm * 1e-3
            if cfg.pf_taper_mm is not None:
                pf = g * cfg.pf_taper_mm * 1e-3
            if verbose:
                print("    dpsi_n/dr at separatrix = {:.2f} 1/m".format(g))
    if verbose:
        print("    SOL taper = {:.4f} psi_n,  PF taper = {:.4f} psi_n"
              .format(sol, pf))
    return sol, pf


def _surface_weights(grid):
    """Flux-surface-average weights per cell: dl/Bp = hthe*dy/Bp (or 1)."""
    hthe = grid.data.get("hthe")
    Bp = grid.data.get("Bpxy")
    dy = grid.data.get("dy")
    if hthe is None or Bp is None:
        return np.ones_like(grid.psinxy)
    w = np.asarray(hthe, float) / np.maximum(np.abs(np.asarray(Bp, float)), 1e-6)
    if dy is not None:
        w = w * np.asarray(dy, float)
    return np.where(np.isfinite(w) & (w > 0), w, 0.0)


def edge_coefficient(grid, jhyp, psin_ref, core, verbose=True):
    """c_edge = <J_hyp / B> on the surface psi_n = psin_ref (core columns).

    Each core column is interpolated to psin_ref; the poloidal values are
    averaged with dl/Bp weights.  One number: J0/B outside the handoff.
    """
    jb = jhyp / grid.Bxy
    wts = _surface_weights(grid)
    vals, ws = [], []
    for jy in np.flatnonzero(core.any(axis=0)):
        col = core[:, jy]
        p = grid.psinxy[col, jy]
        f = jb[col, jy]
        o = np.argsort(p)
        p, f = p[o], f[o]
        if not (p[0] <= psin_ref <= p[-1]):
            continue
        vals.append(np.interp(psin_ref, p, f))
        ws.append(np.interp(psin_ref, p, wts[col, jy][o]))
    if not vals:
        raise ValueError("no core column spans psi_n = {}".format(psin_ref))
    vals, ws = np.asarray(vals), np.asarray(ws)
    c = float(np.sum(vals * ws) / np.sum(ws)) if ws.sum() > 0 else float(np.mean(vals))
    if verbose:
        spread = np.std(vals) / (abs(c) + 1e-300)
        print("    c_edge = <J_hyp/B> at psi_n = {:.4f}: {:.4e} A/m^2/T "
              "(poloidal spread {:.1%} over {} columns)"
              .format(psin_ref, c, spread, len(vals)))
    return c


def leg_profile(grid):
    """Per-column factor 1 at the X-point row -> 0 at the target (quintic)."""
    L = np.ones(grid.ny)
    j11, j22 = grid.jyseps1_1, grid.jyseps2_2
    if j11 >= 0:                                         # inner leg: j = 0..j11
        n = j11 + 1
        L[:n] = smoothstep5((np.arange(n) + 0.5) / n)
    if j22 < grid.ny - 1:                                # outer leg: j22+1..ny-1
        n = grid.ny - 1 - j22
        L[j22 + 1:] = smoothstep5((n - 0.5 - np.arange(n)) / n)
    return L


def solps_flux_function(psin_src, jpar_src, grid, smooth, verbose=True):
    """c_solps(psi_n) = J_SOLPS / B at the outboard midplane, as a C4 spline.

    Dividing by the grid's own OMP field puts the SOLPS current in the same
    field-line-constant form as the hypnotoad extension.
    """
    core = ~grid.leg_columns()
    Rxy = grid.data.get("Rxy")
    if Rxy is None:
        jomp = grid.core_column()
    else:
        R = np.asarray(Rxy, float)[min(grid.ixseps1, grid.nx - 1), :]
        jomp = int(np.flatnonzero(core)[np.argmax(R[core])])
    p_omp = grid.psinxy[:, jomp]
    o = np.argsort(p_omp)
    B_src = np.interp(psin_src, p_omp[o], grid.Bxy[o, jomp])
    f = jpar_src / B_src
    k = 5 if psin_src.size > 6 else min(3, psin_src.size - 1)
    s = smooth * np.nanvar(f) * psin_src.size if smooth > 0 else 0.0
    spl = si.UnivariateSpline(psin_src, f, k=k, s=s)
    if verbose:
        print("    SOLPS: quintic spline of J/B at OMP column j = {} "
              "(B = {:.3f}..{:.3f} T)".format(jomp, B_src.min(), B_src.max()))
    return spl, (psin_src[0], psin_src[-1])


# ===========================================================================
# assembly
# ===========================================================================
def build_jpar(solps, grid, cfg, verbose=True):
    """Assemble the v7 Jpar0.  Returns J and the parts for reporting."""
    psin = grid.psinxy
    B = grid.Bxy
    core, sol, pf = grid.regions()
    a, b = cfg.psin_blend_start, cfg.psin_blend_end
    sol_w, pf_w = resolve_widths(grid, cfg, verbose)

    jhyp = grid.jpar_orig
    c_edge = edge_coefficient(grid, jhyp, b, core, verbose)

    # radial shape functions (all C2)
    S = smoothstep5((psin - a) / (b - a))                    # core handoff
    E = 1.0 - smoothstep5((psin - 1.0) / sol_w)              # SOL taper
    P = 1.0 - smoothstep5((1.0 - psin) / pf_w)               # PF taper
    L = leg_profile(grid)[None, :] if cfg.leg_taper else 1.0
    legs = grid.leg_columns()[None, :]

    J = np.zeros_like(psin)
    J[core] = ((1.0 - S) * jhyp + S * c_edge * B)[core]
    J_sol = c_edge * B * E * np.where(legs, L, 1.0)
    J_pf = c_edge * B * P * np.where(legs, L, 1.0)
    J[sol] = J_sol[sol]
    J[pf] = J_pf[pf]
    j_ext = J.copy()

    # SOLPS edge closure, SOL only, switched on/off with quintic windows
    j_solps = np.zeros_like(psin)
    omp = None
    if cfg.mode == "add" and solps is not None:
        psin_src, jpar_src = solps.omp_profile(psin_max=cfg.psin_max, smooth=0.0)
        spl, (lo, hi) = solps_flux_function(psin_src, jpar_src, grid,
                                            cfg.smooth, verbose)
        on = smoothstep5((psin - 1.0) / cfg.solps_on)
        off = 1.0 - smoothstep5((psin - (hi - cfg.solps_off)) / cfg.solps_off)
        inside = (psin >= lo) & (psin <= hi)
        Ws = np.where(sol & inside, on * off, 0.0)
        cs = np.where(inside, spl(np.clip(psin, lo, hi)), 0.0)
        j_solps = Ws * cs * B * np.where(legs, L, 1.0)
        J = J + j_solps
        omp = (psin_src, jpar_src)

    J = np.where(np.isfinite(J), J, 0.0)
    if verbose:
        report(grid, grid.jpar_current, J, label_old="grid Jpar0 now")
    return J, j_ext, j_solps, c_edge, omp


# ===========================================================================
# text diagnostics
# ===========================================================================
def radial_kink(grid, J, i0=None, i1=None):
    """max K = |J[i+1]-2J[i]+J[i-1]| / (|J[i+1]-J[i-1]|/2 + 1e-3 max|J|)
    over ix in [i0, i1], all columns.  ~0 smooth, O(1) corner."""
    i0 = grid.ixseps1 - 4 if i0 is None else i0
    i1 = grid.ixseps1 + 3 if i1 is None else i1
    Jp, J0, Jm = J[i0 + 1:i1 + 2], J[i0:i1 + 1], J[i0 - 1:i1]
    k = np.abs(Jp - 2 * J0 + Jm) / (0.5 * np.abs(Jp - Jm) + 1e-3 * np.nanmax(np.abs(J)) + 1e-300)
    i, j = np.unravel_index(np.nanargmax(k), k.shape)
    return float(k[i, j]), i0 + i, j


def xpoint_jump(grid, J):
    """Largest one-cell jump of J/B along y on SOL rows, relative to max|J/B|,
    and where it is.  This is the Grad_par(J0/B) delta-function proxy."""
    jb = J / grid.Bxy
    s = jb[grid.ixseps1:, :]
    d = np.abs(np.diff(s, axis=1))
    i, j = np.unravel_index(np.nanargmax(d), d.shape)
    return float(d[i, j] / (np.nanmax(np.abs(jb)) + 1e-300)), grid.ixseps1 + i, j


def report(grid, J_old, J_new, label_old="before"):
    _hdr("diagnostics")
    for lab, J in ((label_old, J_old), ("v7", J_new)):
        k, ki, kj = radial_kink(grid, J)
        x, xi, xj = xpoint_jump(grid, J)
        print("  {:<15} radial kink K (ix {}..{}) max {:6.2f} at ix={} j={};   "
              "max one-cell |d(J/B)|/max along y in SOL {:6.3f} at ix={} j={}->{}"
              .format(lab, grid.ixseps1 - 4, grid.ixseps1 + 3, k, ki, kj,
                      x, xi, xj, xj + 1))
    print("  max |J0|: {} {:.4f} MA/m^2   v7 {:.4f} MA/m^2"
          .format(label_old, np.nanmax(np.abs(J_old)) * 1e-6,
                  np.nanmax(np.abs(J_new)) * 1e-6))


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
# CLI  --  config path only
# ===========================================================================
def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Write a smooth SOLPS-informed Jpar0 into a BOUT++ elm-6f "
                    "grid.  All options are read from the '{}' section of the "
                    "config file.".format(SECTION),
        epilog="There are deliberately no physics flags.  Edit the config.")
    p.add_argument("config", nargs="?", help="SIBIL YAML config")
    p.add_argument("--section", default=SECTION,
                   help="read a different section name.  Default: {}"
                        .format(SECTION))
    p.add_argument("--print-template", action="store_true",
                   help="print a fully commented config section and exit")
    return p.parse_args(argv)


def setup(argv=None):
    """Parse, load settings and inputs; shared with the diagnostic script."""
    args = parse_args(argv)
    if args.print_template:
        print_template()
        return None
    if not args.config:
        print("usage: {} SIBIL_config.yaml\n       {} --print-template"
              .format(os.path.basename(sys.argv[0]), os.path.basename(sys.argv[0])),
              file=sys.stderr)
        return None
    cfg = load_settings(args.config, section=args.section)
    verbose = not cfg.quiet
    need = ["bgrid", "gfile"] + (["b2fstate", "b2fgmtry"] if cfg.mode == "add" else [])
    missing = [k for k in need if not getattr(cfg, k)]
    if missing:
        print("missing required path(s): {}\nset them under '{}:' in {}"
              .format(", ".join(missing), args.section, args.config),
              file=sys.stderr)
        return None
    _hdr("SOLPS -> BOUT++  Jpar0  v7   (mode = {})".format(cfg.mode))
    print("  config : {}  [{}]".format(args.config, args.section))
    if verbose:
        cfg.echo()
    solps = None
    geq = None
    if cfg.mode == "add":
        solps = SolpsJpar(cfg.b2fstate, cfg.b2fgmtry, cfg.gfile,
                          geofile=cfg.geofile, convention=cfg.fch_convention,
                          sign=cfg.sign, verbose=verbose)
        geq = solps.eq
    else:
        _, _, GQDSK = _import_readers()
        geq = _import_eq_resolver()(GQDSK(cfg.gfile), verbose=verbose)
    print("")
    grid = BoutGrid(cfg.bgrid, geq=geq, verbose=verbose)
    return cfg, solps, grid, verbose


def finish(cfg, grid, J, verbose):
    apar = build_apar(grid, J, verbose=verbose) if cfg.write_apar else None
    _hdr("output")
    if cfg.dry_run:
        print("  dry_run: true -- grid file NOT modified")
        print("  set  dry_run: false  in the config to write it.")
    else:
        grid.write(J, outpath=cfg.out, apar=apar)
        print("  original hypnotoad current preserved as 'Jpar0_hypnotoad'")


def main(argv=None):
    got = setup(argv)
    if got is None:
        return 0 if "--print-template" in (argv or sys.argv) else 2
    cfg, solps, grid, verbose = got
    J, j_ext, j_solps, c_edge, omp = build_jpar(solps, grid, cfg, verbose=verbose)
    finish(cfg, grid, J, verbose)
    return 0


if __name__ == "__main__":
    sys.exit(main())
