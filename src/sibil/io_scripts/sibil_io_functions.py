#!/usr/bin/env python3
"""
sibil_io_functions.py

Helper functions for the scripts handling the IO of the SIBIL coupling.

"""
import os
import sys
import yaml
import numpy as np
from pathlib import Path
import re
import numpy as np
import netCDF4 as nc
import yaml
from scipy.interpolate import interp1d

def set_tdata2_block(tdata2_dict: dict,block_sizes,alpha: float, k: int, l: int, turbulent_profile, percentage: float=1.0) -> None:
    """
    Replace all tdata(2, j, k, l) entries for a given (k, l) block using a
    1-D array of new values.

    Parameters
    ----------
    tdata2_dict : dict (modified in-place)
        The dictionary returned by parse_tdata2() (or a copy of it).
    prev_tdata2_dict : dict (modified in-place)
        The dictionary returned by parse_tdata2(prev_inputfile) (or a copy of it).
    k, l        : int
        The 2rd and 3rd indices identifying the block, e.g. k=1, l=1
        corresponds to the block guarded by  ndata(1,1,1)=N.
    values      : sequence of float
        1-D array/list of length N.  values[0] -> j=1, values[1] -> j=2, …
        Length must equal the number of j entries in that block.
    percentage  : float
        The percentage of the total cross field transport provided by the
        turbulence

    Raises
    ------
    ValueError  if len(values) doesn't match the number of existing j entries
                for (k, l) in tdata2_dict.
    """
    num_js = block_sizes[(1, 1)]
    js = np.arange(1, num_js + 1)

    # Create the array
    orig_profile = np.zeros(num_js)
    prev_profile = np.zeros(num_js)
    for i, j in enumerate(js):             # array slot i <-> grid point j
        orig_profile[i] = tdata2_dict[(2, j, k, l)]
    
    # Normalize the turbulent components of the flux
    total_flux_magnitude = np.linalg.norm(orig_profile)
    turbulent_flux_magnitude = np.linalg.norm(turbulent_profile)
    assert turbulent_profile.shape == orig_profile.shape   # catches the zip-truncation class of bug

    turbulence_contributions = (
        percentage * (turbulent_profile/turbulent_flux_magnitude) * total_flux_magnitude
    )
    
    print("Original profile: ", orig_profile)
    print("New profile: ", orig_profile + turbulence_contributions)

    for j, contribution in zip(js, turbulence_contributions, strict=True):
        prev_profile = tdata2_dict[(2, j, k, l)]
        new_profile  = tdata2_dict[(2, j, k, l)] + contribution
        tdata2_dict[(2, j, k, l)] = (1-alpha)*(prev_profile) + (alpha)*(new_profile)

def vec_summary(vec):
    print("Min: ", np.min(vec))
    print("Max: ", np.max(vec))
    print("ArgMin: ", np.argmin(vec))
    print("ArgMax: ", np.argmax(vec))

def get_midplane_distances_from_psin(psin_values, gridfile_path):
    """
    Map arbitrary psin values (e.g. loaded from a text file) to signed
    distance from the separatrix at the midplane, using the psin<->distance
    relationship derived from a BOUT++ grid file.

    Parameters
    ----------
    psin_values : array_like
        Normalised psi values to convert, assumed to use the SAME
        normalisation convention as the grid (psin=1 at separatrix).
    gridfile_path : str
        Path to the BOUT++ NetCDF grid file used to build the mapping.

    Returns
    -------
    dist : np.ndarray
        Signed distance from the separatrix [m], same shape as psin_values.
    """
    psin_values = np.atleast_1d(np.asarray(psin_values, dtype=float))

    # Build the psin -> distance mapping from the grid
    psin_mid, dist_mid = get_all_midplane_distances(gridfile_path)

    sort_idx      = np.argsort(psin_mid)
    psin_sorted   = psin_mid[sort_idx]
    dist_sorted   = dist_mid[sort_idx]
    _, unique_idx = np.unique(psin_sorted, return_index=True)
    psin_sorted   = psin_sorted[unique_idx]
    dist_sorted   = dist_sorted[unique_idx]

    dist_interp = interp1d(psin_sorted, dist_sorted, kind='cubic',
                            bounds_error=False, fill_value='extrapolate')

    return dist_interp(psin_values)

def interpolate_dataset(x_data, y_data, x_query, extrapolate=False):
    x_data  = np.asarray(x_data,  dtype=float)
    y_data  = np.asarray(y_data,  dtype=float)
    x_query = np.atleast_1d(np.asarray(x_query, dtype=float))

    if x_data.shape != y_data.shape:
        raise ValueError(f"x_data and y_data must have the same length "
                         f"({len(x_data)} vs {len(y_data)}).")

    if len(x_data) < 2:
        raise ValueError("At least 2 data points are required for interpolation.")

    # Sort by x in case data is unordered
    sort_idx = np.argsort(x_data)
    x_data   = x_data[sort_idx]
    y_data   = y_data[sort_idx]

    fill = 'extrapolate' if extrapolate else np.nan
    interpolator = interp1d(x_data, y_data, kind='linear',
                            bounds_error=False, fill_value=fill)

    y_query = interpolator(x_query)

    # Warn about any out-of-range queries when not extrapolating
    if not extrapolate:
        out_of_range = (x_query < x_data[0]) | (x_query > x_data[-1])
        if np.any(out_of_range):
            print(f"Warning: {out_of_range.sum()} query point(s) are outside "
                  f"the data range [{x_data[0]}, {x_data[-1]}] and will be NaN.")

    return float(y_query[0]) if y_query.size == 1 else y_query

def extract_tdata_first_index(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    
    # Match tdata(1, j, *, *) entries only
    pattern = r'tdata\s*\(\s*1\s*,\s*(\d+)\s*,\s*\d+\s*,\s*\d+\s*\)\s*=\s*([+-]?\d*\.?\d+[eE]?[+-]?\d*)'
    
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if not matches:
        print("No tdata(1,...) entries found.")
        return []
    
    # Sort by j-index, deduplicate, extract values
    seen = {}
    for j_idx, val in matches:
        j = int(j_idx)
        if j not in seen:
            seen[j] = float(val)
    
    values = [seen[j] for j in sorted(seen.keys())]
    return np.array(values)
    
def get_bout_midplane_distances(gridfile_path):
    """
    Compute the distance from the separatrix at the outboard midplane
    for every radial flux surface in a BOUT++ grid file.

    Parameters
    ----------
    gridfile_path : str
        Path to the BOUT++ NetCDF grid file.

    Returns
    -------
    psin_mid : np.ndarray, shape (nx,)
        Normalised psi at the midplane for each radial index.
    dist_mid : np.ndarray, shape (nx,)
        Signed distance from the separatrix at the midplane [metres].
        Negative = inside separatrix (core), Positive = outside (SOL).
    """
    with nc.Dataset(gridfile_path, 'r') as ds:
        def load(var):
            return np.array(ds.variables[var][:])

        Rxy       = load('Rxy')       # shape (nx, ny)
        Zxy       = load('Zxy')       # shape (nx, ny)
        psixy     = load('psixy')     # shape (nx, ny)
        psi_axis  = float(ds.variables['psi_axis'][:])
        psi_bndry = float(ds.variables['psi_bdry'][:])

    # --- Normalise psi ---
    psinxy = (psixy - psi_axis) / (psi_bndry - psi_axis)  # (nx, ny)

    nx, ny = Rxy.shape

    # --- Find outboard midplane y-index ---
    outboard_x = np.argmax(Rxy[:, ny // 2])
    Z_along_y  = Zxy[outboard_x, :]
    mid_y      = int(np.argmin(np.abs(Z_along_y)))

    # --- Extract radial profiles at the midplane ---
    R_mid    = Rxy[:, mid_y]       # shape (nx,)
    psin_mid = psinxy[:, mid_y]    # shape (nx,)

    # --- Separatrix R at midplane by interpolation ---
    sort_idx     = np.argsort(psin_mid)
    psin_sorted  = psin_mid[sort_idx]
    R_sorted     = R_mid[sort_idx]

    _, unique_idx = np.unique(psin_sorted, return_index=True)
    psin_sorted   = psin_sorted[unique_idx]
    R_sorted      = R_sorted[unique_idx]

    R_sep_interp = interp1d(psin_sorted, R_sorted, kind='cubic',
                            bounds_error=False, fill_value='extrapolate')
    R_sep = float(R_sep_interp(1.0))

    # --- Signed distance for every flux surface ---
    dist_mid = R_mid - R_sep   # shape (nx,)

    return psin_mid, dist_mid

def extract_tdata_first_index(filepath):
    with open(filepath, 'r') as f:
        text = f.read()
    
    # Match tdata(1, j, *, *) entries only
    pattern = r'tdata\s*\(\s*1\s*,\s*(\d+)\s*,\s*\d+\s*,\s*\d+\s*\)\s*=\s*([+-]?\d*\.?\d+[eE]?[+-]?\d*)'
    
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    if not matches:
        print("No tdata(1,...) entries found.")
        return []
    
    # Sort by j-index, deduplicate, extract values
    seen = {}
    for j_idx, val in matches:
        j = int(j_idx)
        if j not in seen:
            seen[j] = float(val)
    
    values = [seen[j] for j in sorted(seen.keys())]
    return np.array(values)
    
def _format_value(value: float) -> str:
    """
    Format a float to match the style typically seen in TRANSPORT namelists.
      1.5, 0.2, 2.5  -> compact decimal   e.g.  '1.5'
      -3.49e-2       -> Fortran sci-not    e.g.  '-3.49E-02'
    """
    if value == 0.0:
        return "0.0"
    abs_val = abs(value)
    # Compact decimal for numbers that look like simple decimals
    if 0.01 <= abs_val < 1e5:
        candidate = f"{value:g}"
        try:
            if float(candidate) == value:
                return candidate
        except ValueError:
            pass
    # Fortran-style scientific notation
    formatted = f"{value:.2E}"
    # Guarantee two-digit exponent (Python on some platforms gives one digit)
    formatted = re.sub(r"E([+-])(\d)$", r"E\g<1>0\2", formatted)
    return formatted

def write_updated_namelist(input_filepath: str,
                           output_filepath: str,
                           new_values: dict) -> None:
    """
    Write a new namelist file identical to *input_filepath* except that every
    tdata(2, j, k, l) = <old> is replaced with the corresponding value from
    *new_values*.  Everything else (whitespace, tabs, tdata(1,...), ndata, ...)
    is preserved character-for-character.

    Parameters
    ----------
    input_filepath  : path to the original namelist file
    output_filepath : path to write the updated file
    new_values      : dict  { (2, j, k, l): float, ... }
                      Keys absent from the dict are left unchanged.
    """
    text = Path(input_filepath).read_text()

    _TDATA_RE = re.compile(
        r"(tdata\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*)"
        r"([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )

    def replacer(m):
        i = int(m.group(2))
        if i != 2:
            return m.group(0)               # leave tdata(1,...) untouched

        j, k, l = int(m.group(3)), int(m.group(4)), int(m.group(5))
        key = (2, j, k, l)
        if key not in new_values:
            return m.group(0)               # not in dict → unchanged

        formatted = _format_value(new_values[key])
        return m.group(1) + formatted       # group(1) = "tdata(...) = "

    updated_text = _TDATA_RE.sub(replacer, text)
    Path(output_filepath).write_text(updated_text)
    print(f"Updated file written to: {output_filepath}")

def parse_block_sizes(filepath: str) -> dict:
    """
    Read a TRANSPORT namelist file and return the declared size of every block.

    Returns
    -------
    dict : { (k, l): N, ... }
        where N = ndata(1, k, l)
    """
    # Matches ndata(1, k, l) = N  — used to discover block sizes and identifiers.
    _NDATA_RE = re.compile(
        r"ndata\(\s*1\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(\d+)",
        re.IGNORECASE,
    )
    sizes = {}
    text = Path(filepath).read_text()
    for m in _NDATA_RE.finditer(text):
        k, l, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sizes[(k, l)] = n
    return sizes
    
def parse_tdata2(filepath: str) -> dict:
    """
    Read a TRANSPORT namelist file and return a dictionary of every
    tdata(2, j, k, l) entry found.

    Returns
    -------
    dict : { (2, j, k, l): float, ... }
    """
    # Matches any tdata(i, j, k, l) = value — tolerates arbitrary whitespace.
    _TDATA_RE = re.compile(
        r"(tdata\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*)"
        r"([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)",
        re.IGNORECASE,
    )
    
    tdata2 = {}
    text = Path(filepath).read_text()

    for m in _TDATA_RE.finditer(text):
        i = int(m.group(2))
        if i == 2:
            j, k, l = int(m.group(3)), int(m.group(4)), int(m.group(5))
            tdata2[(2, j, k, l)] = float(m.group(6))

    return tdata2

def build_config(raw):
    base_path = raw["base_path"]
    files = raw["files"]

    config = {
        "gfile_loc": os.path.join(base_path, files["gfile"]),
        "geofile_loc": os.path.join(base_path, files["geofile"]),
        "bgrid_loc": os.path.join(base_path, files["bgrid"]),
        "b2fstate_loc": os.path.join(base_path, files["b2fstate"]),
        "b2fplasmf_loc": os.path.join(base_path, files["b2fplasmf"]),
        "b2fgmtry_loc": os.path.join(base_path, files["b2fgmtry"]),
        "softmax_scales": {
            k: tuple(v) for k, v in raw["softmax_scales"].items()
        },
    }
    return config

def load_config(path):
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return raw

def sibil_config_summary(config):
    print("=" * 70)
    print("SIBIL Midplane Mapper — Configuration Summary")
    print("=" * 70)

    path_keys = [
        ("gfile", "G-file"),
        ("geofile", "Geometry file"),
        ("bgrid", "B-grid file"),
        ("b2fstate", "b2fstate"),
        ("b2fplasmf", "b2fplasmf"),
        ("b2fgmtry", "b2fgmtry"),
    ]

    for key, label in path_keys:
        path = config["files"][key]
        exists = os.path.exists(path)
        flag = "OK" if exists else "MISSING"
        print(f"  {label:<15}: {path}  [{flag}]")

    print("\n  Softmax scales (max, min):")
    for var, (vmax, vmin) in config["softmax_scales"].items():
        print(f"    {var:<4}: max={vmax:g}, min={vmin:g}")

    print("=" * 70)

def confirm(prompt="Proceed with this configuration? [y/N]: "):
    answer = input(prompt).strip().lower()
    return answer == "y"