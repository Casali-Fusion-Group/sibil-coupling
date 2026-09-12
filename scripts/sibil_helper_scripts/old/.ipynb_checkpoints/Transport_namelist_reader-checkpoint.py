#!/usr/bin/env python
# coding: utf-8

# Fortran TRANSPORT namelist editor.
# 
# Usage (import this module into your own script):
# 
#     from tdata_editor import parse_tdata2, set_tdata2_block, write_updated_namelist
# 
#     # 1. Parse all current tdata(2,...) values into a dict
#     tdata2_dict = parse_tdata2("transport.nml")
# 
#     # 2. Replace one block's values using a 1-D array.
#     #    The block is identified by (k, l) — the 3rd and 4th indices of ndata(1,1,k,l) / tdata(...,k,l).
#     #    The array length must match ndata(1,k,l) for that block.
#     new_weights = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, ...]   # one value per j
#     set_tdata2_block(tdata2_dict, k=1, l=1, values=new_weights)
# 
#     # 3. Write the updated file (input is never modified)
#     write_updated_namelist("b2.transport.inputfile", "b2.transport.inputfile.new", tdata2_dict)
# 
# Dictionary key format:  (2, j, k, l)  ->  float

# In[10]:


import re
from pathlib import Path


# ## Regex patterns

# In[41]:


# Matches any tdata(i, j, k, l) = value — tolerates arbitrary whitespace.
_TDATA_RE = re.compile(
    r"(tdata\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*)"
    r"([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)",
    re.IGNORECASE,
)

# Matches ndata(1, 1, k, l) = N  — used to discover block sizes and identifiers.
_NDATA_RE = re.compile(
    r"ndata\(\s*1\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(\d+)",
    re.IGNORECASE,
)


# # Parse

# In[74]:


def parse_tdata2(filepath: str) -> dict:
    """
    Read a TRANSPORT namelist file and return a dictionary of every
    tdata(2, j, k, l) entry found.

    Returns
    -------
    dict : { (2, j, k, l): float, ... }
    """
    tdata2 = {}
    text = Path(filepath).read_text()

    for m in _TDATA_RE.finditer(text):
        i = int(m.group(2))
        if i == 2:
            j, k, l = int(m.group(3)), int(m.group(4)), int(m.group(5))
            tdata2[(2, j, k, l)] = float(m.group(6))

    return tdata2


def parse_block_sizes(filepath: str) -> dict:
    """
    Read a TRANSPORT namelist file and return the declared size of every block.

    Returns
    -------
    dict : { (k, l): N, ... }
        where N = ndata(1, k, l)
    """
    sizes = {}
    text = Path(filepath).read_text()
    for m in _NDATA_RE.finditer(text):
        k, l, n = int(m.group(1)), int(m.group(2)), int(m.group(3))
        sizes[(k, l)] = n
    return sizes


# ## Block helper

# In[33]:


def set_tdata2_block(tdata2_dict: dict, k: int, l: int, values) -> None:
    """
    Replace all tdata(2, j, k, l) entries for a given (k, l) block using a
    1-D array of new values.

    Parameters
    ----------
    tdata2_dict : dict (modified in-place)
        The dictionary returned by parse_tdata2() (or a copy of it).
    k, l        : int
        The 2rd and 3rd indices identifying the block, e.g. k=1, l=1
        corresponds to the block guarded by  ndata(1,1,1)=N.
    values      : sequence of float
        1-D array/list of length N.  values[0] -> j=1, values[1] -> j=2, …
        Length must equal the number of j entries in that block.

    Raises
    ------
    ValueError  if len(values) doesn't match the number of existing j entries
                for (k, l) in tdata2_dict.
    """
    # Find existing j indices for this (k, l) block
    existing_js = sorted(key[1] for key in tdata2_dict if key[2] == k and key[3] == l)

    if not existing_js:
        raise ValueError(f"No tdata(2, j, {k}, {l}) entries found in the dictionary.")

    if len(values) != len(existing_js):
        raise ValueError(
            f"Block (k={k}, l={l}) has {len(existing_js)} entries "
            f"but {len(values)} values were supplied."
        )

    for j, val in zip(existing_js, values):
        tdata2_dict[(2, j, k, l)] = float(val)


# ## Write

# In[15]:


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
    


# ## Number formatter

# In[17]:


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


# # Inline example — edit and run this file directly to try it out

# In[34]:


INPUT_FILE  = "/global/cfs/projectdirs/m4491/btaczak/solps-runs/HvsD_D_lower_noC/Matched_Prof_D_noC_new/b2.transport.inputfile"   # <-- change this
OUTPUT_FILE = "/global/cfs/projectdirs/m4491/btaczak/solps-runs/HvsD_D_lower_noC/Matched_Prof_D_noC_new/b2.transport.inputfile.new"


# ------------------------------------------------------------------
# Step 1: parse
# ------------------------------------------------------------------
tdata2 = parse_tdata2(INPUT_FILE)


# In[65]:


# Matches ndata(1, j, k, l) = N  — used to discover block sizes and identifiers.
_NDATA_RE = re.compile(
    r"ndata\(\s*1\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*=\s*(\d+)",
    re.IGNORECASE,
)


# In[75]:


# print("Parsed tdata(2,...) entries:")
# for key in sorted(tdata2):
#     print(f"  tdata{key} = {tdata2[key]}")

block_sizes = parse_block_sizes(INPUT_FILE)
print("\nBlock sizes (ndata(1,k,l)):")
for (k, l), n in sorted(block_sizes.items()):
    print(f"  (k={k}, l={l})  ->  N={n}")


# **Particle density-driven diffusivity**
# kind coeff.eq.1 ⇒ dna0  
# **Ion thermal anomalous diffusivity**
# kind coeff.eq.3 ⇒ hcib  
# **Electron thermal anomalous diffusivity**
# kind coeff.eq.4 ⇒ hce0

# In[78]:


block_sizes[(1,1)]


# In[79]:


# ------------------------------------------------------------------
# Step 2: supply new values for each block as a 1-D array
#
# Replace the lists below with your actual arrays.
# Each list must have exactly N entries where N = ndata(1,1,k,l).
# ------------------------------------------------------------------

# Example: block (k=1, l=1) — all weights set to 1.5
n_block_1_1 = block_sizes[(1, 1)]
new_weights_1_1 = [1.5] * n_block_1_1          # <-- replace with your array

set_tdata2_block(tdata2, k=1, l=1, values=new_weights_1_1)

# Example: block (k=1, l=3)
n_block_1_3 = block_sizes[(1, 3)]
new_weights_1_3 = [2.0] * n_block_1_3          # <-- replace with your array

set_tdata2_block(tdata2, k=1, l=3, values=new_weights_1_3)

# ------------------------------------------------------------------
# Step 3: write
# ------------------------------------------------------------------
write_updated_namelist(INPUT_FILE, OUTPUT_FILE, tdata2)


# In[2]:


stem = "/global/homes/b/btaczak/SIBIL_profiles/"
FILES = [
    f"{stem}Profile_1_coupling_scaled.txt",
]


# ndata
