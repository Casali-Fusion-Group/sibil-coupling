# -*- coding: utf-8 -*-

try:
    from importlib.metadata import PackageNotFoundError, version
except (ModuleNotFoundError, ImportError):
    from importlib_metadata import PackageNotFoundError, version
try:
    __version__ = version("basics_package")
except PackageNotFoundError:
    from setuptools_scm import get_version

    __version__ = get_version(root="../..", relative_to=__file__)

__all__ = ["__version__"]

from .core import BASICS_Mapper
from .plotting.grid_summary import (
    plot_poloidal_profiles,
    plot_radial_profiles,
    plot_matrix
    )
from .plotting.grid_comparison import (
    compare_poloidal_profiles,
    compare_radial_profiles
    )
from .grids.bout import Bout
from .grids.solps import Solps
from .grids.mapping import Mapping
print("SPINS Succsessfully Imported")