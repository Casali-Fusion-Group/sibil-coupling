#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
from boututils.datafile import DataFile
from .readers.GQDSK_Reader import GQDSK
from .grids.bout import Bout
from .grids.solps import Solps
from .grids.mapping import Mapping
from .plotting.grid_summary import plot_matrix

class BASICS_Mapper:
    
    def __init__(self, config):
        '''
        Wrapper object to streamline the mapping process for the SOLPS-ITER - 
        BOUT++ coupling. This object first extracts the data from relevant 
        SOLPS-ITER and BOUT++ files. It then creates objects for the SOLPS 
        grid, BOUT++ grid, and the intermediate mapping grid. The coupling
        then takes place by updating the BOUT++ profiles with a radial, then
        poloidal spline fit, facilitated by the mapping grid.

        Parameters
        ----------
        config : DICT
            A CONFIGURATION FILE THAT INCLUDES THE LOCATIONS AND FILENAMES 
            OF ALL DOCUMENTS REQUIRED FOR THE MAPPING. THE CURRENT ALLOWABLE
            INPUTS INCLUDE:
                'gfile_loc': '/gfile/location/g######.####',
                'geofile_loc': '/geofile/location/geofile.geo',
                'bgrid_loc': '/bout/grid/file/location/bout.nc',
                'b2fstate_loc': '/b2fstate/file/location/b2fstate',
                'b2fplasmf_loc': '/b2fplasmf/file/location/b2fplasmf',
                'b2fgmtry_loc': '/b2fgmtry/file/location/b2fgmtry',
                'softmax_scales': scale_for_softmax_smooth

        Returns
        -------
        None.

        '''
        
        # Save the config file for later use outside the mapping
        self.config = config
        
        # Initialize the SOLPS, Map, and BOUT++ grids
        self.bout = Bout("BOUT++", self.config)
        self.solps = Solps("SOLPS-ITER", self.config)
        self.solps.renormalize_flux_tube_lengths(self.bout)
        self.map = Mapping("Map", self.bout, self.solps, self.config)
        self.map.renormalize_flux_tube_lengths(self.bout)
        
        
        # plot_matrix(self.bout.dl,"BOUT++ dl")
        # plot_matrix(self.map.dl,"Map dl")
        # plot_matrix(self.solps.dl,"SOLPS dl")
        
        # plot_matrix(self.bout.dl2d,"BOUT++ dl2d")
        # plot_matrix(self.map.dl2d,"Map dl2d")
        # plot_matrix(self.solps.dl2d,"SOLPS dl2d")
        
        # Perform the mapping by calculating flux tube length, performing
        # radial SOLPS splines, then poloidal mapping splines to define the
        # BOUT++ grid.
        self.bout.calc_profiles_from_map(self.map)
        self.bout.smooth_profiles()
        self.bout.ne = self.bout.ni
        
        # Update the bout grid with the new information.
        self.bout.update_gridfile()
        
        # Quickly summarize the major facts of the grid files for debugging.
        # self.solps.summarize()
        # self.map.summarize()
        self.bout.summarize()

    def get_profiles(self):
        '''
        Return a dictionary with the major values of each grid involved in the
        mapping.

        Returns
        -------
        dict
            DESCRIPTION.

        '''
        return {
            'sni': self.solps.ni,
            'sne': self.solps.ne,
            'sTi': self.solps.Ti,
            'sTe': self.solps.Te,
            'bni': self.bout.ni,
            'bne': self.bout.ne,
            'bTi': self.bout.Ti,
            'bTe': self.bout.Te,
            'mni': self.map.ni,
            'mne': self.map.ne,
            'mTi': self.map.Ti,
            'mTe': self.map.Te,
        }

