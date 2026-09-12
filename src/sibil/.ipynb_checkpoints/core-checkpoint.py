#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as si
import matplotlib.pyplot as plt
from pathlib import Path
import os
from boututils.datafile import DataFile
from sibil_io_functions import load_config
from .readers.GQDSK_Reader import GQDSK
from .grids.bout import Bout
from .grids.solps import Solps
from .grids.mapping import Mapping
from .plotting.grid_summary import plot_matrix
from .readers.B2fstate_Reader import B2fstate
from .readers.B2fplasmf_Reader import B2fplasmf

DEFAULT_CONFIG_PATH = "SIBIL_config.yaml"

class SIBIL_Mapper:
    
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

        
        plot_matrix(self.bout.dl,"BOUT++ dl")
        plot_matrix(self.map.dl,"Map dl")
        plot_matrix(self.solps.dl,"SOLPS dl")
        
        plot_matrix(self.bout.dl2d,"BOUT++ dl2d")
        plot_matrix(self.map.dl2d,"Map dl2d")
        plot_matrix(self.solps.dl2d,"SOLPS dl2d")
        
        # Perform the mapping by calculating flux tube length, performing
        # radial SOLPS splines, then poloidal mapping splines to define the
        # BOUT++ grid.
        self.bout.calc_profiles_from_map(self.map)
        self.bout.smooth_profiles()
        self.bout.ne = self.bout.ni
        
        # Update the bout grid with the new information.
        self.bout.update_gridfile()
        
        # Quickly summarize the major facts of the grid files for debugging.
        self.solps.summarize()
        self.map.summarize()
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

class SIBIL_Midplane_Mapper:
    
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
                'files''gfile'     : '/gfile/location/g######.####',
                'files''geofile'   : '/geofile/location/geofile.geo',
                'files''bgrid'     : '/bout/grid/file/location/bout.nc',
                'files''b2fstate'  : '/b2fstate/file/location/b2fstate',
                'files''b2fplasmf' : '/b2fplasmf/file/location/b2fplasmf',
                'files''b2fgmtry'  : '/b2fgmtry/file/location/b2fgmtry',
                'softmax_scales'   : scale_for_softmax_smooth

        Returns
        -------
        None.

        '''
        
        # Save the config file for later use outside the mapping
        print("Loading the config file")
        self.config = config
        
        # Initialize the SOLPS, Map, and BOUT++ grids
        print("Initializing the mapping objects")
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
        print("Performing mapping calculations")
        self.bout.calc_profiles_from_map(self.map)
        self.bout.smooth_profiles()
        self.bout.ne = self.bout.ni
            
        # TEMP: SET ALL PROFILES EQUAL TO THE MIDPLANE ONLY
        print("Setting all poloidal profiles to midplane profile")
        bout_ni_midplane_profile = self.bout.ni[:,self.bout.jomp]
        bout_ne_midplane_profile = self.bout.ne[:,self.bout.jomp]
        bout_Ti_midplane_profile = self.bout.Ti[:,self.bout.jomp]
        bout_Te_midplane_profile = self.bout.Te[:,self.bout.jomp]

        
        bout_psin = self.bout.psinxy[:,self.bout.jomp]
        bout_ne = self.bout.ne[:,self.bout.jomp]
        bout_Te = self.bout.Te[:,self.bout.jomp]
        
        solps_psin = self.solps.psinxy[:,self.solps.jomp]
        solps_ne = self.solps.ne[:,self.solps.jomp]
        solps_Te = self.solps.Te[:,self.solps.jomp]
        
        fig, axs = plt.subplots(2,1)
        axs[0].scatter(bout_psin, bout_ne,label="BOUT++")
        axs[0].scatter(solps_psin, solps_ne,label="SOLPS")
        axs[0].set_ylabel(r"$n_e$")
        
        axs[1].scatter(bout_psin, bout_Te,label="BOUT++")
        axs[1].scatter(solps_psin, solps_Te,label="SOLPS")
        axs[1].set_xlabel(r"$\psi_n$")
        axs[1].set_ylabel(r"$T_e$")

        outpath = os.path.join(os.getcwd(), "s2b_midplanes.png")
        fig.savefig(outpath, dpi=150)
        print(f"Saved plot to {outpath}")
        

        alpha = self.config["solps_to_bout_pass"]["alpha"]
        if alpha != 1.0 and self.config["iteration"] != 0:
            print("Relaxation parameter detected!")
            print("Finding previous iteration profiles.")
            prev_dir = Path(f"{config["base_path"]}/{int(config["iteration"])-1}_iteration")
            prev_config_path = os.path.join(prev_dir, DEFAULT_CONFIG_PATH)
            print(f"prev_config: {prev_config_path}")
            prev_config = load_config(prev_config_path)
            prev_bout = Bout("prev_BOUT++", prev_config)
            prev_bout_profiles = prev_bout.get_grid_profiles()
            prev_jomp = prev_bout.jomp
            prev_bout_ni_midplane = prev_bout_profiles["Niexp"][:,prev_jomp]
            prev_bout_ne_midplane = prev_bout_profiles["Neexp"][:,prev_jomp]
            prev_bout_Ti_midplane = prev_bout_profiles["Tiexp"][:,prev_jomp]
            prev_bout_Te_midplane = prev_bout_profiles["Teexp"][:,prev_jomp]
            # prev_fname_b2fstate = os.path.join(prev_dir,"b2fstate")
            # prev_fname_b2fplasmf = os.path.join(prev_dir,"b2fplasmf")
            # print(prev_fname_b2fstate)
            # prev_b2fstate = B2fstate(prev_fname_b2fstate)
            # prev_b2fplasmf = B2fplasmf(prev_fname_b2fplasmf, prev_b2fstate.nx, prev_b2fstate.ny, prev_b2fstate.ns)
            # prev_ne = prev_b2fplasmf.ne.T # not needed
            # prev_Te = prev_b2fplasmf.te.T
            # prev_ni = prev_b2fplasmf.ni[:,:,1].T
            # prev_Ti = prev_b2fplasmf.ti.T

            print(rf"Applying relaxation scheme for $\alpha={alpha}$")
            bout_ni_midplane_profile = (1-alpha)*prev_bout_ni_midplane + alpha*bout_ni_midplane_profile
            bout_ne_midplane_profile = (1-alpha)*prev_bout_ne_midplane + alpha*bout_ne_midplane_profile
            bout_Ti_midplane_profile = (1-alpha)*prev_bout_Ti_midplane + alpha*bout_Ti_midplane_profile
            bout_Te_midplane_profile = (1-alpha)*prev_bout_Te_midplane + alpha*bout_Te_midplane_profile

        
        for j in np.arange(self.bout.ny):
            self.bout.ni[:,j] = bout_ni_midplane_profile
            self.bout.ne[:,j] = bout_ne_midplane_profile
            self.bout.Ti[:,j] = bout_Ti_midplane_profile
            self.bout.Te[:,j] = bout_Te_midplane_profile
            
        # Update the bout grid with the new information.
        print("Updating the bout++ grid file")
        self.bout.update_gridfile()
        
        # Quickly summarize the major facts of the grid files for debugging.
        self.solps.summarize()
        self.map.summarize()
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

