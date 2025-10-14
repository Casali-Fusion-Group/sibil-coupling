#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thr Aug 22 09:43:43 2024

@author: ttaczak
"""

import netCDF4 as nc
import numpy as np

class BOUT_data():
    
    data_dict = None
    nx=None
    ny=None
    ixseps1=None
    ixseps2=None
    jyseps1_1=None
    jyseps1_2=None
    jyseps2_1=None
    jyseps2_2=None
    ny_inner=None
    dx=None
    dy=None
    ShiftAngle=None
    zShift=None
    pol_angle=None
    ShiftTorsion=None
    Rxy=None
    Zxy=None
    Bpxy=None
    Btxy=None
    Bxy=None
    hthe=None
    sinty=None
    psixy=None
    yup_xsplit=None
    ydown_xsplit=None
    yup_xin=None
    yup_xout=None
    ydown_xin=None
    ydown_xout=None
    nrad=None
    npol=None
    pressure=None
    Jpar0=None
    Ni0=None
    Te0=None
    Ti0=None
    Ni_x=None
    Te_x=None
    Ti_x=None
    bmag=None
    rmag=None
    bxcvx=None
    bxcvy=None
    bxcvz=None
    bpsign=None
    psi_axis=None
    psi_bndry=None
    jomp=None
    psinxy=None
        
    def __init__(self,fname):
        try:
            open(fname, 'r')
        except FileNotFoundError:
            print(f"File '{fname}' not found.")
            
        self.data_dict = self.extractData(fname)
        #if not len(self.data_dict.keys()) == 47:
        #    raise Warning(f"Warning, len dict={str(len(self.data_dict.keys()))}, not all data stored in object...")
        self.set_all_vars()

    def set_all_vars(self):
        '''
        A function that loops through all of the local variables that are not 
        the dictionary of all variables itself, and sets all of their values 
        to the equivalent variable in the dictionary (requires the same name)
        '''
        for var_name, var_value in self.__class__.__dict__.items():
            if not var_name.startswith('__') and not var_name == "data_dict" and var_name in self.data_dict.keys() and not callable(var_value) :  # Ignore special methods and attributes
                if hasattr(self, var_name):
                    current_value = getattr(self, var_name)
                    # print(f"Modifying {var_name}: {current_value} -> {self.data_dict[var_name]} New Value")
                    setattr(self, var_name, self.data_dict[var_name])
                else:
                    # If the instance doesn't have the class variable, add it
                    # print(f"Setting {var_name} for the instance")
                    setattr(self, var_name, self.data_dict[var_name])
        
        if self.psi_bndry is None:
            self.psi_bndry = self.data_dict["psi_bdry"]
            
        self.jomp = np.max(self.Rxy[0,:])
        self.psinxy = (self.psixy-self.psi_axis)/(self.psi_bndry-self.psi_axis);
                
                
            
    def extractData(self,file):
        # Open the netCDF file
        dataset = nc.Dataset(file, mode='r')

        # Dictionary to store data
        data_dict = {}

        # Loop through each variable in the file and store it in the dictionary
        for var_name in dataset.variables:
            # Extract the variable data as a NumPy array
            data_dict[var_name] = np.array(dataset.variables[var_name][:])

        # Close the dataset
        dataset.close()

        return data_dict

if __name__ == '__main__':
    test_BOUT_output = BOUT_data("Data/d3d_184016.3200_psi092108_nx260ny64.nc")
    working_BOUT_output1 = BOUT_data("Data/cbm18_dens6-0.5BS.grid.nc")
    working_BOUT_output2 = BOUT_data("Data/cbm18_dens8.grid_nx68ny64.nc")
        
