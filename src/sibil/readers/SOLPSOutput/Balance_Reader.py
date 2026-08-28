#!/fusion/projects/codes/pyped/python3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 16:12:38 2024

@author: jakeb
"""

import netCDF4 as nc
import numpy as np

#dset.variables.keys() 

class BalanceNC:
    def __init__(self,fname):
        self.load_netcdf_to_dict(fname)

    def load_netcdf_to_dict(self,file_path):
        # Open the netCDF file
        dataset = nc.Dataset(file_path, mode='r')

        #Dictionary to store data

        #Loop through each variable in the file and store it in the dictionary
        for var_name in dataset.variables:
            # Extract the variable data as a NumPy array
            data = np.array(dataset.variables[var_name][:])
            if data.ndim == 1:
                setattr(self,var_name,data)
            elif data.ndim == 2:
                data = np.transpose(data,axes=(1,0))
                setattr(self,var_name,data)
            elif data.ndim == 3:
                data = np.transpose(data,axes=(2,1,0))
                setattr(self,var_name,data)
            elif data.ndim == 4:
                data = np.transpose(data,axes=(3,2,1,0))
                setattr(self,var_name,data)
            else:
                setattr(self,var_name,data)
        nx,ny,ns = np.shape(self.na)    
        crx = np.reshape(self.crx,((nx*ny),1,4),order='F')
        cry = np.reshape(self.cry,((nx*ny),1,4),order='F')
        ll = np.concatenate((crx[:,:,0],cry[:,:,0]), axis = 1)
        lr = np.concatenate((crx[:,:,1],cry[:,:,1]), axis = 1)
        ul = np.concatenate((crx[:,:,2],cry[:,:,2]), axis = 1)
        ur = np.concatenate((crx[:,:,3],cry[:,:,3]), axis = 1)
        mesh_verts = [[tuple(lr[i]), tuple(ll[i]), tuple(ul[i]), tuple(ur[i])]
                 for i in range(nx*ny)]
        setattr(self, "nx",nx)
        setattr(self, "ny",ny)
        setattr(self, "ns",ns)
        setattr(self, "lower_right_vertex", lr)
        setattr(self, "lower_left_vertex", ll)
        setattr(self, "upper_right_vertex", ul)
        setattr(self, "upper_left_vertex", ur)
        setattr(self, "mesh_verticies", mesh_verts)


if __name__ == "__main__":
    test_balance = BalanceNC("balance.nc")
