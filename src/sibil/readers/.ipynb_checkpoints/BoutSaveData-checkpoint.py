#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 11:33:56 2025

@author: ttaczak
"""
from boutdata.collect import collect
from boutdata.data import BoutData
from boututils.datafile import DataFile
import warnings
import os
from pathlib import Path
import numpy as np

def find_file(target):
    return next(Path('.').rglob(target), None)

def bout_save_data(data_dir,grid_fname, t_ind):
    numpy_dir = data_dir + "/numpy/"
    if os.path.isdir(data_dir):
        os.makedirs(numpy_dir, exist_ok=True)
    else:
        print(f"'{data_dir}' does not exist. Skipping creation of '{numpy_dir}'.")
        raise ValueError(f"Data dir {data_dir} not found!")
    
    d = BoutData(path=data_dir)
    
    with DataFile(grid_fname) as g:
        Rxy = g["Rxy"]
        Zxy = g["Zxy"]
        Bpxy = g["Bpxy"]
        Btxy = g["Btxy"]
        
        
    required_outputs = ["N0", "Ne0", "Ti", "Te", "Vexbx", "Vbtildx", "Vexby", "Vbtildy", "Vexbz", "Vbtildz"]
    for output in required_outputs:
        if not output in d["outputs"].keys() :
            print(f"Did not find {output}!!")
            warnings.warn("BOUT++ Output files do not contain all information required to calculate diffusion coefficients.")
    
    print(f"Found all relevant quantities in {data_dir}. Starting to collect variables.")
            
    # Collect all relevant quantities from the output files into numpy matrices
    ni = collect("Ni",path=data_dir,info=False)
    print("Finished collecting ni.")
    Ti = collect("Ti",path=data_dir,info=False)
    print("Finished collecting Ti.")
    Te = collect("Te",path=data_dir,info=False)
    print("Finished collecting Te.")
    
    # Collect all relevant quantities from the output files into numpy matrices
    ni0 = collect("N0",path=data_dir,info=False)
    print("Finished collecting N0.")
    ne0 = collect("Ne0",path=data_dir,info=False) # Current model assumes quasineutrality
    print("Finished collecting Ne0.")
    Ti0 = collect("Ti0",path=data_dir,info=False)
    print("Finished collecting Ti0.")
    Te0 = collect("Te0",path=data_dir,info=False)
    print("Finished collecting Te0.")
    
    Va = collect("Va",path=data_dir,info=False)
    print("Finished collecting Va.")
    
    Vexbx = collect("Vexbx",path=data_dir,info=False) * Va
    print("Finished collecting Vexbx.")
    Vexby = collect("Vexby",path=data_dir,info=False) * Va
    print("Finished collecting Vexby.")
    Vexbz = collect("Vexbz",path=data_dir,info=False) * Va
    print("Finished collecting Vexbz.")
    
    Vbtildx = collect("Vbtildx",path=data_dir,info=False) * Va
    print("Finished collecting Vbtildx.")
    Vbtildy = collect("Vbtildy",path=data_dir,info=False) * Va
    print("Finished collecting Vbtildy.")
    Vbtildz = collect("Vbtildz",path=data_dir,info=False) * Va
    print("Finished collecting Vbtildz.")
    
    Ve0_diax = collect("Ve0_diax",path=data_dir,info=False) * Va
    print("Finished collecting Ve0_diax.")
    Ve0_diay = collect("Ve0_diay",path=data_dir,info=False) * Va
    print("Finished collecting Ve0_diay.")
    Ve0_diaz = collect("Ve0_diaz",path=data_dir,info=False) * Va
    print("Finished collecting Ve0_diaz.")
    
    print("Calculating derived quantities.")
    
    t_ind = -1
    
    print("Calculating txy profiles.")
    ni_txy = ni0[:,:] + np.sum(ni[:,:,:,:], axis=3)/ni[:,:,:,:].shape[3]
    Ti_txy = Ti0[:,:] + np.sum(Ti[:,:,:,:], axis=3)/Ti[:,:,:,:].shape[3] 
    Te_txy = Te0[:,:] + np.sum(Te[:,:,:,:], axis=3)/Te[:,:,:,:].shape[3]

    ni_turb_txy = np.sum(ni[:,:,:,:], axis=3)/ni[:,:,:,:].shape[3]
    Ti_turb_txy = np.sum(Ti[:,:,:,:], axis=3)/Ti[:,:,:,:].shape[3] 
    Te_turb_txy = np.sum(Te[:,:,:,:], axis=3)/Te[:,:,:,:].shape[3]

    Vexbx_txy = np.sum(Vexbx[:,:,:,:], axis=3)/Vexbx[:,:,:,:].shape[3]
    Vexby_txy = np.sum(Vexby[:,:,:,:], axis=3)/Vexby[:,:,:,:].shape[3]
    Vexbz_txy = np.sum(Vexbz[:,:,:,:], axis=3)/Vexbz[:,:,:,:].shape[3]

    Vbtildx_txy = np.sum(Vbtildx[:,:,:,:], axis=3)/Vbtildx[:,:,:,:].shape[3]
    Vbtildy_txy = np.sum(Vbtildy[:,:,:,:], axis=3)/Vbtildy[:,:,:,:].shape[3]
    Vbtildz_txy = np.sum(Vbtildz[:,:,:,:], axis=3)/Vbtildz[:,:,:,:].shape[3]

    print("Calculating diffusion coefficient profiles.")
    dndr = np.zeros_like(ni_txy)
    dTidr = np.zeros_like(Ti_txy)
    dTedr = np.zeros_like(Te_txy)

    dr = np.zeros_like(ni_txy)
    dn = np.zeros_like(ni_txy)
    dTi = np.zeros_like(Ti_txy)
    dTe = np.zeros_like(Te_txy)

    for t in range(ni.shape[0]):
        for i, (r,z,n_) in enumerate(zip(Rxy.T, Zxy.T, ni_txy[t_ind,:,:].T)):
            dri = np.abs(average_distances(r,z))
            dndr[t,:,i] = np.abs(np.gradient(n_,dri))
            dr[t,:,i] = np.abs(dri)
            dn[t,:,i] = np.abs(np.gradient(n_))
        for i, (r,Ti_) in enumerate(zip(Rxy.T, Ti_txy[t,:,:].T)):
            dTidr[t,:,i] = np.abs(np.gradient(Ti_,average_distances(r,z)))
            dTi[t,:,i] = np.abs(np.gradient(Ti_))
        for i, (r,Te_) in enumerate(zip(Rxy.T, Te_txy[t,:,:].T)):
            dTedr[t,:,i] = np.abs(np.gradient(Te_,average_distances(r,z)))
            dTe[t,:,i] = np.abs(np.gradient(Te_))

    
    print("Calculating time averaged profiles.")
    ni_xy = np.sum(ni_txy[t_ind,:,:],axis=0)/ni_txy.shape[0]
    Ti_xy = np.sum(Ti_txy[t_ind,:,:],axis=0)/Ti_txy.shape[0]
    Te_xy = np.sum(Te_txy[t_ind,:,:],axis=0)/Te_txy.shape[0]

    Vexbx_xy = np.sum(Vexbx_txy[t_ind,:,:],axis=0)/Vexbx_txy.shape[0]
    Vexby_xy = np.sum(Vexby_txy[t_ind,:,:],axis=0)/Vexby_txy.shape[0]
    Vexbz_xy = np.sum(Vexbz_txy[t_ind,:,:],axis=0)/Vexbz_txy.shape[0]

    Vbtildx_xy = np.sum(Vbtildx_txy[t_ind,:,:],axis=0)/Vbtildx_txy.shape[0]
    Vbtildy_xy = np.sum(Vbtildy_txy[t_ind,:,:],axis=0)/Vbtildy_txy.shape[0]
    Vbtildz_xy = np.sum(Vbtildz_txy[t_ind,:,:],axis=0)/Vbtildz_txy.shape[0]
    
    
    print("Saving all data as numpy binary files.")
    
    data_names = ["ni","ni0","ni_txy","ni_xy",  # ion density profiles
                  "ne0",                        # electron density profiles
                  "Ti","Ti0","Ti_txy","Ti_xy",  # ion temperature profiles
                  "Te","Te0","Te_txy","Te_xy",  # electron temperature profiles
                  "Vexbx","Vexbx_txy","Vexbx_xy",
                  "Vexby","Vexby_txy","Vexbx_xy",
                  "Vexbz","Vexbz_txy","Vexbx_xy",
                  "Vbtildx","Vbtildx_txy","Vbtildx_xy",
                  "Vbtildy","Vbtildy_txy","Vbtildx_xy",
                  "Vbtildz","Vbtildz_txy","Vbtildx_xy",
                  "Ve0_diax",
                  "Ve0_diay",
                  "Ve0_diaz",
                  "ni_turb_txy", 
                  "Ti_turb_txy",
                  "Te_turb_txy",
                  "dndr","dTidr", "dTedr",
                  "Rxy","Zxy","Bpxy","Btxy",
                  "Va",
                  "dr","dn","dTi","dTe"]
    
    data_vars = [ni,ni0,ni_txy,ni_xy,  # ion density profiles
                 ne0,                        # electron density profiles
                 Ti,Ti0,Ti_txy,Ti_xy,  # ion temperature profiles
                 Te,Te0,Te_txy,Te_xy,  # electron temperature profiles
                 Vexbx,Vexbx_txy,Vexbx_xy,
                 Vexby,Vexby_txy,Vexby_xy,
                 Vexbz,Vexbz_txy,Vexbz_xy,
                 Vbtildx,Vbtildx_txy,Vbtildx_xy,
                 Vbtildy,Vbtildy_txy,Vbtildy_xy,
                 Vbtildz,Vbtildz_txy,Vbtildz_xy,
                 Ve0_diax,
                 Ve0_diay,
                 Ve0_diaz,
                 ni_turb_txy, 
                 Ti_turb_txy,
                 Te_turb_txy,
                 dndr,dTidr, dTedr,
                 Rxy,Zxy,Bpxy,Btxy,
                 Va,
                 dr,dn,dTi,dTe]

    
    for name, var in zip(data_names,data_vars):
        fid = numpy_dir + name
        var.tofile(fid)
        

def average_distances(r,z):
    """
    Computes the average spacing between each grid point in a 1D array.
    
    Parameters:
        r (numpy.ndarray): 1D array of grid points (sorted or unsorted).
    
    Returns:
        numpy.ndarray: 1D array of average spacing at each grid point.
    """
    r = np.asarray(r)  # Ensure it's a NumPy array
    z = np.asarray(z)  # Ensure it's a NumPy array
    
    # Compute forward and backward differences
    forward_diff_r = np.diff(r,append=0)  # Left difference
    backward_diff_r = np.diff(np.roll(r,1),append=0)  # Right difference
    
    forward_diff_z = np.diff(z,append=0)  # Left difference
    backward_diff_z = np.diff(np.roll(z,1),append=0)  # Right difference
    
    forward_dists = np.sqrt(forward_diff_r**2 + forward_diff_z**2)
    backward_dists = np.sqrt(backward_diff_r**2 + backward_diff_z**2)

    # Compute the average spacing for interior points
    avg_spacing = (forward_dists + backward_dists) / 2

    # Handle first and last elements separately
    avg_spacing[0] = np.sqrt((r[1] - r[0])**2 + (z[1]-z[0])**2)   # First point
    avg_spacing[-1] = np.sqrt((r[-1] - r[-2])**2 + (z[-1]-z[-2])**2) # Last point
    
    return avg_spacing
    
if __name__ == "__main__":
    t_ind = -1
    data_dir = "Data/r1_test_mv_dir"
    grid_fname = "Data/d3d_184016.3200_psi092108_nx260ny64.nc"
    bout_save_data(data_dir,grid_fname,t_ind)
    
