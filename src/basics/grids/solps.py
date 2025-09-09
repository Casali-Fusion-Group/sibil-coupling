#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:24:09 2025

@author: ttaczak
"""
import numpy as np

from ..readers.GQDSK_Reader import GQDSK
from ..readers.B2fstate_Reader import B2fstate
from ..readers.B2fplasmf_Reader import B2fplasmf
from ..readers.B2fgmtry_Reader import B2fgmtry
from boututils.datafile import DataFile
from .static import Static_grid
from matplotlib.collections import PolyCollection
import scipy.interpolate as si


import matplotlib.pyplot as plt

class Solps(Static_grid):
    def __init__(self, grid_name, config):
        Static_grid.__init__(self,grid_name,config)
        self.dl = None
        self.hy = None
        
        self.psin_max = None
        self.psin_min_pfr = None
        self.psin_min_cfs = None
        
        self._initialize_profiles()
        
            
    
    def _initialize_profiles(self):
        self._load_geo_file()
        self._load_dl()
        self._load_plasma_profiles()
        self._load_psixy()
        self._calculate_poloidal_angle_from_midplane()
        self._define_profile_region_indices()
        self._define_psin_extrema()
        self.grid_polygons = self._get_polygons()
        self.calc_flux_tube_lengths()
        
    def renormalize_flux_tube_lengths(self,bgrid):
        for i in range(self.nx):
            if i == self.ixsep:
                continue
            # if i < self.ixsep:
                
            # self.dl[i,self.j_pfr][-1]/2 aligns the end of the gridcell in the divertor region with the end of the gridcell in BOUT++
            sdl_max_pfr = np.max(self.dl2d[i,self.j_pfr]) #+ self.dl[i,self.j_pfr][-1]/2
            spsin = np.mean(self.psinxy[i, self.j_pfr])
            bix_closest_psin = np.argmin(np.mean(bgrid.psinxy[:,bgrid.j_pfr],axis=1) - np.abs(spsin))
            bdl_max_pfr = np.max(bgrid.dl2d[bix_closest_psin,bgrid.j_pfr]) #+ bgrid.dl[i,bgrid.j_pfr][-1]/2
            self.dl2d[i,self.j_pfr] = self.dl2d[i,self.j_pfr] * bdl_max_pfr / sdl_max_pfr 
            
            # self.dl[i,self.j_cfs][-1]/2 aligns the end of the gridcell in the CFS region with the end of the gridcell in BOUT++
            sdl_max_cfs = self.dl2d[i,self.j_cfs][-1] #+ self.dl[i,self.j_cfs][-1]/2
            spsin = np.mean(self.psinxy[i, self.j_pfr])
            bix_closest_psin = np.argmin(np.mean(bgrid.psinxy[:,bgrid.j_pfr],axis=1) - np.abs(spsin))
            bdl_max_cfs = bgrid.dl2d[bix_closest_psin,bgrid.j_cfs][-1] #+ bgrid.dl[i,bgrid.j_cfs][-1]/2
            self.dl2d[i,self.j_cfs] = self.dl2d[i,self.j_cfs] * bdl_max_cfs / sdl_max_cfs
            # else:
            #     sdl_max = np.max(self.dl2d[i,:]) + self.dl[i,:][-1]/2
            #     bdl_max = np.max(bgrid.dl2d[i,:]) + bgrid.dl[i,:][-1]/2
            #     self.dl2d[i,:] = self.dl2d[i,:] * bdl_max / sdl_max
        
    def radial_spline_interpolation(self, var, bgrid, slinear_cutoff_ind = 1, threshold=0):
        interp_var = np.zeros((bgrid.nx,self.ny))
        for j in range(self.ny):
            if not slinear_cutoff_ind == 0:
                if j < self.jyseps11+1 or j > self.jyseps22:
                    
                    x_interp = bgrid.psinxy[:,bgrid.jyseps11-1]
                    inner_interp_func = si.interp1d(self.psinxy[:-slinear_cutoff_ind,j], var[:-slinear_cutoff_ind,j], kind='quadratic', fill_value='extrapolate')
                    interpolated_values = inner_interp_func(x_interp)
                    
                else:
                    
                    x_interp = bgrid.psinxy[:,bgrid.jyseps11+1]
                    inner_interp_func = si.interp1d(self.psinxy[:,j], var[:, j], kind='quadratic', fill_value='extrapolate')
                    outer_interp_func = si.interp1d(self.psinxy[:-slinear_cutoff_ind, j], var[:-slinear_cutoff_ind, j], kind='linear', fill_value='extrapolate')
                    interpolated_values = np.where(x_interp > self.psinxy[:, j][-(slinear_cutoff_ind)], outer_interp_func(x_interp), inner_interp_func(x_interp))
                
                # # Apply thresholding: if extrapolated values go below the threshold, cap them at threshold
                # interpolated_values = np.where(interpolated_values < threshold, threshold, interpolated_values)
                
                interp_var[:, j] = interpolated_values
                
            else:
                
                if j < self.jyseps11+1 or j > self.jyseps22:
                    
                    x_interp = bgrid.psinxy[:,bgrid.jyseps11-1]
                    inner_interp_func = si.interp1d(self.psinxy[:, j], var[:, j], kind='quadratic', fill_value='extrapolate')
                    interpolated_values = inner_interp_func(x_interp)
                    
                    # interpolated_values = fix_interpolation_safe_bidirectional(interpolated_values,x=self.psinxy[:, j])
                else:
                    
                    x_interp = bgrid.psinxy[:,bgrid.jyseps11+1]
                    inner_interp_func = si.interp1d(self.psinxy[:, j], var[:, j], kind='quadratic', fill_value='extrapolate')
                    outer_interp_func = si.interp1d(self.psinxy[:, j], var[:, j], kind='linear', fill_value='extrapolate')
                    interpolated_values = np.where(x_interp > np.max(self.psinxy[:, j]), outer_interp_func(x_interp), inner_interp_func(x_interp))
                    
                    # interpolated_values = fix_interpolation_safe_bidirectional(interpolated_values,x=self.psinxy[:, j])
                    
                # # Apply thresholding: if extrapolated values go below the threshold, cap them by fixing the gradient intelligently
                # interpolated_values = np.where(interpolated_values < threshold, threshold, interpolated_values)
                
                interp_var[:, j] = interpolated_values
        
        return interp_var

        
    def _load_dl(self):
        geometry = B2fgmtry(self.config["b2fgmtry_loc"])
        dl = geometry.hx.T
        hy = geometry.hy.T
        hz = geometry.hz.T
        qc = geometry.qc.T

        self.dl = dl
        self.hy = hy
        
    def calc_flux_tube_lengths(self):
        
        self.dl2d=np.zeros((self.nx,self.ny))
        self.dl2d[:,self.j_pfr] = np.cumsum(self.dl[:,self.j_pfr],axis=1)
        self.dl2d[:,self.j_cfs] = np.cumsum(self.dl[:,self.j_cfs],axis=1)
        # for i in range(self.nx):
        #     if (i<self.ixsep): 
                
        #         # Flux tube length including both closed flux surface and privite flux regions
                
        #         # Closed flux surface
        #         dl1d = np.cumsum(self.dl[i,self.j_cfs]) #-0.5*self.dl[i,self.jyseps22] # self.j_cfs]  #MOST RECENT CHANGE
        #         self.dl2d[i,self.j_cfs] = np.copy(dl1d)
                
        #         # Private flux region
        #         dl1d = np.cumsum(self.dl[i,self.j_pfr]) #-np.sum(self.dl[i,:self.jyseps11+1])-0.5*self.dl[i,self.jyseps22+1] #self.j_pfr]#
        #         self.dl2d[i,self.j_pfr] = np.copy(dl1d)
                
        #     else:
                
        #         # dl = np.cumsum(self.dl[i,:]) 
                
        #         # # theta before, at, and after midplane
        #         # t_omp   = self.theta[i,self.jomp]
        #         # t_omp_p = self.theta[i,self.jomp+1]
        #         # t_omp_m = self.theta[i,self.jomp-1]
                
        #         # # arclength
        #         # d_omp   = self.dl[i,self.jomp] 
        #         # d_omp_p = self.dl[i,self.jomp+1]
                
        #         # # offset for spline fit
        #         # boff = np.sum(self.dl[i,:self.jomp+1]) 
                
        #         # if (t_omp < 0):
        #         #     boff=boff-t_omp/(t_omp_p-t_omp)*d_omp_p
        #         # else:
        #         #     boff=boff-t_omp/(t_omp-t_omp_m)*d_omp
        
        #         dl1d = np.cumsum(self.dl[i,:]) #- boff
        #         self.dl2d[i,:]=dl1d-0.5*self.dl[i,0]   #:] # -1]  #MOST RECENT CHANGE
                
        #         dl1d = np.cumsum(self.dl[i,:]) #- boff
        #         self.dl2d[i,:]=dl1d-0.5*self.dl[i,0]   #:] # -1]  #MOST RECENT CHANGE
        
    def _define_psin_extrema(self):
        # VALUES FOR MAKING THE GRID FILE IN HYPNOTOAD
        self.psin_max_pfr = np.max(self.psinxy[:self.ixsep,self.j_pfr])
        self.psin_min_pfr = np.min(self.psinxy[:self.ixsep,self.j_pfr])
        
        self.psin_max_cfs = np.max(self.psinxy[:self.ixsep,self.j_cfs])
        self.psin_min_cfs = np.min(self.psinxy[:self.ixsep,self.j_cfs])
        
    def _define_profile_region_indices(self):
        self.j_pfr=np.concatenate((np.arange(0,self.jyseps11+1),np.arange(self.jyseps22+1,self.ny)))
        self.j_cfs=np.arange(self.jyseps11+1,self.jyseps22+1)
        
    def _get_polygons(self):
        
        mesh_points = np.loadtxt(self.config["geofile_loc"],dtype="float64", skiprows=1,max_rows=self.nxy,usecols=(2,3,4,5,6,7,8,9,10,11,12,13))
        # Create SOLPS mesh and organize
        ll = mesh_points[:,(2,3)][:]
        lr = mesh_points[:,(4,5)][:]
        ul = mesh_points[:,(6,7)][:]
        ur = mesh_points[:,(8,9)][:]
        
        ll_ix = np.swapaxes(ll[:,1].reshape(self.ny,self.nx),0,1).flatten()
        ll_iy = np.swapaxes(ll[:,0].reshape(self.ny,self.nx),0,1).flatten()
        lr_ix = np.swapaxes(lr[:,1].reshape(self.ny,self.nx),0,1).flatten()
        lr_iy = np.swapaxes(lr[:,0].reshape(self.ny,self.nx),0,1).flatten()
        ul_ix = np.swapaxes(ul[:,1].reshape(self.ny,self.nx),0,1).flatten()
        ul_iy = np.swapaxes(ul[:,0].reshape(self.ny,self.nx),0,1).flatten()
        ur_ix = np.swapaxes(ur[:,1].reshape(self.ny,self.nx),0,1).flatten()
        ur_iy = np.swapaxes(ur[:,0].reshape(self.ny,self.nx),0,1).flatten()
        
        sverts = []
        for llx,lly,lrx,lry,ulx,uly,urx,ury in zip(ll_ix,ll_iy,lr_ix,lr_iy,ul_ix,ul_iy,ur_ix,ur_iy):
            ll_i = (llx,lly)
            lr_i = (lrx,lry)
            ul_i = (ulx,uly)
            ur_i = (urx,ury)
            sverts.append([lr_i,ll_i,ul_i,ur_i])
                
        # Create polygons and color them white
        spolygons = PolyCollection(
            sverts,
            facecolors='white',
            edgecolors='black',
            linewidths=0.05,
            alpha=1.0
        )
        
        return spolygons

        
    def _load_geo_file(self):
        with open(self.config["geofile_loc"]) as file:
            resolution = file.readline().split()
            self.nx, self.ny = int(resolution[1])+2,int(resolution[0])+2
            self.nxy = self.nx * self.ny
            self.ixsep = 15
            self.jyseps11 = 24
            self.jyseps22 = 72
            
            self.j_pfs=np.concatenate((np.arange(0,self.jyseps11+1),np.arange(self.jyseps22+1,self.ny)))
            self.j_cfs=np.arange(self.jyseps11+1,self.jyseps22+1)

            sgrid = np.array([list(map(float, file.readline().split())) for _ in range(self.nxy)])
            self.rxy = np.zeros((self.nx, self.ny))
            self.zxy = np.zeros((self.nx, self.ny))
            self.bpxy = np.zeros((self.nx, self.ny))
            self.btxy = np.zeros((self.nx, self.ny))
            
            for j in range(self.nx):
                
                i1d = (np.arange(0,self.ny,1) + self.ny*j).astype(int)
                self.rxy[j,:]=sgrid[i1d,2]
                self.zxy[j,:]=sgrid[i1d,3]
                self.bpxy[j,:]=sgrid[i1d,12]
                self.btxy[j,:]=sgrid[i1d,13]
                
            self.jomp = np.argmax(self.rxy[0,:])
            

            
    def _load_psixy(self):
        g = GQDSK(self.config["gfile_loc"])
        interp_func = si.RegularGridInterpolator((g.R_grid, g.Z_grid), g.psi, method='cubic')
        points = np.column_stack((self.rxy.flatten(), self.zxy.flatten()))
        self.psixy = np.reshape(interp_func(points), shape=self.rxy.shape)
        self.psinxy = (g.psiaxis - self.psixy) / (g.psiaxis - g.psiedge)
            
    def _load_plasma_profiles(self):
        fname_b2fstate, fname_b2fplasma, fname_b2fgmtry = self.config["b2fstate_loc"],self.config["b2fplasmf_loc"],self.config["b2fgmtry_loc"]
        b2fstate = B2fstate(fname_b2fstate)
        geometry = B2fgmtry(fname_b2fgmtry)
        b2fplasmf = B2fplasmf(fname_b2fplasma, b2fstate.nx, b2fstate.ny, b2fstate.ns)
        
        self.ne = b2fplasmf.ne.T # not needed
        self.Te = b2fplasmf.te.T
        self.ni = b2fplasmf.ni[:,:,1].T
        self.Ti = b2fplasmf.ti.T
        

    def _calculate_poloidal_angle_from_midplane(self):
        g = GQDSK(self.config["gfile_loc"])
        rcenter=g.Raxis  
        zcenter=g.Zaxis 
        romp=rcenter+1# define outboard mid-plane (OMP) in RZ coordinate
        zomp=zcenter    # define outboard mid-plane (OMP) in RZ coordinate
        self.theta = np.zeros_like(self.rxy)
        u = [romp - rcenter, zomp - zcenter, 0]

        for jy in range(self.ny):
            for ix in range(self.nx):
                v = [self.rxy[ix, jy] - rcenter, self.zxy[ix, jy] - zcenter, 0]
                self.theta[ix, jy] = np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))

        for ix in range(self.nx):
            itheta = np.argmax(self.theta[:, ix])
            if self.zxy[ix,itheta] > zomp:
                self.theta[ix,:itheta] = 2 * np.pi - self.theta[ix,:itheta]
            else:
                self.theta[ix,:itheta+1] = 2 * np.pi - self.theta[ix,:itheta+1]

            itheta = np.argmin(self.theta[ix,:])
            if self.zxy[ix,itheta] > zomp:
                self.theta[ix,:itheta+1] = -self.theta[ix,:itheta+1]
            else:
                self.theta[ix,:itheta] = -self.theta[ix,:itheta]
                
        
        