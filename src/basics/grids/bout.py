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
from scipy.optimize import minimize
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from boututils.datafile import DataFile

class Bout(Static_grid):
    def __init__(self, grid_name, config):
        '''
        Specialized grid class to handle the BOUT++ specific methods.
        The only additional parameters are psi_bndry and dl but the 
        functionality allows for reading/writing from BOUT++ files as well
        as extracting profiles from the mapping grid.

        Parameters
        ----------
        grid_name : STR
            DESCRIPTION.
        config : DICT
        config : DICT
            A CONFIGURATION FILE THAT INCLUDES THE LOCATIONS AND FILENAMES 
            OF BOUT GRID FILE AND GFILE. THE REQUIRED INPUTS FOR BOUT
            INCLUDE:
                'gfile_loc': '/gfile/location/g######.####',
                'bgrid_loc': '/bout/grid/file/location/bout.nc'

        Returns
        -------
        None.

        '''
        Static_grid.__init__(self, grid_name, config)
        self.dl = None
        self.psi_bndry = None
        self._load_bout_profile(config["bgrid_loc"])
        self.grid_polygons = self._get_polygons()
        self._define_profile_region_indices()
        self._calculate_poloidal_angle_from_midplane(config["gfile_loc"])
        self.calc_flux_surface_distances()
        self.calc_flux_tube_lengths()
        
    
        
    def smooth_profiles(self):
        '''
        Wrapper for profile smoothing. Currently, the sigma values are fixed
        for the gaussian smoothing, and straight line smoothing is employed
        in regions of greatest curvature. 

        Returns
        -------
        None.

        '''
        self.smooth_radial_profiles(self.ni,sigma=10, smoothed_choords = self.j_pfr,percentile_interpolated=80)
        self.smooth_radial_profiles(self.Ti,sigma=10, smoothed_choords = self.j_pfr,percentile_interpolated=40)
        self.smooth_radial_profiles(self.Te,sigma=10, smoothed_choords = self.j_pfr,percentile_interpolated=60)
        
        # additional_smoothing_indices_list = self.j_cfs #[self.jyseps11+1,self.jyseps11+2,self.jyseps11+3,self.jyseps11+4,self.jyseps22-3,self.jyseps22-2,self.jyseps22-1,self.jyseps22]
        # additional_smoothing_indices = np.array(additional_smoothing_indices_list)
        # self.smooth_radial_profiles(self.ni,sigma=10, smoothed_choords = additional_smoothing_indices, percentile_interpolated=80)
        # self.smooth_radial_profiles(self.Ti,sigma=10, smoothed_choords = additional_smoothing_indices, percentile_interpolated=40)
        # self.smooth_radial_profiles(self.Te,sigma=10, smoothed_choords = additional_smoothing_indices, percentile_interpolated=60)
        
        
        # self.smooth_radial_profiles(self.ni,sigma=5,smoothed_choords = self.j_cfs,percentile_interpolated=70)
        # self.smooth_radial_profiles(self.Ti,sigma=1,smoothed_choords = self.j_cfs,percentile_interpolated=95)
        # self.smooth_radial_profiles(self.Te,sigma=1,smoothed_choords = self.j_cfs,percentile_interpolated=95)
    
        
    
    def plot_grid_intersections(self):
        fig, ax = plt.subplots(1,1,dpi=800)
        self.plot_grid(ax=ax)
        
    def calc_flux_tube_lengths(self):
        
        # self.dl2d = self.poloidal_distance
        
        self.dl2d=np.zeros((self.nx,self.ny))
        
        self.dl2d[:,self.j_pfr] = np.cumsum(self.dl[:,self.j_pfr],axis=1)
        self.dl2d[:,self.j_cfs] = np.cumsum(self.dl[:,self.j_cfs],axis=1)
        
        # for j in range(self.ny):
        #     if j == 0:
        #         self.dl[:,j] = self.poloidal_distance[:,j]
        #     elif j == self.jyseps11+1: # first poloidal coordinate past the divertor
        #         self.dl[:,j] = 0 
        #     elif j == self.jyseps22+1: # first poloidal coordinate in the outer divertor
        #         self.dl[:,j] = self.poloidal_distance[:,j] - self.poloidal_distance[:,self.jyseps11] 
        #     else:
        #         self.dl[:,j] = self.poloidal_distance[:,j] - self.poloidal_distance[:,j-1] 
        
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
                
        #         dl = np.cumsum(self.dl[i,:]) 
                
        #         # theta before, at, and after midplane
        #         t_omp   = self.theta[i,self.jomp]
        #         t_omp_p = self.theta[i,self.jomp+1]
        #         t_omp_m = self.theta[i,self.jomp-1]
                
        #         # arclength
        #         d_omp   = self.dl[i,self.jomp] 
        #         d_omp_p = self.dl[i,self.jomp+1]
                
        #         # offset for spline fit
        #         boff = np.sum(self.dl[i,:self.jomp+1]) 
                
        #         if (t_omp < 0):
        #             boff=boff-t_omp/(t_omp_p-t_omp)*d_omp_p
        #         else:
        #             boff=boff-t_omp/(t_omp-t_omp_m)*d_omp
        
        #         dl1d = np.cumsum(self.dl[i,:]) - boff
        #         self.dl2d[i,:]=dl1d-0.5*self.dl[i,:]   # -1]  #MOST RECENT CHANGE  
        
    def calc_flux_surface_distances(self):
        '''
        Wrapper for estimating the flux surface distance between grid points 
        with dl=sqrt(drxy^2+dzxy^2) 

        Returns
        -------
        None.

        '''
        self.dl=np.zeros((self.nx,self.ny))
        
        self.dl2d = self.poloidal_distance
        
        # for j in range(self.ny):
        #     if j == 0:
        #         self.dl[:,j] = self.poloidal_distance[:,j]
        #     elif j == self.jyseps11+1: # first poloidal coordinate past the divertor
        #         self.dl[self.ixsep+1:,j] = self.poloidal_distance[self.ixsep+1:,j] - self.poloidal_distance[self.ixsep+1:,j-1] 
        #         self.dl[:self.ixsep+1,j] = self.poloidal_distance[:self.ixsep+1,j]
        #     elif j == self.jyseps22+1: # first poloidal coordinate in the outer divertor
        #         self.dl[self.ixsep+1:,j] = self.poloidal_distance[self.ixsep+1:,j] - self.poloidal_distance[self.ixsep+1:,j-1] 
        #         self.dl[:self.ixsep+1,j] = self.poloidal_distance[:self.ixsep+1,j] - self.poloidal_distance[:self.ixsep+1,self.jyseps11] 
        #     else:
        #         self.dl[:,j] = self.poloidal_distance[:,j] - self.poloidal_distance[:,j-1] 
        
        
        # for the private flux region and closed flux surfaces, estimate 
        # the distances separately
        for i in range(self.ixsep):
            self.dl[i,self.j_cfs]=np.sqrt((self.rxy[i,self.j_cfs+1]-self.rxy[i,self.j_cfs])**2 + (self.zxy[i,self.j_cfs+1]-self.zxy[i,self.j_cfs])**2) 
            self.dl[i,self.jyseps22]=np.sqrt((self.rxy[i,self.jyseps11+1]-self.rxy[i,self.jyseps22])**2+(self.zxy[i,self.jyseps11+1]-self.zxy[i,self.jyseps22])**2) 
            self.dl[i,self.j_pfr[0:7]]=np.sqrt((self.rxy[i,self.j_pfr[1:8]]-self.rxy[i,self.j_pfr[0:7]])**2+(self.zxy[i,self.j_pfr[1:8]]-self.zxy[i,self.j_pfr[0:7]])**2) 
            self.dl[i,self.j_pfr[7]]=self.dl[i,self.j_pfr[6]]

        # For the SOL, integrating the flux surfaces is made easier
        for i in range(self.ixsep,self.nx):
            self.dl[i,0:self.ny-1]=np.sqrt((self.rxy[i,1:self.ny]-self.rxy[i,0:self.ny-1])**2+(self.zxy[i,0:self.ny-1]-self.zxy[i,1:self.ny])**2) 

    def calc_profiles_from_map(self, mgrid):
        '''
        Wrapper for extracting the profiles from the mapping grid radial 
        profiles. 

        Parameters
        ----------
        mgrid : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.ni = self.flux_spline_interpolation(mgrid.ni, mgrid)
        self.ne = self.flux_spline_interpolation(mgrid.ne, mgrid)
        self.Ti = self.flux_spline_interpolation(mgrid.Ti, mgrid)
        self.Te = self.flux_spline_interpolation(mgrid.Te, mgrid)
        
    # def couple_grid_for_divertor(self, solps):
    #     pfr_dl2d = np.zeros((self.nx,self.j_pfr.shape(0)))
        
    #     dl1d = np.cumsum(self.dl[i,self.j_pfr])
    #     self.dl2d[i,self.j_pfr] = np.copy(dl1d)
        
    #     half_pfr = int(self.j_pfr.shape[0]/2)
    #     dl1d_second_half = np.cumsum(self.dl[i,self.j_pfr[half_pfr:]])
    #     self.dl2d[i,self.j_pfr[half_pfr:]] = np.copy(dl1d_second_half)
        
    def fix_midplane_and_pfr(self):
        
        ni_midplane = self.ni[:,self.jomp]
        Ti_midplane = self.Ti[:,self.jomp]
        Te_midplane = self.Te[:,self.jomp]
        for jy in np.arange(self.ny):
            self.ni[:,jy] = ni_midplane
            self.Ti[:,jy] = Ti_midplane
            self.Te[:,jy] = Te_midplane
            if jy in self.j_pfr:
                self.ni[:self.ixsep+1,jy] = ni_midplane[self.ixsep+1]
                self.Ti[:self.ixsep+1,jy] = Ti_midplane[self.ixsep+1]
                self.Te[:self.ixsep+1,jy] = Te_midplane[self.ixsep+1]
        
    def flux_spline_interpolation(self, var, mgrid):
        '''
        Interpolates the flux splines from the mapping grid using the flux
        tube lengths previously calculated.

        Parameters
        ----------
        var : NUMPY ARR
            VARIABLE TO BE INTERPOLATED. IS BE AN (self.nx,self.ny) ARRAY.
        mgrid : BASICS GRID
            GRID FILE FROM WHICH THE INFORMATION WILL BE EXTRACTED.

        Returns
        -------
        interp_var : TYPE
            DESCRIPTION.

        '''
        # define the variable used for the BOUT++ grid
        interp_var = np.zeros((self.nx,self.ny))
        
        # # interpolate inner divertor region
        # thx1d = mgrid.dl1d[:,:]
        
        # # interpoalte outer divertor region
        
        # # interpolate CFS region
        
        # # interpolate SOL region above X-point
        
        # Define the interpolation for each flux tube
        for i in range(self.nx):
            if (i<self.ixsep): # including both closed flux surface and privite flux regions
        
                # for closed flux surface region
                thx1d=mgrid.dl2d[i,mgrid.j_cfs]
                dl1d=self.dl2d[i,self.j_cfs]
                interp_var[i,self.j_cfs] = si.interp1d(thx1d,var[i,mgrid.j_cfs],kind='cubic',fill_value='extrapolate')(dl1d)
                
                # interpolation 
                # linear_interp = si.interp1d(mgrid.dl2d[i,mgrid.j_cfs],var[i,mgrid.j_cfs],kind='linear',fill_value='extrapolate')
                # cubic_interp = si.interp1d(mgrid.dl2d[i,mgrid.j_cfs],var[i,mgrid.j_cfs],kind='cubic',fill_value='extrapolate')
                # interp_var[i,self.j_cfs] = np.where(dl1d > np.max(thx1d), linear_interp(dl1d), cubic_interp(dl1d))
                
                interp_var[i,self.j_cfs] = fix_until_positive(interp_var[i,self.j_cfs],x=dl1d)
        
                # for privite flux surface region 
                thx1d=mgrid.dl2d[i,mgrid.j_pfr]
                dl1d=self.dl2d[i,self.j_pfr]
                interp_var[i,self.j_pfr] = si.interp1d(thx1d,var[i,mgrid.j_pfr],kind='cubic',fill_value='extrapolate')(dl1d)
                
                # linear_interp = si.interp1d(thx1d,var[i,mgrid.j_pfr],kind='linear',fill_value='extrapolate')
                # cubic_interp = si.interp1d(thx1d,var[i,mgrid.j_pfr],kind='cubic',fill_value='extrapolate')
                # interp_var[i,self.j_pfr] = np.where(dl1d > np.max(thx1d[:-1]), linear_interp(dl1d), cubic_interp(dl1d))
                
                interp_var[i,self.j_pfr] = fix_until_positive(interp_var[i,self.j_pfr],x=dl1d)
            else:
                
                thx1d=mgrid.dl2d[i,mgrid.j_cfs]
                dl1d =self.dl2d[i,self.j_cfs]
                interp_var[i,self.j_cfs] = si.interp1d(thx1d,var[i,mgrid.j_cfs],kind='cubic',fill_value='extrapolate')(dl1d)
                
                # interp_var[i,self.j_cfs] = fix_until_positive(interp_var[i,self.j_cfs],x=dl1d)
                
                thx1d=mgrid.dl2d[i,mgrid.j_pfr]
                dl1d =self.dl2d[i,self.j_pfr]
                interp_var[i,self.j_pfr] = si.interp1d(thx1d,var[i,mgrid.j_pfr],kind='cubic',fill_value='extrapolate')(dl1d)
                
                # interp_var[i,self.j_pfr] = fix_until_positive(interp_var[i,self.j_pfr],x=dl1d)
                
                # linear_interp = si.interp1d(thx1d,var[i,:],kind='linear',fill_value='extrapolate')
                # cubic_interp = si.interp1d(thx1d,var[i,:],kind='cubic',fill_value='extrapolate')
                # interp_var[i,:] = np.where(dl1d > np.max(thx1d[:-1]), linear_interp(dl1d), cubic_interp(dl1d))
            
        return interp_var
        
     
    
    def _define_profile_region_indices(self):
        self.j_pfr=np.concatenate((np.arange(0,self.jyseps11+1),np.arange(self.jyseps22+1,self.ny)))
        self.j_cfs=np.arange(self.jyseps11+1,self.jyseps22+1)
        
    
    def _get_polygons(self):
        with DataFile(self.config["bgrid_loc"], write=False) as grid:
            try:
                _ = grid["psi_bdry"]
                has_psi_bdry = True
            except Exception:
                has_psi_bdry = False
            
            if has_psi_bdry:
                r_ll = grid["Rxy_corners"]
                r_lr = grid["Rxy_lower_right_corners"] 
                r_ur = grid["Rxy_upper_right_corners"] 
                r_ul = grid["Rxy_upper_left_corners"] 
                
                z_ll = grid["Zxy_corners"]
                z_lr = grid["Zxy_lower_right_corners"] 
                z_ur = grid["Zxy_upper_right_corners"] 
                z_ul = grid["Zxy_upper_left_corners"] 
                
                # Create a list of polygons using 4 tuples of cordinates of each corner of each grid cell
                bverts = []
                for rll,rlr,rur,rul,zll,zlr,zur,zul in zip(r_ll.flatten(),r_lr.flatten(),r_ur.flatten(),r_ul.flatten(),
                                          z_ll.flatten(),z_lr.flatten(),z_ur.flatten(),z_ul.flatten()):
                    bverts.append([(rll,zll), (rlr,zlr), (rur,zur), (rul,zul)])

                # Create polygons and color them correctly
                bpolygons = PolyCollection(bverts,
                                           facecolors='white',
                                           edgecolors='black',
                                           linewidths=0.01,
                                           alpha=1.0)
            # else:
            #     rxy_centers = grid["Rxy"]
            #     zxy_centers = grid["Zxy"]
                
            #     bverts = []
            #     # Loop over each cell center to calculate corner points
            #     for i in range(grid["nx"] - 1):  # Exclude last row for corner estimation
            #         for j in range(grid["ny"] - 1):  # Exclude last column for corner estimation
            #             # Approximate corners by averaging adjacent cell centers
            #             x0, y0 = rxy_centers[i, j], zxy_centers[i, j]         # Current cell center
            #             x1, y1 = rxy_centers[i + 1, j], zxy_centers[i + 1, j] # Below cell center
            #             x2, y2 = rxy_centers[i + 1, j + 1], zxy_centers[i + 1, j + 1] # Diagonal cell center
            #             x3, y3 = rxy_centers[i, j + 1], zxy_centers[i, j + 1] # Right cell center
                
            #             # Define the corners of the current grid cell (clockwise or counter-clockwise)
            #             corners = [(x0, y0), (x3, y3), (x2, y2), (x1, y1)]
            #             bverts.append(corners)
                        
            #     # for verts in bverts:
            #     #     for point in verts:
            #     #         plt.scatter(point[0], point[1], c='b')
            #     #         plt.scatter(point[0], point[1], c='r')
            #     # plt.show()
            #     # print("done")
            else:
                raise ValueError("Older BOUT++ grid detected. Please use python hypnotoad version of BOUT++ grid!")
                
            Rxy = grid["Rxy"]
            Zxy = grid["Zxy"]
            return bpolygons

    def _load_bout_profile(self, grid_fname):
        with DataFile(grid_fname, write=True) as grid:
            # if "Ni" in grid.keys(): self.ni = grid["Ni"]
            # if "Ne" in grid.keys(): self.ne = grid["Ne"]
            # if "Ti" in grid.keys(): self.Ti = grid["Ti"]
            # if "Te" in grid.keys(): self.Te = grid["Te"]
            
            self.rxy = grid["Rxy"]
            self.zxy = grid["Zxy"]
            self.nx, self.ny = self.rxy.shape
            self.nxy = self.nx * self.ny
            self.ixsep = int(grid["ixseps1"]) - 1
            self.jyseps11 = int(grid["jyseps1_1"])
            self.jyseps22 = int(grid["jyseps2_2"])
            self.jomp=np.argmax(self.rxy[0,:])

            self.j_pfr=np.concatenate((np.arange(0,self.jyseps11+1),np.arange(self.jyseps22+1,self.ny)))
            self.j_cfs=np.arange(self.jyseps11+1,self.jyseps22+1)

            self.bpxy = grid["Bpxy"]
            self.btxy = grid["Btxy"]
            self.psixy = grid["psixy"]
            self.psi_axis = grid["psi_axis"]
            self.poloidal_distance = grid["poloidal_distance"] # Poloidal distance (in metres) from the lower divertor target of each 
                                                               # flux surface to the grid point (on open field lines), or from the 
                                                               # poloidal location of the lower X-point (on closed field lines).
            
            try:
                _ = grid["psi_bdry"]
                has_psi_bdry = True
            except Exception:
                has_psi_bdry = False
                
            if has_psi_bdry:
                self.psi_bndry = grid["psi_bdry"]
            else:
                self.psi_bndry = grid["psi_bndry"]
                
            self.psinxy = (self.psixy - self.psi_axis) / (self.psi_bndry - self.psi_axis)
            
    def _calculate_poloidal_angle_from_midplane(self, gfile_loc):
        g = GQDSK(gfile_loc)
        
        rcenter=g.Raxis  
        zcenter=g.Zaxis 
        self.romp=rcenter+1 # define outboard mid-plane (OMP) in RZ coordinate
        self.zomp=zcenter    # define outboard mid-plane (OMP) in RZ coordinate
        
        self.theta = np.zeros_like(self.rxy)
        u = [self.romp - rcenter, self.zomp - zcenter, 0]
        
        for ix in range(self.nx):
            for jy in range(self.ny):
                v = [self.rxy[ix, jy] - rcenter, self.zxy[ix, jy] - zcenter, 0]
                self.theta[ix, jy] = np.arctan2(np.linalg.norm(np.cross(u, v)), np.dot(u, v))

            itheta = np.argmax(self.theta[ix, :]) - 1
            if self.zxy[ix, itheta] > self.zomp:
                self.theta[ix, :itheta] = 2 * np.pi - self.theta[ix, :itheta]
            else:
                self.theta[ix, :itheta+1] = 2 * np.pi - self.theta[ix, :itheta+1]

            itheta = np.argmin(self.theta[ix, :]) - 1
            if self.zxy[ix, itheta] > self.zomp:
                self.theta[ix, :itheta+1] = -self.theta[ix, :itheta+1]
            else:
                self.theta[ix, :itheta] = -self.theta[ix, :itheta]
                
    def smooth_radial_profiles(self, bvar, smoothed_choords = None, sigma=None, percentile_interpolated = 90):
        
        if smoothed_choords is None: 
            choords = np.arange(self.ny)
        else:
            choords = smoothed_choords
            
        # smoothed_var = np.zeros_like(bvar)
        for choord in smoothed_choords:
            
            x = self.psinxy[:,choord]
            y = bvar[:,choord]
            y_log = np.log(y)
            
            # Compute first and second derivatives of y with respect to x
            dy_dx = np.gradient(y_log, x)
            d2y_dx2 = np.gradient(dy_dx, x)
            
            
            # Threshold for "high curvature" - choose by percentile
            threshold = np.percentile(np.abs(d2y_dx2), percentile_interpolated)  # top percentile_interpolated second derivative magnitude
            
            # Find indices where |second derivative| is above threshold
            high_curv_indices = np.where(np.abs(d2y_dx2) > threshold)[0]
            
            runs = find_consecutive_runs(high_curv_indices)
            
            # Create a copy of y to modify
            y_modified = y.copy()
            
            for run in runs:
                start_idx = run[0] - 1  # one before run start
                end_idx = run[-1] + 1   # one after run end
                
                # Clamp to valid indices
                start_idx = max(start_idx, 0)
                end_idx = min(end_idx, len(y) - 1)
            
                # Linear interpolation between y[start_idx] and y[end_idx]
                x_segment = x[start_idx:end_idx+1]
                y_start = y_modified[start_idx]
                y_end = y_modified[end_idx]
            
                # Linear interpolation formula
                y_modified[start_idx:end_idx+1] = y_start + (y_end - y_start) * (x_segment - x_segment[0]) / (x_segment[-1] - x_segment[0])
            
            if sigma is not None:
                # Now apply Gaussian smoothing filter
                y_smoothed = gaussian_filter1d(y_modified, sigma=sigma)
                bvar[:,choord] = y_smoothed
            else:
                bvar[:,choord] = y_modified
        
    # def gaussian_smooth(self, bvar):
    #     bvar_ = np.copy(bvar)
    #     for i in range(bvar_.shape[1]):
    #         y = bvar_[:,i]
    #         x = self.psixy[:,i]
            
    #         y_new = gaussian_filter1d(y, 20, axis=-1, order=0)
            
    #         bvar_[:,i] = y_new
            
    #     return bvar_
    
    def update_gridfile(self,density_units=1e20):
        ev_to_J = 1.60218e-19
        with DataFile(self.config["bgrid_loc"], write=True) as grid:
            grid.write("Niexp", self.ni/density_units,info=True)
            grid.write("Neexp", self.ne/density_units,info=True)
            grid.write("Tiexp", self.Ti,info=True)
            grid.write("Teexp", self.Te,info=True)
            grid.write("P", np.multiply(self.Ti, self.ni) * ev_to_J +  np.multiply(self.Te * ev_to_J, self.ni),info=True)
            grid.write("pressure", np.multiply(self.Ti, self.ni) * ev_to_J  +  np.multiply(self.Te * ev_to_J , self.ni),info=True)
            grid.write("pressure_s", np.multiply(self.Ti, self.ni) * ev_to_J  +  np.multiply(self.Te * ev_to_J , self.ni),info=True)

                
            # grid.write("Ni", self.ni/ni_units,info=True)
            # grid.write("Ne", self.ne/ni_units,info=True)
            # grid.write("Ti", self.Ti,info=True)
            # grid.write("Te", self.Te,info=True)
            #grid.write("Tiexp", np.maximum(self.Ti,0.001),info=True)
            #grid.write("Teexp", np.maximum(self.Te,0.001),info=True)
                
def fix_until_positive(values, x=None, threshold=0):
    """
    Adjust interpolated values: if a value goes below zero,
    find a safe gradient but only require that it stays non-negative 
    until the values naturally rise above zero again.
    
    Args:
        values (np.ndarray): 1D array of interpolated values.
        x (np.ndarray, optional): x-positions (same shape as values). 
                                  If None, assumes uniform spacing.
    
    Returns:
        np.ndarray: Fixed array.
    """
    values = np.array(values)
    fixed_values = values.copy()
    n = len(values)

    if x is None:
        x = np.arange(n)

    for idx in range(n):
        if fixed_values[idx] < threshold:
            # Find how long the negative patch lasts
            patch_end = idx
            while patch_end < n and fixed_values[patch_end] < 0:
                patch_end += 1
            
            if idx == 0:
                # At beginning: search forward
                found_safe = False
                for first_good_idx in range(1, patch_end):
                    grad = (fixed_values[first_good_idx+1] - fixed_values[first_good_idx]) / (x[first_good_idx+1] - x[first_good_idx])
                    
                    # Predict backward only up to patch_end
                    safe = True
                    for j in range(0, first_good_idx+1):
                        dx = x[j] - x[first_good_idx]
                        predicted_value = fixed_values[first_good_idx] + grad * dx
                        if predicted_value < threshold:
                            safe = False
                            break

                    if safe:
                        # Apply extrapolation backwards only up to patch_end
                        for j in range(0, patch_end):
                            dx = x[j] - x[first_good_idx]
                            fixed_values[j] = fixed_values[first_good_idx] + grad * dx
                        found_safe = True
                        break

                if not found_safe:
                    raise RuntimeError("Could not find a safe gradient at the beginning.")

            else:
                # Search backward
                found_safe = False
                for last_good_idx in range(idx-1, 0, -1):
                    grad = (fixed_values[last_good_idx] - fixed_values[last_good_idx-1]) / (x[last_good_idx] - x[last_good_idx-1])
                    
                    # Predict forward only up to patch_end
                    safe = True
                    for j in range(idx, patch_end):
                        dx = x[j] - x[last_good_idx]
                        predicted_value = fixed_values[last_good_idx] + grad * dx
                        if predicted_value < threshold:
                            safe = False
                            break

                    if safe:
                        # Apply extrapolation forward only up to patch_end
                        for j in range(idx, patch_end):
                            dx = x[j] - x[last_good_idx]
                            fixed_values[j] = fixed_values[last_good_idx] + grad * dx
                        found_safe = True
                        break

                if not found_safe:
                    raise RuntimeError("Could not find a safe gradient in the patch.")

            break  # Done fixing after first bad patch

    return fixed_values

# Function to find consecutive runs of indices
def find_consecutive_runs(indices):
    runs = []
    if len(indices) == 0:
        return runs
    
    run_start = indices[0]
    run = [run_start]

    for i in range(1, len(indices)):
        if indices[i] == indices[i-1] + 1:
            run.append(indices[i])
        else:
            runs.append(run)
            run = [indices[i]]
    runs.append(run)
    return runs