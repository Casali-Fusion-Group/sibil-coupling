#!/fusion/projects/codes/pyped/python3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 10:58:09 2023

@author: welsh
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  4 10:30:58 2023

@author: welsh
"""
"""
Modified on Mon Jan 22 2024
Modified on Thu Oct 17 2024
Naming conventions after Matlab script written by Ben Zhu

@author: taczak
"""

import numpy as np
import scipy.interpolate
from scipy.integrate import cumulative_trapezoid
from skimage.measure import find_contours
import matplotlib.pyplot as plt

class GQDSK:
    data_list = None        # All of data in one line separated by spaces
    comments = None         # Metadata including type, date, shot number, shot time
    switch = None           # Type of file, should be 3 or else incompatible datatype
    num_r_pts = None        # Number of grid points in the R direction
    num_z_pts = None        # Number of grid points in the Z direction
    r_box_length = None     # Horizontal dimension of computational/data domain (m)
    z_box_length = None     # Vertical dimension of computational/data domain (m)
    R0EXP = None            # Normalizing radius in CHEASE -> Ben Zhu comment
    rboxleft = None         # Lower left R Coordinate
    Raxis = None            # R at magnetic axis
    Zaxis = None            # Z at magnetic axis
    psiaxis = None          # Psi at magnetic axis
    psiedge = None          # Psi at LCFS
    B0EXP = None            # Magnetic Field Strength of Experiment
    current = None          # Plasma current
    T = None                # 
    p = None                #Pressure
    TTprime = None
    pprime = None
    psi = None
    q = None                #Safety Factor wrt psi
    nLCFS = None
    nlimits = None
    R_LCFS = None
    Z_LCFS = None
    R_limits = None
    Z_limits = None
    R_grid = None
    Z_grid = None
    psi_grid = None
    rhopsi = None
    phi = None
    rhophi = None
    phirz = None
    rhorz = None
    rhos = None
    sep = None
    r_convert = None
    z_convert = None
    contour_convert_r = None
    contour_convert_z = None
    rhos_sep = None
    rhomag = None
    rhonormrz = None
    
    def __init__(self,fname):
        try:
            open(fname, 'r')
        except FileNotFoundError:
            print(f"File '{fname}' not found.")
            
        with open(fname) as file:
            data = self.extractData(file)
            
    def extractData(self,file):
        # Note: This can get slow for larger EQUDSK file but I haven't
        # yet encountered any, so this is more human readible
        
        # Read in all data into one line with random whitespaces
        un_formatted_data = ' '.join(file.readlines())
        
        # Normalize the length of the whitespaces
        data = un_formatted_data.split()
        self.data_list = data
        
        # After some reading, there doesn't seem to be a direct replacement for
        # fscanf in pyhton (ie. dynamically changing numbers being read based
        # on previous data), so this next part is rather forced
        i = 0
        
        ## Line Zero ##
        num_comments = 4; self.comments = data[i:i+num_comments]; i += num_comments
        
        self.switch = int(data[i]); i += 1 
        if not self.switch == 3:
            raise Warning(f"Warning: switch i3={str(self.switch)}, maybe different file structure...")
        
        self.num_r_pts = int(data[i]); i += 1 
        self.num_z_pts = int(data[i]); i += 1 
        
        
        ## First Line ##
        self.r_box_length = float(data[i]); i += 1 
        self.z_box_length = float(data[i]); i += 1 
        self.R0EXP = float(data[i]); i += 1 
        self.rboxleft = float(data[i]); i += 1 
        i += 1 # Nothing
        
        ## Second Line ##
        self.Raxis = float(data[i]); i += 1 
        self.Zaxis = float(data[i]); i += 1 
        self.psiaxis = float(data[i]); i += 1 
        self.psiedge = float(data[i]); i += 1 
        self.B0EXP = float(data[i]); i += 1 
        
        ## Third Line ##
        self.current = float(data[i]); i += 1 
        i += 4 # Nothing or already stored
        
        ## Fourth Line ##
        i += 5 # Nothing or already stored
        
        
        self.T = np.array(data[i:i+self.num_r_pts]).astype(float); i += self.num_r_pts
        
        self.p = np.array(data[i:i+self.num_r_pts]).astype(float); i += self.num_r_pts
        
        self.TTprime = np.array(data[i:i+self.num_r_pts]).astype(float); i += self.num_r_pts
        
        self.pprime = np.array(data[i:i+self.num_r_pts]).astype(float); i += self.num_r_pts
        
        self.psirz = np.array(data[i:i+self.num_r_pts*self.num_z_pts]).astype(float); i += self.num_r_pts*self.num_z_pts   
        self.psi = np.reshape(self.psirz, (self.num_r_pts, self.num_z_pts), order='F')
        
        self.q = np.array(data[i:i+self.num_r_pts]).astype(float); i += self.num_r_pts
        
        self.nLCFS = int(data[i]); i += 1 
        
        self.nlimits = int(data[i]); i += 1 
        
        tmp = np.array(data[i:i+self.nLCFS*2]).astype(float); i += self.nLCFS*2
        self.R_LCFS = tmp[0:-2:2]
        self.Z_LCFS = tmp[1:-1:2]
        
        tmp = np.array(data[i:i+self.nlimits*2]).astype(float); i += self.nlimits*2
        self.R_limits = tmp[0:-2:2]
        self.Z_limits = tmp[1:-1:2]
        
        self.R_grid = np.linspace(self.rboxleft,self.rboxleft+self.r_box_length, self.num_r_pts)
        self.Z_grid = np.linspace(-self.z_box_length/2,self.z_box_length/2, self.num_z_pts)
        
        self.psi_grid = np.linspace(self.psiaxis,self.psiedge,self.num_r_pts) #psi_rho
        self.phi = cumulative_trapezoid(self.q,self.psi_grid, initial=0) * (2.0*np.pi)
        self.rhophi = np.sqrt(self.phi)
        self.phirz = scipy.interpolate.interp1d(self.psi_grid,self.phi, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi)
        self.rhorz = np.sqrt(self.phirz)
        self.rhos = scipy.interpolate.RectBivariateSpline(self.R_grid,self.Z_grid,self.rhorz)
        self.rhopsi = np.sqrt(np.divide(np.abs(self.psi_grid-self.psiaxis),np.abs(self.psiedge-self.psiaxis)));
        self.sep = find_contours(self.psi,level=self.psiedge)[0]
        self.r_convert = scipy.interpolate.interp1d(range(self.num_r_pts),self.R_grid, kind='linear')(self.sep[:,0])
        self.z_convert = scipy.interpolate.interp1d(range(self.num_z_pts),self.Z_grid, kind='linear')(self.sep[:,1])
        self.rhos_sep = self.rhos(self.r_convert[0],self.z_convert[0])
        self.rhomag = self.rhos(self.Raxis,self.Zaxis)
        self.rhonormrz = (self.rhorz-self.rhomag)/(self.rhos_sep-self.rhomag)
        self.contour_convert_r = scipy.interpolate.interp1d(range(self.num_r_pts),self.R_grid, kind='linear')
        self.contour_convert_z = scipy.interpolate.interp1d(range(self.num_z_pts),self.Z_grid, kind='linear')
        
        contours=[]
        psi_list=np.linspace(0.0101,1,50)
        for i in range(len(psi_list)):
            contour=find_contours(self.rhonormrz,psi_list[i])
            contour_size=[]
            for k in range(len(contour)):
                contour_size.append(len(contour[k][:,0]))
            contour_size_index=np.argsort(contour_size)
            new_contour=contour[int(contour_size_index[-1])]
            contours.append(new_contour)
        contours_r = []
        contours_z = []
        for contour in contours[:47]:
            contours_r.append(self.contour_convert_r(contour[:, 0]).reshape(len(contour[:,0]),1))
            contours_z.append(self.contour_convert_z(contour[:, 1]).reshape(len(contour[:,1]),1))
        self.contours_rz = []
        for i in range(len(contours_r)):
            self.contours_rz.append(np.concatenate((contours_r[i],contours_z[i]),axis=1))
        self.contours_rz = list(reversed(self.contours_rz))  
        

#if __name__ == '__main__':
#    test_GEQDSK = GEQDSK("C:/Users/jakeb/OneDrive/Documents/UTK_Canvas/Research/g184016.03200")
    #plt.plot(test_GEQDSK.rhorz)
    #x,y = np.meshgrid(test_GEQDSK.R_grid,test_GEQDSK.Z_grid)
    #fig = plt.figure(figsize=(8, 6))
    #ax = fig.add_subplot(111, projection='3d')
    #surf = ax.plot_surface(x,y, test_GEQDSK.psi, cmap='viridis', edgecolor='none')
