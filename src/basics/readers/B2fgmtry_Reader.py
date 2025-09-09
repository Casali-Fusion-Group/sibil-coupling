#!/fusion/projects/codes/pyped/python3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:40:43 2024

@author: jakeb
"""
import numpy as np
class B2fgmtry:
    def __init__(self,filename):
        """

        Parameters
        ----------
        filename : b2fgmtry

        Returns
        -------
        self.data : All variables in b2fgmtry
        self.verts: verticies of each mesh cell that are organized 
                    lower left(ll), lower right(lr), upper left(ul), 
                    upper right(ur).

        """
        self.data = self.reader(filename)
        self.verticies(self.data)
    
    def reader(self,filename):
        variables = {}
        with open(filename,'r') as f:
            for line in f:
                if 'nx,ny' in line:
                    liner = f.readline().split()
                    variables['nx'] = int(liner[0])+2
                    setattr(self, "nx", int(liner[0])+2)
                    variables['ny'] = int(liner[-1])+2
                    setattr(self, "ny", int(liner[-1])+2)
                elif 'label' in line:
                    continue
                elif '*cf' in line:
                    info = line.split()
                    name = info[-1]
                    size = int(info[-2])
                    numtype = info[-3]
                    data=[]
                    while len(data) < size:
                        newline = next(f)
                        splits = newline.split()
                        if splits[0] == '*cf:':
                            break
                        for i in range(len(splits)):
                            if numtype == 'int':
                                data.append(int(splits[i]))
                            elif numtype == 'real':
                                data.append(float(splits[i]))
                    if size == 1:
                        data=data[0]
                    if size > 100:
                        n_nxny = int(size/(variables['nx']*variables['ny']))
                        if n_nxny == 1:
                            data=np.reshape(data,[variables['nx'],variables['ny']],order='F')
                        elif n_nxny == 2:
                            data=np.reshape(data,[variables['nx'],variables['ny'],2],order='F')
                        elif n_nxny == 3:
                            data=np.reshape(data,[variables['nx'],variables['ny'],3],order='F')
                        elif n_nxny == 4:
                            data=np.reshape(data,[variables['nx'],variables['ny'],4],order='F')    
                    variables[name] = data
                    setattr(self, name, data)
        if variables['redef_gmtry'] == 1:
            hz = np.zeros((variables['nx'],variables['ny']))
            for j in range(variables['ny']):
                for i in range(variables['nx']):
                    hz[i,j] = (np.sum(variables['ffbz'][i,j,:])/4)/variables['bb'][i,j,2]
            variables['hz'] = hz
            setattr(self,"hz",hz)

    def verticies(self,data):
        crx=getattr(self, "crx")
        cry=getattr(self, "cry")
        nx = getattr(self, "nx")
        ny = getattr(self, "ny")
        size = int(nx)*int(ny)
        var = {}

        crx = np.reshape(crx,((nx*ny),1,4),order='F')
        cry = np.reshape(cry,((nx*ny),1,4),order='F')
        ll = np.concatenate((crx[:,:,0],cry[:,:,0]), axis = 1)
        lr = np.concatenate((crx[:,:,1],cry[:,:,1]), axis = 1)
        ul = np.concatenate((crx[:,:,2],cry[:,:,2]), axis = 1)
        ur = np.concatenate((crx[:,:,3],cry[:,:,3]), axis = 1)
        var["lr"] = lr
        var["ll"] = ll
        var["ul"] = ul
        var["ur"] = ur
        mesh_verts = [[tuple(lr[i]), tuple(ll[i]), tuple(ul[i]), tuple(ur[i])]
                 for i in range(size)]
        var["verts_s"] = mesh_verts
        setattr(self, "lower_right_vertex", lr)
        setattr(self, "lower_left_vertex", ll)
        setattr(self, "upper_right_vertex", ul)
        setattr(self, "upper_left_vertex", ur)
        setattr(self, "mesh_verticies", lr)

    
