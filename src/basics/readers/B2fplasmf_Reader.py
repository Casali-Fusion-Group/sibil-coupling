#!/fusion/projects/codes/pyped/python3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 10:22:23 2024

@author: jakeb
"""

import numpy as np
class B2fplasmf:
    def __init__(self,file,nx,ny,ns):
        self.file = file
        self.process_file(self.read_file(file),nx,ny,ns)
        
    def read_file(self,data):
        try:
            with open(data, 'r') as file:
                data = file.read()
        except FileNotFoundError:
            print(f"File '{self.file}' not found.")
            return None
        return data
    def process_file(self,data,nx,ny,ns):
        entries = data.split('*cf:')
        # remove version from data to make processing easier but save if needed
        thisdict = {"version": entries[0].strip().split("\n")[0]}
        self.b2fplasmf_info = thisdict["version"]
        entries.remove(entries[0])
        for entry in entries:
            # Process each entry here
            # For example, you can split the entry into lines if needed
            lines = entry.strip().split('\n')
            info = lines[0].split()
            data_type = info[0]
            len_array = int(info[1])
            var = info[2]
                        

            if data_type == "int":
                var = var.split(",")
                numbers = lines[1:][0].split()
                for i in range(len_array):
                    thisdict[var[i]] = int(numbers[i])
                    setattr(self, var[i], int(numbers[i]))

            if data_type == "real":
                numbers = []
                for line in lines[1:]:
                    line = line.split()
                    numbers.extend(line)
                for i in range(len(numbers)):
                    numbers[i] = float(numbers[i])

                numarr = np.array(numbers)

                if len_array == (nx*ny):
                    numarr = np.reshape(
                        numarr, (nx,ny), order='F')
                elif len_array == (nx*ny*ns):
                    numarr = np.reshape(
                        numarr, (nx,ny,ns), order='F')
                elif len_array == (nx*ny*2):
                    numarr = np.reshape(
                        numarr, (nx,ny,2), order='F')
                elif len_array == (nx*ny*3):
                    numarr = np.reshape(
                        numarr, (nx,ny,3), order='F')
                elif len_array == (nx*ny*4):
                    numarr = np.reshape(
                        numarr, (nx,ny,4), order='F')
                elif len_array == nx*ny*ns*2:
                    numarr = np.reshape(
                        numarr, (nx,ny,2,ns), order='F')
                elif len_array == nx*ny*ns*3:
                    numarr = np.reshape(
                        numarr, (nx,ny,3,ns), order='F')
                elif len_array == nx*ny*ns*4:
                    numarr = np.reshape(
                        numarr, (nx,ny,4,ns), order='F')
                elif len_array == nx*ny*ns*2*ns:
                    numarr = np.reshape(
                        numarr, (nx,ny,2,ns,ns), order='F')
                elif len_array < (nx)*(ny):
                    pass
                else:
                    print("Improperly sized data matrix for variable ", var)
                    break
                if var == "te" or var == "ti":
                    for i in range(len(numarr)):
                        numarr[i] = self.jtoev(numarr[i])
                setattr(self,var,numarr)
    def jtoev(self, tmp):
        return tmp * 6.2415E18
