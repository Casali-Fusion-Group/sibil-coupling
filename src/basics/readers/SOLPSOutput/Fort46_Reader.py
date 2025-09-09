#!/fusion/projects/codes/pyped/python3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:45:46 2024

@author: jakeb
"""
import numpy as np

class Fort46:
    pdena = None #Atom particle density (in cm−3), dimension (NTRII,NATM)
    pdenm = None #Molecule particle density (in cm−3), dimension (NTRII,NMOL)
    pdeni = None #Test ion particle density (in cm−3), dimension (NTRII,NION)
    edena = None #Energy density carried by atoms (in eV.cm−3), dimension (NTRII,NATM)
    edenm = None #Energy density carried by molecules (in eV.cm−3), dimension (NTRII,NMOL)
    edeni = None #Energy density carried by test ions (in eV.cm−3), dimension (NTRII,NION)
    vxdena = None #X-directed momentum density carried by atoms (in g.cm−2.s−1), dimension (NTRII,NATM)
    vxdenm = None #X-directed momentum density carried by molecules (in g.cm−2.s−1), dimension (NTRII,NMOL)
    vxdeni = None #X-directed momentum density carried by test ions (in g.cm−2.s−1), dimension (NTRII,NION)
    vydena = None #Y -directed momentum density carried by atoms (in g.cm−2.s−1), dimension (NTRII,NATM)
    vydenm = None #Y -directed momentum density carried by molecules (in g.cm−2.s−1), dimension (NTRII,NMOL)
    vydeni = None #Y -directed momentum density carried by test ions (in g.cm−2.s−1), dimension (NTRII,NION)
    vzdena = None #Z-directed momentum density carried by atoms (in g.cm−2.s−1), dimension (NTRII,NATM)
    vzdenm = None #Z-directed momentum density carried by molecules (in g.cm−2.s−1), dimension (NTRII,NMOL)
    vzdeni = None #Z-directed momentum density carried by test ions (in g.cm−2.s−1), dimension (NTRII,NION)
    volume_eir = None #Triangle volumes (in cm3), dimension (NTRII)
    pux = None #X-component of the poloidal unit vector at the triangle center
    puy = None #Y-component of the poloidal unit vector at the triangle center
    pvx = None #X-component of the radial unit vector at the triangle center
    pvy = None #Y-component of the radial unit vector at the triangle center
    
    def __init__(self,file):
        self.file = self.read_file(file)
        self.data = self.process_data(self.file)
        
    def read_file(self,filename):
        try:
            with open(filename, 'r') as file:
                data = file.read()
            return data
        except FileNotFoundError:
            print(f"File '{filename}' not found.")
            return None
    
    def process_data(self,data):
        if data is None:
            return
        entries = data.split('*eirene data field')
        thisdict = {"info": entries[0].strip().split("\n")[0]}
        thisdict["ntrii"] = int(thisdict["info"].split()[0])
        thisdict["particles"] = entries[0].strip().split("\n")[2:-2]
        for i in range(len(thisdict["particles"])):
            thisdict["particles"][i] = thisdict["particles"][i].rstrip().lstrip()
        thisdict["natm"] = int(entries[0].strip().split("\n")[1].split()[0])
        thisdict["nmol"] = int(entries[0].strip().split("\n")[1].split()[1])
        thisdict["nion"] = int(entries[0].strip().split("\n")[1].split()[2])
        entries.remove(entries[0])

        for entry in entries:
            entry = entry.replace("with size", "")
            lines = entry.strip().split('\n')
            info = lines[0].split()
            var = info[0]
            if var == "volumes":
                var = "volume_eir"
            size = info[1]
            numbers = []
            for line in lines[1:]:
                line = line.split()
                numbers.extend(line)
            for i in range(len(numbers)):
                numbers[i] = float(numbers[i])

            numarr = np.array(numbers)
            # Conditions
            
            if (thisdict["ntrii"]*thisdict["natm"]) == int(size):
                numarr = np.reshape(
                    numarr, (thisdict["ntrii"], thisdict["natm"]), order='F')
                setattr(self, var, numarr)
                continue
            if (thisdict["ntrii"]*thisdict["nmol"]) == int(size):
                numarr = np.reshape(
                    numarr, (thisdict["ntrii"], thisdict["nmol"]), order='F')
                setattr(self, var, numarr)
                continue
            if (thisdict["ntrii"]*thisdict["nion"]) == int(size):
                numarr = np.reshape(
                    numarr, (thisdict["ntrii"], thisdict["nion"]), order='F')
                setattr(self, var, numarr)
                continue
            if (thisdict["ntrii"]) == int(size):
                numarr = np.reshape(
                    numarr, (thisdict["ntrii"]), order='F')
                setattr(self, var, numarr)
