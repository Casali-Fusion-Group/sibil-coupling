#!/fusion/projects/codes/pyped/python3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 11:36:07 2024

@author: jakeb
"""
import numpy as np
class IonizationPotential:
    def __init__(self,file):
        self.read_file(file)
    def read_file(self,filename):
        try:
            with open(filename, 'r') as file:
                data = file.read()
        except FileNotFoundError:
            print(f"File '{filename}' not found.")
            return
        data = data.replace("d","E")
        data = data.split("\n")
        data = data[:-4]
        for i in range(len(data)):
            data[i] = data[i].split()
            for num in range(len(data[i])):
                data[i][num] = float(data[i][num])
        setattr(self,"pot",data)

