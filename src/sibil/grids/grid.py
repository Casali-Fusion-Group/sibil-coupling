#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 12 13:24:09 2025

@author: ttaczak
"""
import numpy as np

class Grid:
    def __init__(self, grid_name):
        
        self.grid_name = grid_name
    
        self.ni = None
        self.ne = None
        self.Ti = None
        self.Te = None
        
        self.nx = None
        self.ny = None
        self.nxy = None
        
        self.rxy = None
        self.zxy = None
        self.zomp = None
        self.romp = None
        
        self.theta = None

        self.ixsep = None
        self.jyseps11 = None
        self.jyseps22 = None
        self.jomp = None
        
        self.j_pfr = None
        self.j_cfs = None

        self.psi_axis = None
        self.psixy = None
        self.psinxy = None
        
        self.hthe = None
        
        self.psin_max_pfr = None
        self.psin_min_pfr = None
        
    def summarize(self):
        names  = ["ni", "ne", "Ti", "Te"]
        values = [self.ni, self.ne, self.Ti, self.Te]
    
        print("-" * 39)
        print(f"{self.grid_name:<10} | {'Min':>12} | {'Max':>12}")
        print("-" * 39)
        
        for name, value in zip(names, values):
            min_val = np.min(value)
            max_val = np.max(value)
            min_str = smart_format(min_val)
            max_str = smart_format(max_val)
            print(f"{name:<10} | {min_str:>12} | {max_str:>12}")
        print("-" * 39)
        print()

        
def smart_format(val, sci_thresh=1e5):
    if abs(val) >= sci_thresh:
        return f"{val:.4e}"
    else:
        return f"{val:.4f}"
                
        

