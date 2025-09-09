#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 14:06:41 2025

@author: ttaczak
"""
import numpy as np
import matplotlib.pyplot as plt
import basics


def compare_poloidal_profiles(sgrid,
                              bgrid,
                              var_name,
                              psin = None,
                              six = None,
                              bix = None,
                              regions = ["CFS", "SOL", "PFR"]):
    
    return_figs = []
    
    if var_name == "ni":
        svar = sgrid.ni
        bvar = bgrid.ni
        plot_name = "n_i"
        var_units = "m^{-3}"
        
    elif var_name == "Ti":
        svar = sgrid.Ti
        bvar = bgrid.Ti
        plot_name = "T_i"
        var_units = "eV"
        
    elif var_name == "Te":
        svar = sgrid.Te
        bvar = bgrid.Te
        plot_name = "T_e"
        var_units = "eV"
        
    spsin_max = np.max(sgrid.psinxy)
    spsin_cfs_min = np.min(sgrid.psinxy[:,sgrid.j_cfs])
    spsin_pfr_min = np.min(sgrid.psinxy[:,sgrid.j_pfr])
    
    bpsin_max = np.max(bgrid.psinxy)
    bpsin_cfs_min = np.min(bgrid.psinxy[:,bgrid.j_cfs])
    bpsin_pfr_min = np.min(bgrid.psinxy[:,bgrid.j_pfr])
        
    if psin is None:
        
        if "CFS" in regions:
            fig, ax = plt.subplots(1,1)
            fig.suptitle(r"CFR")
            
            # print(f"{psin_sol_p25:.3f}",f"{psin_sol_p75:.3f}",f"{psin_cfs_p25:.3f}",f"{psin_cfs_p25:.3f}",f"{psin_pfr_p25:.3f}",f"{psin_pfr_p25:.3f}")
            # print(ix_sol_p25,ix_sol_p75,ix_cfs_p25,ix_cfs_p75,ix_pfr_p25,ix_pfr_p75)
            
            ax.plot(sgrid.dl2d[six,sgrid.j_cfs],svar[six,sgrid.j_cfs],marker='o',c='r', markersize=5,markerfacecolor='none', label=f"{sgrid.grid_name}")
            ax.plot(bgrid.dl2d[bix,bgrid.j_cfs],bvar[bix,bgrid.j_cfs],marker='o',c='b', markersize=5,markerfacecolor='none', label=f"{bgrid.grid_name}")
            ax.set_xlabel(rf"$s(m)$")
            ax.set_ylabel(rf"${plot_name} \ \ (six={six} bix={bix}) \ [{var_units}]$")
            ax.legend()
            
            return_figs.append(fig)
            plt.show()
            plt.close()
        
        
        if "SOL" in regions:
            fig, ax = plt.subplots(1,1)
            fig.suptitle(r"SOL")
            
            # print(f"{psin_sol_p25:.3f}",f"{psin_sol_p75:.3f}",f"{psin_sol_p25:.3f}",f"{psin_sol_p25:.3f}",f"{psin_pfr_p25:.3f}",f"{psin_pfr_p25:.3f}")
            # print(ix_sol_p25,ix_sol_p75,ix_sol_p25,ix_sol_p75,ix_pfr_p25,ix_pfr_p75)
            
            ax.plot(sgrid.dl2d[six,:],svar[six,:],marker='o',c='r', markersize=5,markerfacecolor='none', label=f"{sgrid.grid_name}")
            ax.plot(bgrid.dl2d[bix,:],bvar[bix,:],marker='o',c='b', markersize=5,markerfacecolor='none', label=f"{bgrid.grid_name}")
            ax.set_xlabel(rf"$s(m)$")
            ax.set_ylabel(rf"${plot_name} \ \ (six={six} bix={bix}) \ [{var_units}]$")
            ax.legend()
            
            return_figs.append(fig)
            plt.show()
            plt.close()
        
        
        if "PFR" in regions:
            fig, ax = plt.subplots(1,1)
            fig.suptitle(r"PFR")
            
            # print(f"{psin_sol_p25:.3f}",f"{psin_sol_p75:.3f}",f"{psin_pfr_p25:.3f}",f"{psin_pfr_p25:.3f}",f"{psin_pfr_p25:.3f}",f"{psin_pfr_p25:.3f}")
            # print(ix_sol_p25,ix_sol_p75,ix_pfr_p25,ix_pfr_p75,ix_pfr_p25,ix_pfr_p75)
            
            ax.plot(sgrid.dl2d[six,sgrid.j_pfr],svar[six,sgrid.j_pfr],marker='o',c='r', markersize=5,markerfacecolor='none', label=f"{sgrid.grid_name}")
            ax.plot(bgrid.dl2d[bix,bgrid.j_pfr],bvar[bix,bgrid.j_pfr],marker='o',c='b', markersize=5,markerfacecolor='none', label=f"{bgrid.grid_name}")
            ax.set_xlabel(rf"$s(m)$")
            ax.set_ylabel(rf"${plot_name} \ \ (six={six} bix={bix}) \ [{var_units}]$")
            ax.legend()
            
            return_figs.append(fig)
            plt.show()
            plt.close()
    
    else:
        
        if psin > 1.0: # in the SOL
            spsinxy = np.mean(sgrid.psinxy, axis=1)
            six = np.argmin(np.abs(spsinxy - psin))
            
            bpsinxy = np.mean(bgrid.psinxy, axis=1)
            bix = np.argmin(np.abs(bpsinxy - psin))
            
            fig, ax = plt.subplots(1,1)
            fig.suptitle(r"SOL")
            
            # print(f"{psin_sol_p25:.3f}",f"{psin_sol_p75:.3f}",f"{psin_sol_p25:.3f}",f"{psin_sol_p25:.3f}",f"{psin_pfr_p25:.3f}",f"{psin_pfr_p25:.3f}")
            # print(ix_sol_p25,ix_sol_p75,ix_sol_p25,ix_sol_p75,ix_pfr_p25,ix_pfr_p75)
            
            # sx = sgrid.dl2d[six,:sgrid.jyseps11+1]
            # bx = bgrid.dl2d[bix,:bgrid.jyseps11+1]
            # sy = svar[six,:sgrid.jyseps11+1]
            # by = bvar[bix,:bgrid.jyseps11+1]
            # ax.plot(sx,sy,marker='o',c='r', markersize=5,markerfacecolor='none', label=f"{sgrid.grid_name}")
            # ax.plot(bx,by,marker='o',c='b', markersize=5,markerfacecolor='none', label=f"{bgrid.grid_name}")
            
            # sdl_pfr = sx[-1]
            # bdl_pfr = bx[-1]
            sx = sgrid.dl2d[six,sgrid.jyseps11+1:sgrid.jyseps22+1] # + sdl_pfr
            bx = bgrid.dl2d[bix,bgrid.jyseps11+1:bgrid.jyseps22+1] # + bdl_pfr
            sy = svar[six,sgrid.jyseps11+1:sgrid.jyseps22+1]
            by = bvar[bix,bgrid.jyseps11+1:bgrid.jyseps22+1]
            ax.plot(sx,sy,marker='o',c='r', markersize=5,markerfacecolor='none')#, label=f"{sgrid.grid_name}")
            ax.plot(bx,by,marker='o',c='b', markersize=5,markerfacecolor='none')#, label=f"{bgrid.grid_name}")
            
            # sx = sx[-1] - sdl_pfr + sgrid.dl2d[six,sgrid.jyseps22+1:] 
            # bx = bx[-1] - bdl_pfr + bgrid.dl2d[bix,bgrid.jyseps22+1:]
            # sy = svar[six,sgrid.jyseps22+1:]
            # by = bvar[bix,bgrid.jyseps22+1:]
            # ax.plot(sx,sy,marker='o',c='r', markersize=5,markerfacecolor='none')#, label=f"{sgrid.grid_name}")
            # ax.plot(bx,by,marker='o',c='b', markersize=5,markerfacecolor='none')#, label=f"{bgrid.grid_name}")
            # ax.set_xlabel(rf"$s(m)$")
            # ax.set_ylabel(rf"${plot_name} \ \ (\psi_n={psin:0.3}) \ [{var_units}]$")
            # ax.legend()
            
            return_figs.append(fig)
            plt.show()
            plt.close()
            
        elif psin > spsin_cfs_min or psin > spsin_pfr_min: #in PFR and CFS
        
            
            if psin > spsin_cfs_min and psin < spsin_max:
                
            
            
                spsinxy = np.mean(sgrid.psinxy[:,sgrid.j_cfs], axis=1)
                six = np.argmin(np.abs(spsinxy - psin))
                
                bpsinxy = np.mean(bgrid.psinxy[:,bgrid.j_cfs], axis=1)
                bix = np.argmin(np.abs(bpsinxy - psin))
                
                fig, ax = plt.subplots(1,1)
                fig.suptitle(r"CFR")
                
                
                # print(f"{psin_sol_p25:.3f}",f"{psin_sol_p75:.3f}",f"{psin_cfs_p25:.3f}",f"{psin_cfs_p25:.3f}",f"{psin_pfr_p25:.3f}",f"{psin_pfr_p25:.3f}")
                # print(ix_sol_p25,ix_sol_p75,ix_cfs_p25,ix_cfs_p75,ix_pfr_p25,ix_pfr_p75)
                
                ax.plot(sgrid.dl2d[six,sgrid.j_cfs],svar[six,sgrid.j_cfs],marker='o',c='r', markersize=5,markerfacecolor='none', label=f"{sgrid.grid_name}")
                ax.plot(bgrid.dl2d[bix,bgrid.j_cfs],bvar[bix,bgrid.j_cfs],marker='o',c='b', markersize=5,markerfacecolor='none', label=f"{bgrid.grid_name}")
                ax.set_xlabel(rf"$s(m)$")
                ax.set_ylabel(rf"${plot_name} \ \ (\psi_n={psin:0.3}) \ [{var_units}]$")
                ax.legend()
                
                return_figs.append(fig)
                plt.show()
                plt.close()
            
            
            if psin > spsin_pfr_min and psin < spsin_max:
                
            
                spsinxy = np.mean(sgrid.psinxy[:,sgrid.j_pfr], axis=1)
                six = np.argmin(np.abs(spsinxy - psin))
                
                bpsinxy = np.mean(bgrid.psinxy[:,bgrid.j_pfr], axis=1)
                bix = np.argmin(np.abs(bpsinxy - psin))
                
                fig, ax = plt.subplots(1,1)
                fig.suptitle(r"PFR")
                
                # print(f"{psin_sol_p25:.3f}",f"{psin_sol_p75:.3f}",f"{psin_pfr_p25:.3f}",f"{psin_pfr_p25:.3f}",f"{psin_pfr_p25:.3f}",f"{psin_pfr_p25:.3f}")
                # print(ix_sol_p25,ix_sol_p75,ix_pfr_p25,ix_pfr_p75,ix_pfr_p25,ix_pfr_p75)
                
                ax.plot(sgrid.dl2d[six,sgrid.j_pfr],svar[six,sgrid.j_pfr],marker='o',c='r', markersize=5,markerfacecolor='none', label=f"{sgrid.grid_name}")
                ax.plot(bgrid.dl2d[bix,bgrid.j_pfr],bvar[bix,bgrid.j_pfr],marker='o',c='b', markersize=5,markerfacecolor='none', label=f"{bgrid.grid_name}")
                ax.set_xlabel(rf"$s(m)$")
                ax.set_ylabel(rf"${plot_name} \ \ (\psi_n={psin:0.3}) \ [{var_units}]$")
                ax.legend()
                
                return_figs.append(fig)
                plt.show()
                plt.close()
    
    return return_figs

def compare_radial_profiles():
    return None