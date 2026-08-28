#!/fusion/projects/codes/pyped/python3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 17:55:40 2024

@author: jakeb
"""
import numpy as np
class B2fstate:
    b2fstate_info = None
    zamin = None  #"Atom charge (max over bundle)"
    zamax = None #"Atom charge (min over bundle)"
    zn = None #"Nuclear charge"
    am = None #"Atom/ion mass"
    na = None #(-1:nx,-1:ny,0:ns-1) real*8 array. For (ix,iy,is) in (-1:nx,-1:ny,0:ns-1), na(ix,iy,is) specifies the density of atomic species (is) on the (ix,iy) cell. It will hold that 0.lt.na(,,).
    ne = None #Electron Density
    ua = None #(-1:nx,-1:ny,0:ns-1) real*8 array. For (ix,iy,is) in (-1:nx,-1:ny,0:ns-1), ua(ix,iy,is) specifies the parallel velocity of atomic species (is) on the (ix,iy) cell. The sign of the parallel velocity indicates whether it is flowing in the same direction as the local poloidal magnetic field.
    uadia = None #"Drift velocities in diamagnetic and radial directions"
    te = None #(-1:nx,-1:ny) real*8 array. For (ix,iy) in (-1:nx,-1:ny), te(ix,iy) specifies the electron temperature on the (ix,iy) cell. It will hold that 0.lt.te(,).
    ti = None #(-1:nx,-1:ny) real*8 array.For (ix,iy) in (-1:nx,-1:ny), ti(ix,iy) specifies the all atomtemperature on the (ix,iy) cell. It will hold that 0.lt.ti(,).
    po = None #(-1:nx,-1:ny) real*8 array. For (ix,iy) in (-1:nx,-1:ny), po(ix,iy) specifies the electric potential on the (ix,iy) cell.
    
    
    #For all the fluxes below, the sign convention is as follows.
    #Fluxes between poloidal neighbors are positive if directed in the direction of increasing poloidal (ix) index.
    #Fluxes between radial neighbors are positive if directed in the direction of increasing radial (iy) index.
    
    #For (ix,iy,is) in (0:nx,-1:ny,0:ns-1), fnax(ix,iy,is) specifies the flux of atoms of species (is) through the face between the (ix,iy) cell and its left neighbor.
    #For ix.eq.-1, fnax(ix,-1:ny,0:ns-1) will hold 0.
    #For (ix,iy,is) in (-1:nx,0:ny,0:ns-1), fnay(ix,iy,is) specifies the flux of atoms of species (is) through the face between the (ix,iy) cell and its bottom neighbor.
    #For iy.eq.-1, fnay(-1:nx,iy,0:ns-1) will hold 0.
    fna = None #(-1:nx,-1:ny,0:1,0:ns-1) real*8 array. (Let fnax(,,)=fna(,,0,) and fnay(,,)=fna(,,1,).)
   
    
   #For (ix,iy) in (0:nx,-1:ny), fhex(ix,iy) specifies the electron heat flux through the face between the (ix,iy) cell and its left neighbor. For ix.eq.-1, fhex(ix,-1:ny) will hold 0.
   #For (ix,iy) in (-1:nx,0:ny), fhey(ix,iy) specifies the electron heat flux through the face between the (ix,iy) cell and its bottom neighbor. For iy.eq.-1, fhey(-1:nx,iy) will hold 0.
    fhe = None #(-1:nx,-1:ny,0:1) real*8 array. (Let fhex(,)=fhe(,,0) and fhey(,)=fhe(,,1).)
    
    #For (ix,iy) in (0:nx,-1:ny), fhix(ix,iy) specifies the all atom heat flux through the face between the (ix,iy) cell and its left neighbor. For ix.eq.-1, fhix(ix,-1:ny) will hold 0.
    #For (ix,iy) in (-1:nx,0:ny), fhiy(ix,iy) specifies the all atom heat flux through the face between the (ix,iy) cell and its bottom neighbor. For iy.eq.-1, fhiy(-1:nx,iy) will hold 0.
    fhi = None #(-1:nx,-1:ny,0:1) real*8 array. (Let fhix(,)=fhi(,,0) and fhiy(,)=fhi(,,1).)
    
    #For (ix,iy) in (0:nx,-1:ny), fchx(ix,iy) specifies the electric current through the face between the (ix,iy) cell and its left neighbor. For ix.eq.-1, fchx(ix,-1:ny) will hold 0.
    #For (ix,iy) in (-1:nx,0:ny), fchy(ix,iy) specifies the electric current through the face between the (ix,iy) cell and its bottom neighbor. For iy.eq.-1, fchy(-1:nx,iy) will hold 0.
    fch = None #(-1:nx,-1:ny,0:1) real*8 array. (Let fchx(,)=fch(,,0) and fchy(,)=fch(,,1).)
    
    fch_32 = None #"Convective poloidal/radial current"
    fch_52 = None #"Conductive poloidal/radial current"
    kinrgy = None #Total Kinetic Energy
    time = None #Simulation Timestep
    fch_p = None #"Parallel current"
    
    #For (ix,iy,is) in (0:nx,-1:ny,0:ns-1), fnax_mdf(ix,iy,is) specifies the modified flux of atoms of species (is) through the face between the (ix,iy) cell and its left neighbor.
    #For ix.eq.-1, fnax_mdf(ix,-1:ny,0:ns-1) will hold 0.
    #For (ix,iy,is) in (-1:nx,0:ny,0:ns-1), fnay_mdf(ix,iy,is) specifies the modified flux of atoms of species (is) through the face between the (ix,iy) cell and its bottom neighbor.
    #For iy.eq.-1, fnay_mdf(-1:nx,iy,0:ns-1) will hold 0. fna_mdf is equal to fna in runs without drifts.
    fna_mdf = None #(-1:nx,-1:ny,0:1,0:ns-1) real*8 array. (Let fnax_mdf(,,)=fna_mdf(,,0,) and fnay_mdf(,,)=fna_mdf(,,1,).)
    
    #For (ix,iy) in (0:nx,-1:ny), fhex_mdf(ix,iy) specifies the modified electron heat flux through the face between the (ix,iy) cell and its left neighbor. For ix.eq.-1, fhex_mdf(ix,-1:ny) will hold 0.
    #For (ix,iy) in (-1:nx,0:ny), fhey_mdf(ix,iy) specifies the modified electron heat flux through the face between the (ix,iy) cell and its bottom neighbor. For iy.eq.-1, fhey_mdf(-1:nx,iy) will hold 0. fhe_mdf is equal to fhe in runs without drifts.
    fhe_mdf = None #(-1:nx,-1:ny,0:1) real*8 array. (Let fhex_mdf(,)=fhe_mdf(,,0) and fhey_mdf(,)=fhe_mdf(,,1).)
    
    #For (ix,iy) in (0:nx,-1:ny), fhix_mdf(ix,iy) specifies the  modified all atom heat flux through the face between the (ix,iy) cell and its left neighbor. For ix.eq.-1, fhix_mdf(ix,-1:ny) will hold 0.
    #For (ix,iy) in (-1:nx,0:ny), fhiy_mdf(ix,iy) specifies the modified all atom heat flux through the face between the (ix,iy) cell and its bottom neighbor. For iy.eq.-1, fhiy_mdf(-1:nx,iy) will hold 0.
    fhi_mdf = None #(-1:nx,-1:ny,0:1) real*8 array. (Let fhix_mdf(,)=fhi_mdf(,,0) and fhiy_mdf(,)=fhi_mdf(,,1).)
    
    #For (ix,iy,is) in (0:nx,-1:ny,0:ns-1), fna_fcorx(ix,iy,is) specifies the flux of atoms of species (is) for calculating momentum transport through the face between the (ix,iy) cell and its left neighbor.
    #For ix.eq.-1, fna_fcorx(ix,-1:ny,0:ns-1) will hold 0.
    #For (ix,iy,is) in (-1:nx,0:ny,0:ns-1), fna_fcory(ix,iy,is) specifies the flux of atoms of species (is) for calculating momentum transport through the face between the (ix,iy) cell and its bottom neighbor.
    #For iy.eq.-1, fna_fcory(-1:nx,iy,0:ns-1) will hold 0. fna_fcor takes into account the hz toroidal length derivative and the Coriolis force.
    fna_fcor = None #(-1:nx,-1:ny,0:1,0:ns-1) real*8 array. (Let fna_fcorx(,,)=fna_fcor(,,0,) and fna_fcory(,,)=fna_fcor(,,1,).)
    
    fna_nodrift = None #"Polodial/Radial particle flux (not including drift contributions)"
    fna_he = None #"Poloidal/Radial particle flux (for electron heat equation)"
    fnaPSch = None #"Poloidal/Radial Pfirsch-Schlueter particle flux" 
    fhePSch = None #"Poloidal/Radial electron Pfirsch-Schlueter energy flux"
    fhiPSch = None #"Poloidal/Radial ion Pfirsch-Schlueter energy flux"
    fna_eir = None #"Poloidal/Radial particle flux (passed to Eirene)"
    fne_eir = None #"Polodial/Radial electron flux (passed to Eirene)"
    fhe_eir = None #Poloidal/Radial electron energy flux (passed to Eirene)"
    fhi_eir = None #"Poloidal/Radial ion energy flux (passed to Eirene)"
    fna_32 = None #"Poloidal/Radial particle flux (3/2 piece)"
    fna_52 = None #"Poloidal/Radial particle flux (5/2 piece)"
    fni_32 = None #"Poloidal/Radial total ion flux (3/2 piece)"
    fni_52 = None #"Poloidal/Radial total ion flux (5/2 piece)"
    fne_32 = None #"Polodial/Radial electron flux (3/2 piece)"
    fne_52 = None #"Polodial/Radial electron flux (5/2 piece)"
    fchdia = None #"Poloidal/Radial diamagnetic current"
    fchin = None #"Poloidal/Radial ion-neutral current"
    fchvispar = None #"Poloidal/Radial parallel viscosity current"
    fchvisper = None #"Poloidal/Radial perpendicular viscosity current"
    fchvisq = None #"Poloidal/Radial heat viscosity current"
    fchinert = None #"Poloidal/Radial ion inertial current"
    vaecrb = None #"ExB drift velocities in diamagnetic and radial directions"
    vadia = None #"Divergent diamagnetic drift velocities in diamagnetic and radial directions"
    wadia = None #"Diamagnetic drift velocities in diamagnetic and radial directions"
    veecrb = None #"Electron ExB drift velocities in diamagnetic and radial directions"
    vedia = None #"Divergent diamagnetic drift electron velocities in diamagnetic and radial directions"
    floe_noc = None #"Poloidal/Radial electron energy flow"
    floi_noc = None #"Poloidal/Radial ion energy flow"

    def __init__(self,data):
        
        try:
            with open(data, 'r') as file:
                data = file.read()
        except FileNotFoundError:
            print(f"File '{self.filename}' not found.")
            return None
        
        entries = data.split('*cf:')
        # remove version from data to make processing easier but save if needed
        thisdict = {"version": entries[0].strip().split("\n")[0]}
        self.b2fstate_info = thisdict["version"]
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
                    if var[i] == "ns":
                        thisdict[var[i]] = int(numbers[i])
                        setattr(self, var[i], int(numbers[i]))
                    else:
                        thisdict[var[i]] = int(numbers[i])+2
                        setattr(self, var[i], int(numbers[i])+2) #Two Guard Cells


            if data_type == "char":
                thisdict[var] = lines[1:][0]
                stuff = lines[1:][0].split()
                thisdict["date"] = stuff[4]
                thisdict["time-mil"] = stuff[5]

            if data_type == "real":
                numbers = []
                for line in lines[1:]:
                    line = line.split()
                    numbers.extend(line)
                for i in range(len(numbers)):
                    numbers[i] = float(numbers[i])
            
                numarr = np.array(numbers)
                if len_array == (thisdict["nx"])*(thisdict["ny"]):
                    numarr = np.reshape(
                        numarr, (thisdict["nx"], thisdict["ny"]), order='F')
                elif len_array == (thisdict["nx"])*(thisdict["ny"])*(thisdict["ns"]):
                    numarr = np.reshape(
                        numarr, (thisdict["nx"], thisdict["ny"], thisdict["ns"]), order='F')
                elif len_array == (thisdict["nx"])*(thisdict["ny"])*2:
                    numarr = np.reshape(
                        numarr, (thisdict["nx"], thisdict["ny"], 2), order='F')
                elif len_array == (thisdict["nx"])*(thisdict["ny"])*(thisdict["ns"]*2):
                    numarr = np.reshape(
                        numarr, (thisdict["nx"], thisdict["ny"], 2, thisdict["ns"]), order='F')
                elif len_array < (thisdict["nx"])*(thisdict["ny"]):
                    pass
                else:
                    print("Improperly sized data matrix for variable ", var)
                    break
                if var == "te" or var == "ti":
                    for i in range(len(numarr)):
                        numarr[i] = self.jtoev(numarr[i])
                setattr(self, var, numarr)
    def jtoev(self, tmp):
        return tmp * 6.2415E18
