#!/fusion/projects/codes/pyped/python3/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 11:32:54 2024

@author: jakeb
"""
import numpy as np

class Fort44:
    fort44_info = None
    natm = None #Number of atom species declared in the Eirene input file
    ncl = None #Number of segments for resolved tallies on non-standard surfaces
    nion = None #Number of ion species declared in the Eirene input file
    nlim = None #Number of ”additional surfaces” (physical walls) declared in block 3B of the Eirene input file
    nmol = None #Number of molecule species declared in the Eirene input file
    npls = None #Number of plasma background species declared in the Eirene input file (including BGK species)
    nstra = None #Number of source strata declared in the Eirene input file (Stratum 0 is the sum over strata)
    nsts = None #Number of ”non-standard surfaces” (plasma boundaries) declared in block 3A of the Eirene input file
    nx = None #Number of poloidal cells in the Eirene computational grid
    ny = None #Number of radial cells in the Eirene computational grid
    particles = None #Species Present
    wall_geometry = None #X-Y coordinates of wall elements
    dab2 = None #Atom density (m−3), dimension (NDX,NDY,NATM)
    tab2 = None #Atom temperature (eV ), dimension (NDX,NDY,NATM)
    dmb2 = None #Molecule density (m−3), dimension (NDX,NDY,NMOL)
    tmb2 = None #Molecule temperature (eV ), dimension (NDX,NDY,NMOL)
    dib2 = None #Test ion density (m−3), dimension (NDX,NDY,NION)
    tib2 = None #Test ion temperature (eV ), dimension (NDX,NDY,NION)
    rfluxa = None #Radial flux density of atoms (m−2.s−1), dimension (NDX,NDY,NATM)
    rfluxm = None #Radial flux density of molecules (m−2.s−1), dimension (NDX,NDY,NMOL)
    pfluxa = None #Poloidal flux density of atoms (m−2.s−1), dimension (NDX,NDY,NATM)
    pfluxm = None #Poloidal flux density of molecules (m−2.s−1), dimension (NDX,NDY,NMOL)
    refluxa = None #Radial energy flux density carried by atoms (W.m−2), dimension (NDX,NDY,NATM)
    refluxm = None #Radial energy flux density carried by molecules (W.m−2), dimension (NDX,NDY,NMOL)
    pefluxa = None #Poloidal energy flux density carried by atoms (W.m−2), dimension (NDX,NDY,NATM)
    pefluxm = None #Poloidal energy flux density carried by molecules (W.m−2), dimension (NDX,NDY,NMOL)
    emiss = None #Hα emissivity due to atoms (photons.m−3.s−1), dimension (NDX,NDY)
    emissmol = None #Hα emissivity due to molecules and molecular ions (photons.m−3.s−1), dimension (NDX,NDY)
    srcml = None #Molecule particle source (if the header is older than 20240627, then A.cm−3, otherwise molecules.s−1), dimension (NDX,NDY,NMOL)
    edissml = None #Energy spent for dissociating hydrogenic molecules (if the header is older than 20240627, then W.cm−3, otherwise W), dimension (NDX,NDY,NMOL)
    wldnek = None #Net kinetic energy deposited by neutrals (W), total over strata, dimension (NLIM+NSTS)
    wldnep = None #Energy released by recombination of neutrals and ions into molecules (W), total over strata, dimension(NLIM+NSTS)
    wldna = None #Flux of atoms impinging on surface (particles.s−1), total over strata, dimension (NLIM+NSTS,NATM)
    ewlda = None #Average energy of impinging atoms on surface (eV ), total over strata, dimension (NLIM+NSTS,NATM)
    wldnm = None #Flux of molecules impinging on surface (particles.s−1), total over strata, dimension (NLIM+NSTS,NMOL)
    ewldm = None #Average energy of impinging molecules on surface (eV ), total over strata, dimension (NLIM+NSTS,NMOL)
    p1 = None #Endpoint of surface (X coordinates, in m), total over strata, dimension (NLIM)
    p2 = None #Endpoint of surface ( Y coordinates, in m), total over strata, dimension (NLIM)
    wldra = None #Flux of reflected atoms from surface (particles.s−1), total over strata, dimension (NLIM+NSTS,NATM)
    wldrm = None #Flux of reflected molecules from surface (particles.s−1), total over strata, dimension (NLIM+NSTS,NMOL)
    wldneki = None #Net kinetic energy deposited by neutrals (W), dimension (NLIM+NSTS), each strata
    wldnepi = None #Energy released by recombination of neutrals and ions into molecules (W), dimension(NLIM+NSTS), each strata
    wldnai = None #Flux of atoms impinging on surface (particles.s−1), dimension (NLIM+NSTS,NATM), each strata
    ewldai = None #Average energy of impinging atoms on surface (eV ), dimension (NLIM+NSTS,NATM), each strata
    wldnmi = None #Flux of molecules impinging on surface (particles.s−1), dimension (NLIM+NSTS,NMOL), each strata
    ewldmi = None #Average energy of impinging molecules on surface (eV ), dimension (NLIM+NSTS,NMOL), each strata
    wldrai = None #Flux of reflected atoms from surface (particles.s−1), dimension (NLIM+NSTS,NATM), each strata
    wldrmi = None #Flux of reflected molecules from surface (particles.s−1), dimension (NLIM+NSTS,NMOL), each strata
    wldpp = None #Flux of plasma ions impinging on surface (particles.s−1), total over strata, dimension (NLIM+NSTS,NPLS)
    wldpa = None #Net flux of atoms emitted from surface (particles.s−1), total over strata, dimension (NLIM+NSTS,NATM)
    wldpm = None #Net flux of molecules emitted from surface (particles.s−1), total over strata, dimension (NLIM+NSTS,NMOL)
    wldpeb = None #Power carried by particles emitted from surface (W), total over strata, dimension (NLIM+NSTS)
    wldspt = None #Flux of sputtered wall material (particles.s−1), total over strata, dimension (NLIM+NSTS)
    wldspta = None #Flux of sputtered wall material per atom (particles.s−1), total over strata, dimension (NLIM+NSTS,NATM)
    wldsptm = None #Flux of sputtered wall material per molecule (particles.s−1), total over strata, dimension(NLIM+NSTS,NMOL)
    wldppi = None #Add strata Flux of plasma ions impinging on surface (particles.s−1), dimension (NLIM+NSTS,NPLS)
    wldpai = None #Add strata Net flux of atoms emitted from surface (particles.s−1), dimension (NLIM+NSTS,NATM)
    wldpmi = None #Add strata Net flux of molecules emitted from surface (particles.s−1), dimension (NLIM+NSTS,NMOL)
    wldpebi = None #Add strata Power carried by particles emitted from surface (W), dimension (NLIM+NSTS)
    wldspti = None #Add strata Flux of sputtered wall material (particles.s−1), dimension (NLIM+NSTS)
    wldsptam= None #Add Strata Flux of sputtered wall material per atom (particles.s−1), dimension (NLIM+NSTS,NATM)
    wldsptmm = None #Add strata Flux of sputtered wall material per molecule (particles.s−1), dimension (NLIM+NSTS,NMOL)
    isrftype = None #ILIIN surface type variable in Eirene, dimension (NLIM+NSTS)
    wlarea = None #Surface area (m2), dimension (NLIM+NSTS)
    wlabsrp_A = None #Absorption rate for atoms, dimension (NATM, NLIM+NSTS)
    wlabsrp_M = None #Absorption rate for molecules, dimension (NMOL, NLIM+NSTS)
    wlabsrp_I = None #Absorption rate for test ions, dimension (NION, NLIM+NSTS)
    wlabsrp_P = None #Absorption rate for plasma ions, dimension (NPLS, NLIM+NSTS)
    wlpump_A = None #Pumped flux per atom (particles.s−1), dimension (NATM, NLIM+NSTS)
    wlpump_M = None #Pumped flux per molecule (particles.s−1), dimension (NMOL, NLIM+NSTS)
    wlpump_I = None #Pumped flux per test ion (particles.s−1), dimension (NION, NLIM+NSTS)
    wlpump_P = None #Pumped flux per plasma ion (particles.s−1), dimension (NPLS, NLIM+NSTS)
    eneutrad = None #Radiation rate due to atoms (W), dimension (NDX,NDY,NATM)
    emolrad = None #Radiation rate due to molecules (W), dimension (NDX,NDY,NMOL)
    eionrad = None #Radiation rate due to test ions (W), dimension (NDX,NDY,NION)
    eirdiag = None #Indices for segments on resolved non-standard surfaces, dimension (5× NSTS +1)
    sarea_res = None #Surface area of surface segment (m2), dimension (NCL)
    wldna_res = None #Flux of atoms impinging on surface segment (particles.m−2.s−1), dimension (NATM, NCL)
    wldnm_res = None #Flux of molecules impinging on surface segment (particles.m−2.s−1), dimension (NMOL, NCL)
    ewlda_res = None #Energy flux carried by impinging atoms on surface segment (W.m−2), dimension (NATM, NCL)
    ewldm_res = None #Energy flux carried by impinging molecules on surface segment (W.m−2), dimension (NMOL, NCL)
    ewldea_res = None #Energy flux carried by emitted atoms from surface segment (W.m−2), dimension (NATM, NCL)
    ewldem_res = None #Energy flux carried by emitted molecules from surface segment (W.m−2), dimension (NMOL, NCL)
    ewldrp_res = None #Kinetic energy flux carried by recycling atoms and molecules (W.m−2), dimension (NCL)
    ewldmr_res = None #Energy flux due to recombination of atoms and atomic ions into molecules (W.m−2), dimension (NMOL,NCL)
    wldspt_res = None #Flux of sputtered wall material (A.m−2), dimension (NCL)
    wldspta_res = None #Flux of sputtered wall material per atom (A.m−2), dimension (NCL, NATM)
    wldsptm_res = None #Flux of sputtered wall material per molecule (A.m−2), dimension (NCL, NMOL)
    wlpump_resA = None #Pumped flux per atom (A.m−2), dimension (NCL, NATM)
    wlpump_resM = None #Pumped flux per molecule (A.m−2), dimension (NCL, NMOL)
    wlpump_resI = None #Pumped flux per test ion (A.m−2), dimension (NCL, NION)
    wlpump_resP = None #Pumped flux per plasma ion (A.m−2), dimension (NCL, NPLS)
    ewldt_res = None #Total wall power loading from Eirene particles (W.m−2), dimension (NCL)
    pdena_int = None #Integral number of atoms over the entire Eirene computational grid, dimension (NATM, 0:NSTRA)
    pdenm_int = None #Integral number of molecules over the entire Eirene computational grid, dimension (NMOL, 0:NSTRA)
    pdeni_int = None #Integral number of test ions over the entire Eirene computational grid, dimension (NION, 0:NSTRA)
    pdena_int_b2 = None #Integral number of atoms over the B2.5 computational grid, dimension (NATM, 0:NSTRA)
    pdenm_int_b2 = None #Integral number of molecules over the B2.5 computational grid, dimension (NMOL, 0:NSTRA)
    pdeni_int_b2 = None #Integral number of test ions over the B2.5 computational grid, dimension (NION, 0:NSTRA)
    edena_int = None #Integral energy carried by atoms over the entire Eirene computational grid (J), dimension (NATM,0:NSTRA)
    edenm_int = None #Integral energy carried by molecules over the entire Eirene computational grid (J), dimension (NMOL,0:NSTRA)
    edeni_int = None #Integral energy carried by test ions over the entire Eirene computational grid (J), dimension (NION,0:NSTRA)
    edena_int_b2 = None #Integral energy carried by atoms over the B2.5 computational grid (J), dimension (NATM, 0:NSTRA)
    edenm_int_b2 = None #Integral energy carried by molecules over the B2.5 computational grid (J), dimension (NMOL,0:NSTRA)
    edeni_int_b2 = None #Integral energy carried by test ions over the B2.5 computational grid (J), dimension (NION, 0:NSTRA)
    
    
            
    def __init__(self,fort44,inputdat):
        self.process_data(self.read_fort44(fort44), self.read_input(inputdat))
    
    def read_fort44(self,filename):
        try:
            with open(filename, 'r') as file:
                data = file.read()
            return data
        except FileNotFoundError:
            print(f"File '{filename}' not found.")
            return None
    
    def read_input(self,inp):
        selected_data = []
        for line in self.report_iterator(inp):
            selected_data.append(line.replace('\\', '').strip())
        return selected_data
    
    def report_iterator(self,inputname):
        with open(inputname, 'r') as inputname:
            yield_next_line = False
            for line in inputname:
                if yield_next_line:
                    yield_next_line = False
                    yield line
                if "*** 3a." in line:
                    yield_next_line = True
                if "*** 3b." in line:
                    yield_next_line = True
                if "** 5a." in line:
                    yield_next_line = True
                if "*** 7." in line:
                    yield_next_line = True

    def process_data(self, data, selected_data):
        if data is None:
            return

        entries = data.split('*eirene data field')
        self.fort44_info = entries[0].strip().split("\n")
        self.nx = int(self.fort44_info[0].split()[0])
        self.ny = int(self.fort44_info[0].split()[1])
        self.particles = self.fort44_info[2:-2]
        for i in range(len(self.particles)):
           self.particles[i] = self.particles[i].rstrip().lstrip()
        self.natm = int(entries[0].strip().split("\n")[1].split()[0])
        self.nmol = int(entries[0].strip().split("\n")[1].split()[1])
        self.nion = int(entries[0].strip().split("\n")[1].split()[2])
        self.nlim = int(selected_data[1])
        self.nsts = int(selected_data[0])
        self.npls = int(selected_data[2])-4
        self.nstra = int(selected_data[3])+2
        self.wldneki = np.zeros((self.nstra,self.nlim+self.nsts+1,1))
        self.wldnepi = np.zeros((self.nstra,self.nlim+self.nsts+1,1))
        self.wldnai = np.zeros((self.nstra,(self.nlim+self.nsts+1),self.natm))
        self.ewldai = np.zeros((self.nstra,(self.nlim+self.nsts+1),self.natm))
        self.wldnmi = np.zeros((self.nstra,(self.nlim+self.nsts+1),self.nmol))
        self.ewldmi = np.zeros((self.nstra,(self.nlim+self.nsts+1),self.nmol))
        self.wldrai = np.zeros((self.nstra,(self.nlim+self.nsts+1),self.natm)) 
        self.wldrmi = np.zeros((self.nstra,(self.nlim+self.nsts+1),self.nmol)) 
        self.wldppi = np.zeros((self.nstra,(self.nlim+self.nsts+1),self.npls)) 
        self.wldpai = np.zeros((self.nstra,(self.nlim+self.nsts+1),self.natm))
        self.wldpmi = np.zeros((self.nstra,(self.nlim+self.nsts+1),self.nmol))
        self.wldpebi = np.zeros((self.nstra,(self.nlim+self.nsts+1),1))
        self.wldspti = np.zeros((self.nstra,(self.nlim+self.nsts+1),1))
        self.wldsptam = np.zeros((self.nstra,(self.nlim+self.nsts+1),self.natm))
        self.wldsptmm = np.zeros((self.nstra,(self.nlim+self.nsts+1),self.nmol))


        
        entries.remove(entries[0])
        for entry in entries:
            entry = entry.replace("with size", "")
            lines = entry.strip().split('\n')
            info = lines[0].split()
            if len(info) > 2:
                info[0] = info[0]+info[1]
                info[1] = info[2]
                info.pop(2)
            var = info[0]
            size = info[1]
            numbers = []
            if var == "eirdiag":
                for line in lines[1:]:
                    line = line.split()
                    numbers.extend(line)
                numbers = [float(num) if num.replace('-', '').isdigit() else num for num in numbers]
                self.ncl = int(max(numbers))
        for entry in entries:
            entry = entry.replace("with size", "")
            lines = entry.strip().split('\n')
            info = lines[0].split()
            if len(info) > 2:
                info[0] = info[0]+info[1]
                info[1] = info[2]
                info.pop(2)
            var = info[0]
            size = int(info[1])
            numbers = []
            for line in lines[1:]:
                line = line.split()
                numbers.extend(line)
            bad_i = []

            # Unique Cases
            for i in range(len(numbers)):
                try:
                    numbers[i] = float(numbers[i])
                except:
                    bad_i.append(i)
                    continue
            for i in bad_i:
                del numbers[0]
            bad_i = []
            if "edissml" in var:
                for i in range(3):
                    del numbers[-1]
            numarr = np.array(numbers)
            # Conditions
            
            if hasattr(self,var):
                if (self.nx*self.ny*self.natm) == int(size):
                    numarr = np.reshape(
                        numarr, (self.nx, self.ny, self.natm), order='F')
                    setattr(self, var, numarr)
                    continue
                # (ndx,ndx,nmol)
                if (self.nx*self.ny*self.nmol) == int(size):
                    numarr = np.reshape(
                        numarr, (self.nx, self.ny, self.nmol), order='F')
                    setattr(self, var, numarr)
                    continue
                # (ndx,ndy,nion)
                if (self.nx*self.ny*self.nion) == int(size):
                    numarr = np.reshape(
                        numarr, (self.nx, self.ny, self.nion), order='F')
                    setattr(self, var, numarr)
                    continue
                # (ndx,ndy)
                if (self.nx*self.ny) == int(size):
                    numarr = np.reshape(
                        numarr, (self.nx, self.ny), order='F')
                    setattr(self, var, numarr)
                    continue
                # (nlim+nsts)
                if(self.nlim+self.nsts+1) == int(size):
                    numarr = np.reshape(
                        numarr, (self.nlim+self.nsts+1), order='F')
                    setattr(self, var, numarr)
                    continue
                # (NLIM+NSTS,NATM) or (NATM, NLIM+NSTS)
                if ((self.nlim+self.nsts+1)*self.natm) == int(size):
                    if "wlabsrp" in var or "wlpump" in var:
                        # (natm,nlim+nsts+1)
                        numarr = np.reshape(
                            numarr, (self.natm, self.nlim+self.nsts+1), order='F')
                    else:
                        # (nlim+nsts+1,natm)
                        numarr = np.reshape(
                            numarr, (self.nlim+self.nsts+1, self.natm), order='F')
                    setattr(self, var, numarr)
                    continue
                #(NLIM+NSTS,NMOL)    or  (NMOL, NLIM+NSTS)
                if ((self.nlim+self.nsts+1)*self.nmol) == int(size):
                    if "wlabsrp" in var or "wlpump" in var:
                        # (nmol,nlim+nsts+1)
                        numarr = np.reshape(
                            numarr, (self.nmol, self.nlim+self.nsts+1), order='F')
                    else:
                        numarr = np.reshape(
                            numarr, (self.nlim+self.nsts+1, self.nmol), order='F')
                    setattr(self, var, numarr)
                    continue
                # (NION, NLIM+NSTS)
                if ((self.nlim+self.nsts+1)*self.nion) == int(size):
                    if "wlabsrp" in var or "wlpump" in var:
                        # (nion,nlim+nsts+1)
                        numarr = np.reshape(
                            numarr, (self.nion, self.nlim+self.nsts+1), order='F')
                    else:
                        numarr = np.reshape(
                            numarr, (self.nlim+self.nsts+1, self.nion), order='F')
                    setattr(self, var, numarr)
                    continue
                # (NLIM+NSTS,NPLS) or (NPLS, NLIM+NSTS)
                if ((self.nlim+self.nsts+1)*self.npls) == int(size):
                    if "wlabsrp" in var or "wlpump" in var:
                        # (npls,nlim+nsts+1)
                        numarr = np.reshape(
                            numarr, (self.npls, self.nlim+self.nsts+1), order='F')
                    else:
                        numarr = np.reshape(
                            numarr, (self.nlim+self.nsts+1, self.npls), order='F')
                    setattr(self, var, numarr)
                    continue
                   #  (NATM, NCL)  or (NCL, NATM)
                if (self.natm*(self.ncl)) == int(size):
                    if "wldsp" in var or "wlpump_res" in var:
                        # (npls,nlim+nsts+1)
                        numarr = np.reshape(
                            numarr, (self.ncl, self.natm), order='F')
                    else:
                        numarr = np.reshape(
                            numarr, (self.natm, self.ncl), order='F')
                    setattr(self, var, numarr)
                    continue
                #  (NMOL, NCL)  or  (NCL, NMOL)
                if (self.nmol*(self.ncl)) == int(size):
                    if "wldsp" in var or "wlpump_res" in var:
                        # (npls,nlim+nsts+1)
                        numarr = np.reshape(
                            numarr, (self.ncl, self.nmol), order='F')
                    else:
                        numarr = np.reshape(
                            numarr, (self.nmol, self.ncl), order='F')
                    setattr(self, var, numarr)
                    continue
                # (NCL, NION)
                if (self.nion*(self.ncl)) == int(size):
                    if "wldsp" in var or "wlpump_res" in var:
                        # (npls,nlim+nsts+1)
                        numarr = np.reshape(
                            numarr, (self.ncl, self.nion), order='F')
                    else:
                        numarr = np.reshape(
                            numarr, (self.nion, self.ncl), order='F')
                    print(var)
                    setattr(self, var, numarr)
                    continue
                #(NCL, NPLS)
                if (self.ncl*self.npls) == int(size):
                    if "wlabsrp" in var or "wlpump" in var:
                        # (npls,nlim+nsts+1)
                        numarr = np.reshape(
                            numarr, (self.ncl, self.npls), order='F')
                    else:
                        numarr = np.reshape(
                            numarr, (self.npls, self.ncl), order='F')
                    setattr(self, var, numarr)
                    continue
                # (NCL)
                if((self.ncl)) == int(size):
                    numarr = np.reshape(numarr, (self.ncl), order='F')
                    setattr(self, var, numarr)
                    continue
                if self.natm*self.nstra == int(size):
                    numarr = np.reshape(
                        numarr, (self.natm, self.nstra), order='F')
                    setattr(self, var, numarr)
                    continue
                if self.nmol*self.nstra == int(size):
                    numarr = np.reshape(
                        numarr, (self.nmol, self.nstra), order='F')
                    setattr(self, var, numarr)
                    continue
                if self.nion*self.nstra == int(size):
                    numarr = np.reshape(
                        numarr, (self.nion, self.nstra), order='F')
                    setattr(self, var, numarr)
                    continue
                if var == "wall_geometry":
                    numarr = np.reshape(numarr,(int(size/2),2))
                    setattr(self, var, numarr)
                    continue
                if var == "eirdiag":
                    setattr(self, var, numarr)
                    continue
                if (self.nlim+self.nsts+5) == int(size):
                    setattr(self, var, numarr)
                    continue
                if self.natm+self.nstra == int(size):
                    setattr(self, var, numarr)
            
            part1 = var[0:var.index('(')]
            part2 = (var[var.index('(') + 1:-1])
            try:
                part2 = int(part2)
            except:
                part2 = part2
            
            if ((self.nlim+self.nsts+1)) == int(size):
                if "wlabsrp" in var or "wlpump" in var:
                    # (nmol,nlim+nsts+1)
                    numarr = np.reshape(
                        numarr, (self.nmol, self.nlim+self.nsts+1), order='F')
                else:
                    numarr = np.reshape(
                        numarr, (self.nlim+self.nsts+1, self.nmol), order='F')
                if part1+"i" == "wldneki":
                    self.wldneki[part2] = numarr
                    continue
                if part1+"i" == "wldnepi":
                    self.wldnepi[part2] = numarr
                    continue
                if part1+"i" == "wldpebi":
                    self.wldpebi[part2] = numarr
                    continue
                if part1+"i" == "wldspti":
                    self.wldspti[part2] = numarr
                    continue
            if ((self.nlim+self.nsts+1)*self.natm) == int(size):
                if "wlabsrp" in var or "wlpump" in var:
                    # (natm,nlim+nsts+1)
                    numarr = np.reshape(
                        numarr, (self.natm, self.nlim+self.nsts+1), order='F')
                else:
                    # (nlim+nsts+1,natm)
                    numarr = np.reshape(
                        numarr, (self.nlim+self.nsts+1, self.natm), order='F')
                if part1+"i" == "wldnai":
                    self.wldnai[part2] = numarr
                    continue
                if part1+"i" == "ewldai":
                    self.ewldai[part2] = numarr
                    continue
                if part1+"i" == "wldrai":
                    self.wldrai[part2] = numarr
                    continue
                if part1+"i" == "wldpai":
                    self.wldpai[part2] = numarr
                    continue
                if part1+"m" == "wldsptam":
                    self.wldsptam[part2] = numarr
                    continue
                if part1+"_"+part2 == "wlpump_A":
                    self.wlpump_A = numarr
                    continue
                if part1+"_"+part2 == "wlabsrp_A":
                    self.wlabsrp_A = numarr
                    continue
            #(NLIM+NSTS,NMOL)    or  (NMOL, NLIM+NSTS)
            if ((self.nlim+self.nsts+1)*self.nmol) == int(size):
                if "wlabsrp" in var or "wlpump" in var:
                    # (nmol,nlim+nsts+1)
                    numarr = np.reshape(
                        numarr, (self.nmol, self.nlim+self.nsts+1), order='F')
                else:
                    numarr = np.reshape(
                        numarr, (self.nlim+self.nsts+1, self.nmol), order='F')
                if part1+"i" == "wldnmi":
                    self.wldnai[part2] = numarr
                    continue
                if part1+"i" == "ewldmi":
                    self.ewldai[part2] = numarr
                    continue
                if part1+"i" == "wldrmi":
                    self.wldrmi[part2] = numarr
                    continue
                if part1+"i" == "wldpmi":
                    self.wldpmi[part2] = numarr
                    continue
                if part1+"m" == "wldsptmm":
                    self.wldsptmm[part2] = numarr
                    continue
                if part1+"_"+part2 == "wlpump_M":
                    self.wlpump_M = numarr
                    continue
                if part1+"_"+part2 == "wlabsrp_M":
                    self.wlabsrp_M = numarr
                    continue
            if ((self.nlim+self.nsts+1)*self.nion) == int(size):
                if "wlabsrp" in var or "wlpump" in var:
                    # (nmol,nlim+nsts+1)
                    numarr = np.reshape(
                        numarr, (self.nmol, self.nlim+self.nsts+1), order='F')
                else:
                    numarr = np.reshape(
                        numarr, (self.nlim+self.nsts+1, self.nmol), order='F')
                if part1+"_"+part2 == "wlpump_I":
                    self.wlpump_I = numarr
                    continue
                if part1+"_"+part2 == "wlabsrp_I":
                    self.wlabsrp_I = numarr
                    continue
            if ((self.nlim+self.nsts+1)*self.npls) == int(size):
                if "wlabsrp" in var or "wlpump" in var:
                    # (nmol,nlim+nsts+1)
                    numarr = np.reshape(
                        numarr, (self.npls, self.nlim+self.nsts+1), order='F')
                else:
                    numarr = np.reshape(
                        numarr, (self.nlim+self.nsts+1, self.npls), order='F')
                if part1+"i" == "wldppi":
                    self.wldppi[part2] = numarr
                    continue
                if part1+"_"+part2 == "wlpump_P":
                    self.wlpump_P = numarr
                    continue
                if part1+"_"+part2 == "wlabsrp_P":
                    self.wlabsrp_P = numarr
                    continue
            if self.ncl*self.natm == int(size):
                numarr = np.reshape(numarr, (self.ncl,self.natm),order = "F")
                self.wlpump_resA = numarr
                continue
            if self.ncl*self.nmol == int(size):
                if part1+part2 == "wlpump_resM":
                    numarr = np.reshape(numarr, (self.ncl,self.nmol),order = "F")
                    self.wlpump_resM = numarr
                    continue
            if self.ncl*self.nion == int(size):
                if part1+part2 == "wlpump_resI":
                    numarr = np.reshape(numarr, (self.ncl,self.nion),order = "F")
                    self.wlpump_resI = numarr
                continue
            if self.ncl*self.npls == int(size):
                numarr = np.reshape(numarr, (self.ncl,self.npls),order = "F")
                self.wlpump_resP = numarr
                continue
        self.wldnek = self.wldneki[0]
        self.wldnep = self.wldnepi[0]
        self.wldna = self.wldnai[0]
        self.ewlda = self.ewldai[0]
        self.wldnm = self.wldnmi[0]
        self.ewldm = self.ewldmi[0]
        self.wldra = self.wldrai[0]
        self.wldrm = self.wldrmi[0]
        self.wldpp = self.wldppi[0]
        self.wldpa = self.wldpai[0]
        self.wldpm = self.wldpmi[0]
        self.wldpeb = self.wldpebi[0]
        self.wldspt = self.wldspti[0]
        self.wldspta = self.wldsptam[0]
        self.wldsptm = self.wldsptmm[0]

