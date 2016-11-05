#
# AflowPI_TB.py
#
# Utility to construct and operate on TB Hamiltonians from the projections of DFT wfc on the pseudoatomic orbital basis (PAO)
#
# Copyright (C) 2015 Luis A. Agapito, 2016 Marco Buongiorno Nardelli
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).

# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).

# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).

from __future__ import print_function
from scipy import linalg as LA
from scipy import fftpack as FFT
from numpy import linalg as LAN
import numpy as np
import cmath
import sys
import re
sys.path.append('/home/marco/Programs/AflowPI_TB/')
import AflowPI_TB_lib as API
sys.path.append('./')
 
#units
Ry2eV      = 13.60569193

input_file = sys.argv[1]

read_S, shift_type, fpath, shift, pthr, do_comparison, double_grid,\
	print_kgrid, nk1, nk2, nk3 = API.read_input(input_file)

if (not read_S):
	U, my_eigsmat, alat, a_vectors, b_vectors, \
	nkpnts, nspin, kpnts, kpnts_wght, \
	nbnds, Efermi, nawf =  API.read_QE_output_xml(fpath,read_S)
   	Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
	sumk = np.sum(kpnts_wght)
	kpnts_wght /= sumk
	for ik in range(nkpnts):
        	Sks[:,:,ik]=np.identity(nawf)
	print('...using orthogonal algorithm')
else:
	U,Sks, my_eigsmat, alat, a_vectors, b_vectors, \
	nkpnts, nspin, kpnts, kpnts_wght, \
	nbnds, Efermi, nawf =  API.read_QE_output_xml(fpath,read_S)
	sumk = np.sum(kpnts_wght)
	kpnts_wght /= sumk
	print('...using non-orthogonal algorithm')

# Get grid of k-vectors in the fft order for the nscf calculation
if print_kgrid:
	API.get_K_grid_fft(nk1,nk2,nk3,b_vectors, print_kgrid)

# Building the Projectability
Pn = API.build_Pn(nawf,nbnds,nkpnts,nspin,U)

print('Projectability vector ',Pn)

# Check projectability and decide bnd

bnd = 0
for n in range(nbnds):
   if Pn[n] > pthr:
      bnd += 1
print('# of bands with good projectability (>',pthr,') = ',bnd)
 
# Building the TB Hamiltonian 
nbnds_norm = nawf
Hks = API.build_Hks(nawf,bnd,nbnds,nbnds_norm,nkpnts,nspin,shift,my_eigsmat,shift_type,U)

# Take care of non-orthogonality, if needed
# Hks from projwfc is orthogonal. If non-orthogonality is required, we have to apply a basis change to Hks as
# Hks -> Sks^(1/2)*Hks*Sks^(1/2)+

if read_S:
	S2k  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
	for ik in range(nkpnts):
		w, v = LAN.eigh(Sks[:,:,ik],UPLO='U')
		w = np.sqrt(w)
		S2k[:,:,ik] = v*w

	Hks_no = np.zeros((nawf,nawf,nkpnts,nspin),dtype=complex)
	for ispin in range(nspin):
		for ik in range(nkpnts):
			Hks_no[:,:,ik,ispin] = S2k[:,:,ik].dot(Hks[:,:,ik,ispin]).dot(np.conj(S2k[:,:,ik]).T)
	Hks = Hks_no

# Plot the TB and DFT eigevalues. Writes to comparison.pdf
if do_comparison:
	API.plot_compare_TB_DFT_eigs(Hks_no,Sks,my_eigsmat,read_S)
	quit()

# Define the Hamiltonian and overlap matrix in real space: HRs and SRs (noinv and nosym = True in pw.x)

# Define real space lattice vectors for Fourier transform of Hks
nr1=nk1
nr2=nk2
nr3=nk3
#R,R_wght,nrtot = API.get_R_grid_WanT(nr1,nr2,nr3,a_vectors)
#R,R_wght,nrtot = API.get_R_grid_WS(nr1,nr2,nr3,a_vectors)
#R,R_wght,nrtot,idx = API.get_R_grid_q(nr1,nr2,nr3,a_vectors)
R,R_wght,nrtot,idx = API.get_R_grid_fft(nr1,nr2,nr3,a_vectors)
if abs(np.sum(R_wght)-float(nr1*nr2*nr3)) > 1.0e-8:
	print(np.sum(R_wght), float(nr1*nr2*nr3))
	sys.exit('wrong sum rule on R weights')
#print('Number of R vectors for Fourier interpolation ',nrtot)
#for n in range(nrtot):
#	print("%.5f" % R[n,0],"%.5f" % R[n,1],"%.5f" % R[n,2])

# Original k grid to R grid
Hkaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
Skaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)
for i in range(nk1):
	for j in range(nk2):
		for k in range(nk3):
			Hkaux[:,:,i,j,k,:] = Hks[:,:,idx[i,j,k],:]	
			Skaux[:,:,i,j,k] = Sks[:,:,idx[i,j,k]]	

if double_grid:
	# Fourier interpolation on double grid
	nk1p = 2*nk1
	nk2p = 2*nk2
	nk3p = 2*nk3
	nrtotp= nk1p*nk2p*nk3p
	print('Number of R vectors for extended Fourier interpolation ',nrtotp)

	# Extended k to R (zero padding)
	Hkauxp  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p,nspin),dtype=complex)
	Skauxp  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p),dtype=complex)
	HRaux  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p,nspin),dtype=complex)
	SRaux  = np.zeros((nawf,nawf,nk1p,nk2p,nk3p),dtype=complex)
	for ispin in range(nspin):
        	for i in range(nawf):
                	for j in range(nawf):
				for k1 in range(nk1):
					for k2 in range(nk2):
						for k3 in range(nk3):
							Hkauxp[i,j,2*k1,2*k2,2*k3,ispin] = Hkaux[i,j,k1,k2,k3,ispin]
                        				if read_S and ispin == 0:
                                				SRauxp[i,j,2*k1,2*k2,2*k3] = Skaux[i,j,2*k1,2*k2,2*k3]
			
	for ispin in range(nspin):
        	for i in range(nawf):
                	for j in range(nawf):
                        	HRaux[i,j,:,:,:,ispin] = FFT.ifftn(Hkauxp[i,j,:,:,:,ispin])
                        	if read_S and ispin == 0:
                                	SRaux[i,j,:,:,:] = FFT.ifftn(Skauxp[i,j,:,:,:])
	nk1 = nk1p
	nk2 = nk2p
	nk3 = nk3p
	nrtot = nrtotp

else:
	HRaux  = np.zeros((nawf,nawf,nk1,nk2,nk3,nspin),dtype=complex)
	SRaux  = np.zeros((nawf,nawf,nk1,nk2,nk3),dtype=complex)
	for ispin in range(nspin):
		for i in range(nawf):
			for j in range(nawf):
				HRaux[i,j,:,:,:,ispin] = FFT.ifftn(Hkaux[i,j,:,:,:,ispin])
				if read_S and ispin == 0:
					SRaux[i,j,:,:,:] = FFT.ifftn(Skaux[i,j,:,:,:])

HRs  = np.zeros((nawf,nawf,nrtot,nspin),dtype=complex)
SRs  = np.zeros((nawf,nawf,nrtot),dtype=complex)
R = np.zeros((nrtot,3),dtype=float)
R_wght = np.ones((nrtot),dtype=float)

for i in range(nk1):
        for j in range(nk2):
                for k in range(nk3):
                        n = k + j*nk3 + i*nk2*nk3
                        Rx = float(i)
                        Ry = float(j)
                        Rz = float(k)
                        if Rx >= nk1/2: Rx=Rx-nk1
                        if Ry >= nk2/2: Ry=Ry-nk2
                        if Rz >= nk3/2: Rz=Rz-nk3
                        R[n,:] = Rx*a_vectors[0,:]+Ry*a_vectors[1,:]+Rz*a_vectors[2,:]
                        HRs[:,:,n,:] = HRaux[:,:,i,j,k,:]
                        if read_S and ispin == 0:
                        	SRs[:,:,n] = SRaux[:,:,i,j,k]

# Define k-point mesh for bands interpolation
kq,kmod,nkpi = API.kpnts_interpolation_mesh()

# Define the Hamiltonian and overlap matrix in reciprocal space on the interpolated k-point mesh
Hks_int  = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex)
Sks_int  = np.zeros((nawf,nawf,nkpi),dtype=complex)
for ispin in range(nspin):
        for ik in range(nkpi):
                for nr in range(nrtot):
                   	phase=R_wght[nr]*cmath.exp(2.0*np.pi*kq[:,ik].dot(R[nr,:])*1j)
                       	Hks_int[:,:,ik,ispin] += HRs[:,:,nr,ispin]*phase
			if read_S and ispin == 0:
                       		Sks_int[:,:,ik] += SRs[:,:,nr]*phase

# Take care of non-orthogonality, if needed
# Hks from projwfc is orthogonal. If non-orthogonality is required, we have to apply a basis change to Hks as
# Hks -> Sks^(1/2)*Hks*Sks^(1/2)+
if read_S:
	S2k_int  = np.zeros((nawf,nawf,nkpi),dtype=complex)
	for ik in range(nkpi):
		w, v = LAN.eigh(Sks_int[:,:,ik],UPLO='U')
		w = np.sqrt(w)
		S2k_int[:,:,ik] = v*w

	Hks_no_int = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex)
	for ispin in range(nspin):
		for ik in range(nkpi):
			Hks_no_int[:,:,ik,ispin] = S2k_int[:,:,ik].dot(Hks_int[:,:,ik,ispin]).dot(np.conj(S2k_int[:,:,ik]).T)
	Hks_int = Hks_no_int

# Plot TB eigenvalues on interpolated mesh
API.plot_TB_eigs(Hks_int,Sks_int,kmod,read_S)

