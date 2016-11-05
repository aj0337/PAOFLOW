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

read_S, shift_type, fpath, shift, pthr, do_comparison = API.read_input(input_file)

if (not read_S):
	U, my_eigsmat, alat, a_vectors, \
	nkpnts, nspin, kpnts, kpnts_wght, \
	nbnds, Efermi, nawf, nk1, nk2, nk3 =  API.read_QE_output_xml(fpath,read_S)
   	Sks  = np.zeros((nawf,nawf,nkpnts),dtype=complex)
	for ik in range(nkpnts):
        	Sks[:,:,ik]=np.identity(nawf)
	print('...using orthogonal algorithm')
else:
	U,Sks, my_eigsmat, alat, a_vectors, \
	nkpnts, nspin, kpnts, kpnts_wght, \
	nbnds, Efermi, nawf, nk1,nk2,nk3 =  API.read_QE_output_xml(fpath,read_S)
	print('...using non-orthogonal algorithm')

# Define real space lattice vectors for Fourier transform of Hks
nr1=nk1=4
nr2=nk2=4
nr3=nk3=4
#R,R_wght,nrtot = API.get_R_grid(nr1,nr2,nr3,a_vectors)
R,R_wght,nrtot = API.get_R_grid_WS(nr1,nr2,nr3,a_vectors)
if abs(np.sum(R_wght)-float(nr1*nr2*nr3)) > 1.0e-8:
	sys.exit('wrong sum rule on R weights')
print('Number of R vectors for Fourier interpolation ',nrtot)

# Define k-point mesh for interpolation
k,kmod,nkpi = API.kpnts_interpolation_mesh()

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
else:
	Hks_no = Hks

# Plot the TB and DFT eigevalues. Writes to comparison.pdf
if do_comparison:
	API.plot_compare_TB_DFT_eigs(Hks_no,Sks,my_eigsmat,read_S)
	quit()

# Define the Hamiltonian and overlap matrix in real space: HRs and SRs (noinv and nosym = True in pw.x)
sumk = np.sum(kpnts_wght)
kpnts_wght /= sumk
HRs  = np.zeros((nawf,nawf,nrtot,nspin),dtype=complex)
SRs  = np.zeros((nawf,nawf,nrtot),dtype=complex)
for ispin in range(nspin):
	for nr in range(nrtot):
		for ik in range(nkpnts):
			phase=kpnts_wght[ik]*cmath.exp(2.0*np.pi*kpnts[ik,:].dot(R[nr,:])*(-1j))
			HRs[:,:,nr,ispin] += Hks[:,:,ik,ispin]*phase
			if read_S and ispin == 0:
				SRs[:,:,nr] += Sks[:,:,ik]*phase

# Define the Hamiltonian and overlap matrix in reciprocal space on the interpolated k-point mesh
Hks_int  = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex)
Sks_int  = np.zeros((nawf,nawf,nkpi),dtype=complex)
for ispin in range(nspin):
        for ik in range(nkpi):
                for nr in range(nrtot):
                        phase=R_wght[nr]*cmath.exp(2.0*np.pi*k[:,ik].dot(R[nr,:])*1j)
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
else:
	Hks_no_int = Hks_int

# Plot TB eigenvalues on interpolated mesh
API.plot_TB_eigs(Hks_no_int,Sks_int,kmod,read_S)

