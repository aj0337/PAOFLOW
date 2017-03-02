#
# PAOpy
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016 ERMES group (http://ermes.unt.edu)
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
#
# References:
# Luis A. Agapito, Andrea Ferretti, Arrigo Calzolari, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Effective and accurate representation of extended Bloch states on finite Hilbert spaces, Phys. Rev. B 88, 165127 (2013).
#
# Luis A. Agapito, Sohrab Ismail-Beigi, Stefano Curtarolo, Marco Fornari and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonian Matrices from Ab-Initio Calculations: Minimal Basis Sets, Phys. Rev. B 93, 035104 (2016).
#
# Luis A. Agapito, Marco Fornari, Davide Ceresoli, Andrea Ferretti, Stefano Curtarolo and Marco Buongiorno Nardelli,
# Accurate Tight-Binding Hamiltonians for 2D and Layered Materials, Phys. Rev. B 93, 125137 (2016).
#
from scipy import fftpack as FFT
import numpy as np
import cmath
import sys

from mpi4py import MPI
from mpi4py.MPI import ANY_SOURCE

#import matplotlib.pyplot as plt

from write_TB_eigs import *
from kpnts_interpolation_mesh import *
#from new_kpoint_interpolation import *
from do_non_ortho import *
from load_balancing import *

# initialize parallel execution
comm=MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def do_eigh_calc(HRaux,SRaux,kq,R_wght,R,idx,read_S):
    # Compute bands on a selected mesh in the BZ

    nkpi=kq.shape[0]

    nawf,nawf,nk1,nk2,nk3,nspin = HRaux.shape
    Hks_int  = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex) # final data arrays

    Hks_int[:,:,:,:] = band_loop_H(nspin,nk1,nk2,nk3,nawf,nkpi,HRaux,R_wght,kq,R,idx)

    Sks_int  = np.zeros((nawf,nawf,nkpi),dtype=complex)
    if read_S:
        Sks_int  = np.zeros((nawf,nawf,nkpi),dtype=complex)
        Sks_int[:,:,:] = band_loop_S(nspin,nk1,nk2,nk3,nawf,nkpi,SRaux,R_wght,kq,R,idx)

    E_kp = np.zeros((nkpi,nawf,nspin),dtype=float)
    v_kp = np.zeros((nkpi,nawf,nawf,nspin),dtype=complex)

    for ispin in xrange(nspin):
        for ik in range(nkpi):
            if read_S:
                E_kp[ik,:,ispin],v_kp[ik,:,:,ispin] = LA.eigh(Hks_int[:,:,ik,ispin],Sks_int[:,:,ik])
            else:
                E_kp[ik,:,ispin],v_kp[ik,:,:,ispin] = LAN.eigh(Hks_int[:,:,ik,ispin],UPLO='U')


#    if rank == 0:
#        plt.matshow(abs(Hks_int[:,:,1445,0]))
#        plt.colorbar()
#        plt.show()
#
#        np.save('Hks_noSO0',Hks_int[:,:,0,0])

    return(E_kp,v_kp)

def band_loop_H(nspin,nk1,nk2,nk3,nawf,nkpi,HRaux,R_wght,kq,R,idx):

    auxh = np.zeros((nawf,nawf,nkpi,nspin),dtype=complex)
    HRaux = np.reshape(HRaux,(nawf,nawf,nk1*nk2*nk3,nspin),order='C')

    for ik in xrange(nkpi):
        for ispin in xrange(nspin):
             auxh[:,:,ik,ispin] = np.sum(HRaux[:,:,:,ispin]*np.exp(2.0*np.pi*kq[ik,:].dot(R[:,:].T)*1j),axis=2)

    return(auxh)

def band_loop_S(nspin,nk1,nk2,nk3,nawf,nkpi,SRaux,R_wght,kq,R,idx):

    auxs = np.zeros((nawf,nawf,nkpi),dtype=complex)

    for ik in xrange(nkpi):
        for i in xrange(nk1):
            for j in xrange(nk2):
                for k in xrange(nk3):
                    phase=R_wght[idx[i,j,k]]*cmath.exp(2.0*np.pi*kq[ik,:].dot(R[idx[i,j,k],:])*1j)
                    auxs[:,:,ik] += SRaux[:,:,i,j,k]*phase

    return(auxs)