#
# PAOFLOW
#
# Copyright 2016-2024 - Marco BUONGIORNO NARDELLI (mbn@unt.edu)
#
# Reference:
#
#F.T. Cerasoli, A.R. Supka, A. Jayaraj, I. Siloi, M. Costa, J. Slawinska, S. Curtarolo, M. Fornari, D. Ceresoli, and M. Buongiorno Nardelli, Advanced modeling of materials with PAOFLOW 2.0: New features and software design, Comp. Mat. Sci. 200, 110828 (2021).
#
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang, 
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on 
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .

import numpy as np

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def do_Sks_to_SRs ( data_controller ):
  from scipy import fftpack as FFT

  arry,attr = data_controller.data_dicts()

  #----------------------------------------------------------
  # Define the Hamiltonian and overlap matrix in real space:
  #   HRs and SRs (noinv and nosym = True in pw.x)
  #----------------------------------------------------------
  if rank == 0:
    # Original k grid to R grid
    arry['SRs'] = np.zeros_like(arry['Sks'])
    arry['SRs'] = FFT.ifftn(arry['Sks'], axes=[2])