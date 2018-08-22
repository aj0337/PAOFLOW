#
# PAOFLOW
#
# Utility to construct and operate on Hamiltonians from the Projections of DFT wfc on Atomic Orbital bases (PAO)
#
# Copyright (C) 2016-2018 ERMES group (http://ermes.unt.edu, mbn@unt.edu)
#
# Reference:
# M. Buongiorno Nardelli, F. T. Cerasoli, M. Costa, S Curtarolo,R. De Gennaro, M. Fornari, L. Liyanage, A. Supka and H. Wang,
# PAOFLOW: A utility to construct and operate on ab initio Hamiltonians from the Projections of electronic wavefunctions on
# Atomic Orbital bases, including characterization of topological materials, Comp. Mat. Sci. vol. 143, 462 (2018).
#
# This file is distributed under the terms of the
# GNU General Public License. See the file `License'
# in the root directory of the present distribution,
# or http://www.gnu.org/copyleft/gpl.txt .
#
import os
from write2bxsf import *
#from write3Ddatagrid import *

def do_fermisurf ( data_controller ):
#def do_fermisurf(fermi_dw,fermi_up,E_k,alat,b_vectors,nk1,nk2,nk3,nawf,ispin,npool,inputpath):
    import numpy as np
    from mpi4py import MPI
    from communication import gather_full

    rank = MPI.COMM_WORLD.Get_rank()

    arrays = data_controller.data_arrays
    attributes = data_controller.data_attributes

    #maximum number of bands crossing fermi surface

    E_k_full = gather_full(arrays['E_k'], attributes['npool'])

    if rank==0:
      nbndx_plot = 10
      nawf = attributes['nawf']
      nktot = attributes['nkpnts']
      nk1,nk2,nk3 = attributes['nk1'],attributes['nk2'],attributes['nk3']

      fermi_dw,fermi_up = attributes['fermi_dw'],attributes['fermi_up']

      E_k_rs = np.reshape(E_k_full, (nk1,nk2,nk3,nawf,attributes['nspin']))

      for ispin in range(attributes['nspin']):

        eigband = np.zeros((nk1,nk2,nk3,nbndx_plot), dtype=float)
        ind_plot = np.zeros(nbndx_plot)

        Efermi = 0.0

        #collect the interpolated eignvalues
        icount = 0
        for ib in range(nawf):
          if ((np.amin(E_k_full[:,ib,ispin]) < fermi_up and np.amax(E_k_full[:,ib,ispin]) > fermi_up) or \
          (np.amin(E_k_full[:,ib,ispin]) < fermi_dw and np.amax(E_k_full[:,ib,ispin]) > fermi_dw) or \
          (np.amin(E_k_full[:,ib,ispin]) > fermi_dw and np.amax(E_k_full[:,ib,ispin]) < fermi_up)):
            if ( icount > nbndx_plot ):
              print('Too many bands contributing')
              MPI.COMM_WORLD.Abort()
            eigband[:,:,:,icount] = E_k_rs[:,:,:,ib,ispin]
            ind_plot[icount] = ib
            icount +=1
        x0 = np.zeros(3, dtype=float)   

        write2bxsf(fermi_dw,fermi_up,eigband, nk1, nk2, nk3, icount, ind_plot, Efermi, attributes['alat'],x0, arrays['b_vectors'], 'FermiSurf_'+str(ispin)+'.bxsf',attributes['inputpath'])   

#  WHAT IS THIS WRITE?
# NO ISPIN IN FILENAME
#        for ib in range(icount):
#          np.savez(os.path.join(inputpath,'Fermi_surf_band_'+str(ib)), nameband = eigband[:,:,:,ib])

      E_k_rs = None

    E_k_full = None
