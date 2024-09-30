from src.PAOFLOW import PAOFLOW
import numpy as np

paoflow = PAOFLOW(savedir='./nscf_nspin1',  
                  outputdir='./output_nspin1', 
                  verbose=True,
                  dft="VASP")


basis_path = '../../../BASIS/'
basis_config = {'Si':['3S','3P','3D','4S','4P','4F']}

paoflow.projections(basispath=basis_path, configuration=basis_config)
paoflow.projectability()
paoflow.pao_hamiltonian()
paoflow.bands(ibrav=2, nk=500)


import matplotlib.pyplot as plt

data_controller = paoflow.data_controller
arry,attr = paoflow.data_controller.data_dicts()
eband = arry['E_k']

fig = plt.figure(figsize=(6,4))
plt.plot(eband[:,0], color='black')
for ib in range(1, eband.shape[1]):
    plt.plot(eband[:,ib], color='black')
plt.xlim([0,eband.shape[0]-1])
plt.ylabel("E (eV)")
plt.show()
