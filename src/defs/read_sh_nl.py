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

def read_sh_nl (data_controller):
    # reads in shelks from pseudo files
    from os.path import join,exists
    import numpy as np


#    arry,attr = data_controller.data_dicts()
    arry = data_controller.data_arrays
    attr = data_controller.data_attributes
 
    # Get Shells for each species
    sdict = {}
    for s in arry['species']:
      sdict[s[0]] = read_pseudopotential(join(attr['workpath'],attr['savedir'],s[1]))

    for s,p in sdict.items():
        tmp_list=[]
        for o in p:
            tmp_list.append(o)
            # if l=0 include it twice
            if o==0:
                tmp_list.append(o)

        sdict[s] = np.array(tmp_list)

    # value of l
    shell   = np.hstack([sdict[a] for a in arry['atoms']])

    return shell[::2],np.ones(shell.shape[0],dtype=int)

############################################################################################
############################################################################################
############################################################################################

def read_pseudopotential ( fpp ):
  '''
  Reads a psuedopotential file to determine the included shells and occupations.

  Arguments:
      fnscf (string) - Filename of the pseudopotential, copied to the .save directory

  Returns:
      sh, nl (lists) - sh is a list of orbitals (s-0, p-1, d-2, etc)
                       nl is a list of occupations at each site
      sh and nl are representative of one atom only
  '''

  import numpy as np
  import xml.etree.cElementTree as ET
  import re
  from tempfile import NamedTemporaryFile

  sh = []
 
  # clean xnl before reading
  with open(fpp) as ifo:
      temp_str=ifo.read()

  temp_str = re.sub('&',' ',temp_str)
  f = NamedTemporaryFile(mode='w',delete=True)
  f.write(temp_str)

  try:
      iterator_obj = ET.iterparse(f.name,events=('start','end'))
      iterator     = iter(iterator_obj)
      event,root   = next(iterator)

      for event,elem in iterator:        
          try:
              for i in elem.findall("PP_PSWFC/"):
                  sh.append(int(i.attrib['l']))
          except Exception as e:
              pass

      sh   = np.array(sh)

  except Exception as e:
      with open(fpp) as ifo:
          ifs=ifo.read()
      res=re.findall("(.*)\s*Wavefunction",ifs)[1:]      
      sh=np.array(list(map(int,list([x.split()[1] for x in res]))))

  return sh

