from __future__ import print_function

from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin
from ai4materials.utils.utils_data_retrieval import write_ase_db
from asap3 import EMT  # Way too slow with ase.EMT !
import ase.calculators.emt
import copy
import numpy as np
from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_config import setup_logger




# write_ase_db(ase_atoms_list, output_folder, db_name='melting_copper', db_type='db', overwrite=True,
#              folder_name='db_ase')  # for item in ase_atoms_list:

