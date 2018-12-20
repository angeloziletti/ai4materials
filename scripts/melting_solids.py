from __future__ import print_function

from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin
from ai4materials.utils.utils_data_retrieval import write_ase_db
from ase import units
import os.path
# from asap3 import EMT  # Way too slow with ase.EMT !
import ase.calculators.emt
import numpy as np
from ai4materials.utils.utils_config import set_configs
from ai4materials.utils.utils_config import setup_logger


output_folder = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/structures_for_paper/melting_copper/'

configs = set_configs(main_folder=output_folder)
logger = setup_logger(configs, level='INFO', display_configs=False)

# output_file = os.path.abspath(os.path.normpath(os.path.join(output_folder, '0_200k_structures.pkl')))

supercell_size = 3

target_temps = np.linspace(0., 200, 10)  # Kelviv
n_samples = 5
max_nb_trials = 1000


def save_temp(a):  # store a reference to atoms in the definition.
    """Function to save the actual temperature in the atoms structure"""
    ekin = a.get_kinetic_energy() / len(a)
    temp = ekin / (1.5 * units.kB)
    # a.info['temp'] = int(round(temp))
    a.info['temp'] = temp
    return a


ase_atoms_list = []

for target_temp in target_temps:

    atoms = FaceCenteredCubic(symbol="Cu", size=(supercell_size, supercell_size, supercell_size), pbc=True)
    # Describe the interatomic interactions with the Effective Medium Theory
    # see here for supported chemical elements (fcc only)
    # https://wiki.fysik.dtu.dk/asap/EMT
    # set up a crystal
    # ASAP3 calculator
    # atoms.set_calculator(EMT())

    # we use  the much slower ASE implementation because it is not possible to save an ASE db to file
    # if we use the ASAP calculator
    atoms.set_calculator(ase.calculators.emt.EMT())

    # We want to run MD with constant energy using the Langevin algorithm
    # with a time step of 5 fs, the temperature T and the friction
    # coefficient to 0.02 atomic units.
    dyn = Langevin(atoms, 1 * units.fs, target_temp * units.kB, 0.02)
    # dyn.attach(save_temp, interval=100)

    # We also want to save the positions of all atoms after every 100th time step.
    # traj = Trajectory(output_traj_file, 'w', atoms)
    # dyn.attach(traj.write, interval=100)

    print("Running dynamics for temp {}".format(target_temp))
    # # now run the dynamics
    # for i in range(n_samples):
    dyn.run(1000)
    atoms = save_temp(atoms)

    # read trajectory file

    import copy
    interval = 100
    idx_sample = 0

    idx_trial = 0
    while idx_sample <= n_samples:
        print("Trial number: {}".format(idx_trial))
        dyn.run(interval)
        atoms = save_temp(atoms)
        # print(atoms.info['temp'])
        if int(round(atoms.info['temp'])) == target_temp:
            print("Adding configuration with target temp {}".format(atoms.info['temp']))
            # you need to deepcopy the object
            atoms_out = copy.deepcopy(atoms)
            ase_atoms_list.append(atoms_out)
            idx_sample += 1

            if idx_sample >= n_samples:
                print("Reached target n_samples for temp {}: {}".format(target_temp, n_samples))
                break

        idx_trial += 1

        if idx_trial >= max_nb_trials:
            raise Exception("Maximum number of trials ({}) exceeded.".format(max_nb_trials))
        # del dyn
    del atoms

write_ase_db(ase_atoms_list, output_folder, db_name='melting_copper', db_type='db', overwrite=True,
             folder_name='db_ase')  # for item in ase_atoms_list:

