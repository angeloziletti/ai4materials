from __future__ import print_function

from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin
from ase.io.trajectory import Trajectory
from ase import units
import os.path
from asap3 import EMT  # Way too slow with ase.EMT !
import numpy as np

output_folder = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/structures_for_paper/melting_copper/'

output_traj_file = os.path.abspath(os.path.normpath(os.path.join(output_folder, 'moldyn3_langevin_2.traj')))

supercell_size = 3

# target_temps = np.linspace(0., 200, 6)  # Kelviv
n_samples = 5

# set up a crystal
# atoms = FaceCenteredCubic(symbol="Cu", size=(supercell_size, supercell_size, supercell_size), pbc=True)


def save_temp(a):  # store a reference to atoms in the definition.
    """Function to save the actual temperature in the atoms structure"""
    ekin = a.get_kinetic_energy() / len(a)
    temp = ekin / (1.5 * units.kB)
    # a.info['temp'] = int(round(temp))
    a.info['temp'] = temp
    print(temp)
    return a


ase_atoms_list = []

target_temps = [10, 20, 30, 40]

for target_temp in target_temps:

    atoms = FaceCenteredCubic(symbol="Cu", size=(supercell_size, supercell_size, supercell_size), pbc=True)

    # Describe the interatomic interactions with the Effective Medium Theory
    # see here for supported chemical elements (fcc only)
    # https://wiki.fysik.dtu.dk/asap/EMT
    # set up a crystal

    atoms.set_calculator(EMT())
    # We want to run MD with constant energy using the Langevin algorithm
    # with a time step of 5 fs, the temperature T and the friction
    # coefficient to 0.02 atomic units.
    dyn = Langevin(atoms, 1 * units.fs, target_temp * units.kB, 0.002)
    # dyn.attach(save_temp, interval=100)

    # We also want to save the positions of all atoms after every 100th time step.
    traj = Trajectory(output_traj_file, 'w', atoms)
    dyn.attach(traj.write, interval=100)

    print("Running dynamics for temp {}".format(target_temp))
    # # now run the dynamics
    # for i in range(n_samples):
    dyn.run(10000)
    atoms = save_temp(atoms)
    ase_atoms_list.append(atoms)
    print("Done.")

    # read trajectory file
    # traj = Trajectory(output_traj_file)

    idx_sample = 0
    for idx, atoms in enumerate(ase_atoms_list):
        # print("Configuration with target temp {}".format(atoms.info['temp']))

        if atoms.info['temp'] == target_temp:
            print("Adding configuration with target temp {}".format(target_temp))
            ase_atoms_list.append(atoms)
            idx_sample += 1

            if idx_sample >= n_samples:
                print("Reached target n_samples for temp {}: {}".format(target_temp, n_samples))
                break

    del dyn
    del atoms

print(len(ase_atoms_list))