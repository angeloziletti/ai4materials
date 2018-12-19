from __future__ import print_function

from ase.lattice.cubic import FaceCenteredCubic
from ase.md.langevin import Langevin
from ase.md.nvtberendsen import NVTBerendsen
from ase.io.trajectory import Trajectory
from ase import units

from asap3 import EMT  # Way too slow with ase.EMT !
size = 3

T = 5  # Kelvin

# Set up a crystal
# atoms = FaceCenteredCubic(
#     # directions=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
#                           symbol="Cu",
#                           size=(size, size, size),
#                           pbc=True)

from ase.spacegroup import crystal

a = 2.87
atoms = crystal('Fe', [(0, 0, 0)], spacegroup=229, cellpar=[a, a, a, 90, 90, 90])

# Describe the interatomic interactions with the Effective Medium Theory
atoms.set_calculator(EMT())
# We want to run MD with constant energy using the Langevin algorithm
# with a time step of 5 fs, the temperature T and the friction
# coefficient to 0.02 atomic units.
dyn = Langevin(atoms, 1 * units.fs, T * units.kB, 0.002)


def printenergy(a=atoms):  # store a reference to atoms in the definition.
    """Function to print the potential, kinetic and total energy."""
    epot = a.get_potential_energy() / len(a)
    ekin = a.get_kinetic_energy() / len(a)
    temp = int(round(ekin / (1.5 * units.kB)))
    if temp == T:
        print('Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  '
              'Etot = %.3feV' % (epot, ekin, temp, epot + ekin))


dyn.attach(printenergy, interval=100)

# We also want to save the positions of all atoms after every 100th time step.
traj = Trajectory('moldyn3_langevin.traj', 'w', atoms)
dyn.attach(traj.write, interval=100)

# Now run the dynamics
printenergy()
dyn.run(50000)

import numpy as np
# http://pymatgen.org/pymatgen.analysis.diffusion_analyzer.html?highlight=mean%20square%20displacement
#  get_msd_plot(plt=None, mode=u'specie')[source]
# Get the plot of the smoothed msd vs time graph. Useful for checking convergence. This can be written to an image file.
#

# https://stackoverflow.com/questions/31264591/mean-square-displacement-python
# r = np.sqrt(xdata ** 2 + ydata ** 2)
# diff = np.diff(r)  # this calculates r(t + dt) - r(t)
# diff_sq = diff ** 2
# MSD = np.mean(diff_sq)
