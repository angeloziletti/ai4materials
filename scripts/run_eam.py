import numpy as np

from ase.calculators.eam import EAM
from ase.build import bulk

# test to generate an EAM potential file using a simplified
# approximation to the Mishin potential Al99.eam.alloy data

from scipy.interpolate import InterpolatedUnivariateSpline as spline

cutoff = 6.28721

n = 21
rs = np.arange(0, n) * (cutoff / n)
rhos = np.arange(0, 2, 2. / n)

# potentials from https://github.com/EACcodes/EAMpotentials/tree/master/solid/Fe%2BH
# https://carter.princeton.edu/research/eam-potentials/

mishin = EAM(potential='/home/ziletti/Documents/calc_nomadml/rot_inv_3d/md/bulkB.alloy')
m_density = mishin.electron_density[0](rs)
m_embedded = mishin.embedded_energy[0](rhos)
m_phi = mishin.phi[0, 0](rs)

m_densityf = spline(rs, m_density)
m_embeddedf = spline(rhos, m_embedded)
m_phif = spline(rs, m_phi)

a = 2.856  # Angstrom lattice spacing
fe = bulk('Fe', 'bcc', a=a)


mishin_approx = EAM(elements=['Fe'], embedded_energy=np.array([m_embeddedf]),
                    electron_density=np.array([m_densityf]),
                    phi=np.array([[m_phif]]), cutoff=cutoff)

fe.set_calculator(mishin_approx)


