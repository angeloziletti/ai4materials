======
Utils
======

This package contains utility functions for data analytics applied to materials science data.
Specifically,

* :py:mod:`ai4materials.utils.utils_config`: functions to set up useful parameters for calculations.
* :py:mod:`ai4materials.utils.utils_crystals`: functions related to crystal structures
* :py:mod:`ai4materials.utils.utils_data_retrieval`: functions to retrieve data.
* :py:mod:`ai4materials.utils.utils_mp`: functions for parallel execution.
* :py:mod:`ai4materials.utils.utils_plotting`: functions for plotting results of modelling.
* :py:mod:`ai4materials.utils.utils_vol_data`: functions to deal with three-dimensional volumetric data.


Utils crystals
===============
.. module:: ai4materials.utils.utils_crystals
   :synopsis: Utils crystals

This package contains functions to build pristine and defective supercells,
starting from ASE (Atomistic Simulation Environment) Atoms object
`[link] <https://wiki.fysik.dtu.dk/ase/ase/atoms.html?highlight=atoms%20object>`_.
It also allows to obtain the spacegroup of a given structure,
or to get the standard conventional cell (using Pymatgen).


Pristine and defective supercell generation
-------------------------------------------

The main functions available to modify crystal structures are:

* :py:mod:`ai4materials.utils.utils_crystals.create_supercell` creates a pristine supercell starting from a given atom structure.
* :py:mod:`ai4materials.utils.utils_crystals.random_displace_atoms` creates a supercell with randomly displace atoms.
* :py:mod:`ai4materials.utils.utils_crystals.create_vacancies` creates a supercell with vacancies.
* :py:mod:`ai4materials.utils.utils_crystals.substitute_atoms` creates a supercell with randomly substitute atoms.
For additional details on each function, see their respective descriptions below.


Example: pristine supercell creation
--------------------------------------
.. module:: ai4materials.utils.utils_crystals.create_supercell
   :synopsis: Pristine supercell

Starting from a given ASE structure, the script below uses :py:mod:`ai4materials.utils.utils_crystals.create_supercell`
to generate a supercell of (approximately) 128 atoms:

.. testcode::

    from ase.io import write
    from ase.build import bulk
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms
    from ai4materials.utils.utils_crystals import create_supercell
    cu_fcc = bulk('Cu', 'fcc', a=3.6, orthorhombic=True)
    supercell_cu_fcc =  create_supercell(cu_fcc, create_replicas_by='nb_atoms', target_nb_atoms=128)
    write('cu_fcc.png', cu_fcc)
    write('cu_fcc_supercell.png', supercell_cu_fcc)

This is the original structure:

.. image:: cu_fcc.png

and this is the supercell obtained replicating the unit cells up to a target number of atoms (``target_nb_atoms``)

.. image:: cu_fcc_supercell.png


Example: defective supercell creation
--------------------------------------
.. module:: ai4materials.utils.utils_crystals.create_vacancies
   :synopsis: Supercell with vacancies


Starting from a given ASE structure, the script below uses :py:mod:`ai4materials.utils.utils_crystals.create_vacancies`
to generate a defective supercell of (approximately) 128 atoms with 25% vacancies:

.. testcode::

    from ase.io import write
    from ase.build import bulk
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms
    from ai4materials.utils.utils_crystals import create_vacancies
    cu_fcc = bulk('Cu', 'fcc', a=3.6, orthorhombic=True)
    supercell_vac25_cu_fcc =  create_vacancies(cu_fcc, target_vacancy_ratio=0.25, create_replicas_by='nb_atoms', target_nb_atoms=128)
    write('cu_fcc.png', cu_fcc)
    write('cu_fcc_supercell_vac25.png', supercell_vac25_cu_fcc)

.. image:: cu_fcc_supercell_vac25.png

Similarly, it is possible to generate a supercell with randomly displaced atoms with
:py:mod:`ai4materials.utils.utils_crystals.random_displace_atoms`.
In the script below,
we generate a defective supercell of (approximately) 200 atoms with displacements sampled from a Gaussian
distribution with standard deviation of 0.5 Angstrom:

.. testcode::

    from ase.io import write
    from ase.build import bulk
    import matplotlib.pyplot as plt
    from ase.visualize.plot import plot_atoms
    from ai4materials.utils.utils_crystals import random_displace_atoms
    cu_fcc = bulk('Cu', 'fcc', a=3.6, orthorhombic=True)
    supercell_rand_disp_cu_fcc =  random_displace_atoms(cu_fcc, displacement=0.5, create_replicas_by='nb_atoms', noise_distribution='gaussian', target_nb_atoms=256)
    write('cu_fcc.png', cu_fcc)
    write('supercell_rand_disp_cu_fcc_05A.png', supercell_rand_disp_cu_fcc)

.. image:: supercell_rand_disp_cu_fcc_05A.png

.. sectionauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>


Submodules
----------

.. toctree::

   ai4materials.utils.unit_conversion
   ai4materials.utils.utils_binaries
   ai4materials.utils.utils_config
   ai4materials.utils.utils_crystals
   ai4materials.utils.utils_data_retrieval
   ai4materials.utils.utils_mp
   ai4materials.utils.utils_parsing
   ai4materials.utils.utils_plotting
   ai4materials.utils.utils_vol_data

Module contents
---------------

.. automodule:: ai4materials.utils
    :members:
    :undoc-members:
    :show-inheritance:
