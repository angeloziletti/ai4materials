# coding=utf-8
# Copyright 2016-2018 Angelo Ziletti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2018, Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "23/09/18"

from ase.neighborlist import NeighborList
from ase.build import find_optimal_cell_shape_pure_python
from ase.build import get_deviation_from_optimal_cell_shape
from ase.build import make_supercell
from ase.spacegroup import get_spacegroup as ase_get_spacegroup
import ase
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
from math import cos
from math import sin
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
import math
from scipy import constants
import scipy
import pandas as pd
from itertools import izip
from itertools import permutations
from decimal import Decimal
import copy
from ai4materials.utils.utils_binaries import get_chemical_formula_binaries
import ai4materials.utils.unit_conversion as uc
from ai4materials.utils.utils_config import get_data_filename
import scipy.misc
import logging
import spglib
import random
from pint import UnitRegistry

logger = logging.getLogger('ai4materials')


def get_spacegroup_analyzer(atoms, symprec=None, angle_tolerance=-1.):
    """Given an ASE structure and a symprec, return the SpacegroupAnalyzer.\n

    The SpacegroupAnalyzer is the object used in pymatgen to detect crystal symmetries: \n
    http://pymatgen.org/_modules/pymatgen/symmetry/analyzer.html

    Parameters:

    atoms: `ase.Atoms` object
        Atomic structure.

    symprec: list of floats
        Tolerance for symmetry finding. According to the pymatgen documentation: \n
        A value of 1e-3 is fairly strict and works well for properly refined
        structures with atoms in the proper symmetry coordinates. For
        structures with slight deviations from their proper atomic
        positions (e.g., structures relaxed with electronic structure
        codes), a looser tolerance of 0.1 (the value used in Materials
        Project) is often needed.

    angle_tolerance: float
        Tolerance of angle between lattice vectors in degrees to be tolerated in the symmetry finding.
        It is used by Spglib. By specifying a negative value, the behavior becomes the same as usual functions.
        For more info: https://atztogo.github.io/spglib/variable.html#variables-angle-tolerance

    Returns:

    `pymatgen.SpacegroupAnalyzer` object
        Return the SpacegroupAnalyzer of pymatgen.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    if symprec is None:
        symprec = [1e-03, 1e-06]
    if not isinstance(symprec, list):
        symprec = [symprec]

    spacegroup_analyzer = {}
    for symprec_ in symprec:
        spacegroup_analyzer[symprec_] = SpacegroupAnalyzer(AseAtomsAdaptor.get_structure(atoms), symprec=symprec_,
                                                           angle_tolerance=angle_tolerance)

    return spacegroup_analyzer


def get_spacegroup(atoms, symprec=None, angle_tolerance=-1.0):
    """Given an ASE structure and a (list of) symmetry threshold, return the spacegroup number.\n

    Internally, it uses the SpacegroupAnalyzer is the object used in pymatgen to detect crystal symmetries: \n
    http://pymatgen.org/_modules/pymatgen/symmetry/analyzer.html

    Parameters:

    atoms: ``ase.Atoms`` object
        Atomic structure.

    symprec: list of floats
        Tolerance for symmetry finding. According to the pymatgen documentation: \n
        A value of 1e-3 is fairly strict and works well for properly refined
        structures with atoms in the proper symmetry coordinates. For
        structures with slight deviations from their proper atomic
        positions (e.g., structures relaxed with electronic structure
        codes), a looser tolerance of 0.1 (the value used in Materials
        Project) is often needed.

    Returns:

    list of floats
        Spacegroup numbers obtained for the given symprec thresholds.

    .. seealso:: Internally, it uses the SpacegroupAnalyzer object of pymatgen; see
        :py:mod:`ai4materials.utils.utils_crystals.get_spacegroup_analyzer`

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    space_group_analyzer = get_spacegroup_analyzer(atoms, symprec=symprec, angle_tolerance=angle_tolerance)

    spacegroup_nbs = []
    for key, value in space_group_analyzer.items():
        spacegroup_nb = value.get_space_group_number()
        atoms.info['spacegroup_nb'][str(key)] = spacegroup_nb
        spacegroup_nbs.append(spacegroup_nb)

    return spacegroup_nbs


def get_conventional_std_cell(atoms):
    """Given an ASE atoms object, return the ASE atoms object in the conventional standard cell.

    It uses symmetries to find the conventional standard cell.
    In particular, it gives a structure with a conventional cell according to the standard defined in
    W. Setyawan, and S. Curtarolo, Comput. Mater. Sci.49(2), 299-312 (2010). \n

    This is simply a wrapper around the pymatgen implementation:
    http://pymatgen.org/_modules/pymatgen/symmetry/analyzer.html

    Parameters:

    atoms: `ase.Atoms` object
        Atomic structure.

    Returns:

    `ase.Atoms` object
        Return the structure in a conventional cell.

    .. seealso:: To create a standard cell that it is independent from symmetry operations use
        :py:mod:`ai4materials.utils.utils_crystals.get_conventional_std_cell_no_sym`

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    # save atoms.info dict otherwise it gets lost in the conversion
    atoms_info = atoms.info
    mg_structure = AseAtomsAdaptor.get_structure(atoms)
    finder = SpacegroupAnalyzer(mg_structure)

    mg_structure = finder.get_conventional_standard_structure()
    conventional_standard_atoms = AseAtomsAdaptor.get_atoms(mg_structure)

    conventional_standard_atoms.info = atoms_info

    return conventional_standard_atoms


def get_conventional_std_cell_no_sym(atoms):
    """Given an ASE atoms object, return the ASE atoms object in the conventional standard cell.

    It does NOT use symmetries to obtain the standard cell. \n

    Gives a structure with a conventional cell according to the standard defined for TRICLINIC cells in
    W. Setyawan, and S. Curtarolo, Comput. Mater. Sci.49(2), 299-312 (2010). \n
    The triclinic convention is employed to make sure that no information on the symmetry of the lattice
    is used in the generation of the standard cell. \n

    The code is taken from the triclinic cell case in pymatgen:
    http://pymatgen.org/_modules/pymatgen/symmetry/analyzer.html

    Parameters:

    atoms: `ase.Atoms` object
        Atomic structure.

    Returns:

    `ase.Atoms` object
        Return the structure in a conventional cell (the convention used is the one for triclinic cells)

    .. seealso:: To create a standard cell that exploits symmetry information use
        :py:mod:`ai4materials.utils.utils_crystals.get_conventional_std_cell`

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    # save atoms.info dict otherwise it gets lost in the conversion
    atoms_info = atoms.info
    mg_structure = AseAtomsAdaptor.get_structure(atoms)
    finder = SpacegroupAnalyzer(mg_structure)

    # get structure in the spglib format
    # according to the Spglib documentation
    # (https://atztogo.github.io/spglib/python-spglib.html?highlight=standardize_cell#standardize-cell)
    # ‘no_idealize=True’ disables to idealize lengths and angles of basis vectors and positions of atoms according to crystal symmetry
    # the structure is not refined and thus symmetries are not used
    lattice, scaled_positions, numbers = spglib.standardize_cell(finder._cell, to_primitive=False, no_idealize=True)
    species = [finder._unique_species[i - 1] for i in numbers]
    struct = Structure(lattice, species, scaled_positions).get_sorted_structure()

    # this is the convention for triclinic cells in pymatgen
    # http://pymatgen.org/_modules/pymatgen/symmetry/analyzer.html
    latt = struct.lattice
    a, b, c = latt.lengths_and_angles[0]
    alpha, beta, gamma = [math.pi * i / 180 for i in latt.lengths_and_angles[1]]
    new_matrix = None

    transf = None
    test_matrix = [[a, 0, 0], [b * cos(gamma), b * sin(gamma), 0.0],
                   [c * cos(beta), c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma), c * math.sqrt(
                       sin(gamma) ** 2 - cos(alpha) ** 2 - cos(beta) ** 2 + 2 * cos(alpha) * cos(beta) * cos(
                           gamma)) / sin(gamma)]]

    def is_all_acute_or_obtuse(m):
        recp_angles = np.array(Lattice(m).reciprocal_lattice.angles)
        return np.all(recp_angles <= 90) or np.all(recp_angles > 90)

    if is_all_acute_or_obtuse(test_matrix):
        transf = np.eye(3)
        new_matrix = test_matrix

    test_matrix = [[-a, 0, 0], [b * cos(gamma), b * sin(gamma), 0.0],
                   [-c * cos(beta), -c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma), -c * math.sqrt(
                       sin(gamma) ** 2 - cos(alpha) ** 2 - cos(beta) ** 2 + 2 * cos(alpha) * cos(beta) * cos(
                           gamma)) / sin(gamma)]]

    if is_all_acute_or_obtuse(test_matrix):
        transf = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]
        new_matrix = test_matrix

    test_matrix = [[-a, 0, 0], [-b * cos(gamma), -b * sin(gamma), 0.0],
                   [c * cos(beta), c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma), c * math.sqrt(
                       sin(gamma) ** 2 - cos(alpha) ** 2 - cos(beta) ** 2 + 2 * cos(alpha) * cos(beta) * cos(
                           gamma)) / sin(gamma)]]

    if is_all_acute_or_obtuse(test_matrix):
        transf = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
        new_matrix = test_matrix

    test_matrix = [[a, 0, 0], [-b * cos(gamma), -b * sin(gamma), 0.0],
                   [-c * cos(beta), -c * (cos(alpha) - cos(beta) * cos(gamma)) / sin(gamma), -c * math.sqrt(
                       sin(gamma) ** 2 - cos(alpha) ** 2 - cos(beta) ** 2 + 2 * cos(alpha) * cos(beta) * cos(
                           gamma)) / sin(gamma)]]
    if is_all_acute_or_obtuse(test_matrix):
        transf = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
        new_matrix = test_matrix

    latt = Lattice(new_matrix)

    new_coords = np.dot(transf, np.transpose(struct.frac_coords)).T
    new_struct = Structure(latt, struct.species_and_occu, new_coords, site_properties=struct.site_properties,
                           to_unit_cell=True).get_sorted_structure()

    # put the results back in an ASE structure
    conventional_standard_atoms = AseAtomsAdaptor.get_atoms(new_struct)
    conventional_standard_atoms.info = atoms_info

    return conventional_standard_atoms


def get_primitive_std_cell(atoms):
    """Given an ASE atoms object, return the ASE atoms object in the primitive standard cell.

    It uses symmetries to find the primitive standard cell.
    In particular, it gives a structure with a conventional cell according to the standard defined in
    W. Setyawan, and S. Curtarolo, Comput. Mater. Sci.49(2), 299-312 (2010). \n

    This is simply a wrapper around the pymatgen implementation:
    http://pymatgen.org/_modules/pymatgen/symmetry/analyzer.html

    Parameters:

    atoms: `ase.Atoms` object
        Atomic structure.

    Returns:

    `ase.Atoms` object
        Return the structure in the primitive cell

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    # save atoms.info dict otherwise it gets lost in the conversion
    atoms_info = atoms.info
    mg_structure = AseAtomsAdaptor.get_structure(atoms)

    finder = SpacegroupAnalyzer(mg_structure)
    mg_structure = finder.get_primitive_standard_structure()
    primitive_standard_atoms = AseAtomsAdaptor.get_atoms(mg_structure)
    primitive_standard_atoms.info = atoms_info

    return primitive_standard_atoms


def modify_crystal(ase_atoms, function_to_apply, **kwargs):
    """Apply a transformation to a crystal structure"""
    ase_atoms = function_to_apply(ase_atoms, **kwargs)

    return ase_atoms


def nb_of_replicas(atoms, create_replicas_by, min_nb_atoms, target_nb_atoms=None, max_diff_nb_atoms=None, radius=None,
                   target_replicas=None):
    """Return the number of replicas of the input cell in each direction.

    The number of replicas that are returned will be later used by
    :py:mod:`ai4materials.utils.utils_crystals.create_supercell`
    to actually make a supercell.

    Parameters:

    atoms: `ase.Atoms`
        Atomic structure.

    create_replicas_by: { 'nb_atoms', 'radius', 'user-defined'}
        Method to calculate the replicas. \n
        `create_replicas_by` = 'nb_atoms' creates supercells 'naively' as cubic as possible with the number of atoms
        as close as possible to `target_nb_atoms`. \n
        `create_replicas_by` = 'radius' creates supercells 'naively' as cubic as possible with the number of atoms
        with the given radius. \n

    min_nb_atoms: int
        Minimum number of atoms for the replicated structure.
        If the resulting structure has less atoms, an error will be raised.

    target_nb_atoms: int, optional
        Target number of atoms in the supercell created. The actual number of atoms might differ from it.
        See also `max_diff_nb_atoms` below. \n
        Used only if `create_replicas_by`='nb_atoms'.

    max_diff_nb_atoms: int, optional
        Maximum (absolute) difference between the `target_nb_atoms` and the actual number of atoms present
        in the supercell. If the difference is larger, an error will be raised.
        Used only if `create_replicas_by`='nb_atoms'.

    radius: float, optional
        Used only if `create_replicas_by`='radius'.

    target_replicas: int or list/tuple of int, optional
        Number of replicas of the created supercell. \n
        `target_replicas` = 3 replicates the cell 3 times along the 3 directions \n
        `target_replicas` = (4,2,2) replicates the cell 4 times along the 1st direction, and 2 times along the other
        two directions. \n
        Used only if `create_replicas_by`='user-defined'.

    Returns:

    list of int
        List of three integers corresponding to the number of repetition of the input cell along
        the three dimensions.

    .. seealso:: Algorithms for supercell determination are found in
        :py:mod:`ai4materials.utils.utils_crystals.nb_atoms_to_replicas` and
        :py:mod:`ai4materials.utils.utils_crystals.radius_to_replicas`.\n
        Starting from the number of replicas here, the actual supercell is made in
        :py:mod:`ai4materials.utils.utils_crystals.create_supercell`

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    if create_replicas_by == 'nb_atoms':
        replicas = nb_atoms_to_replicas(atoms, min_nb_atoms, target_nb_atoms, max_diff_nb_atoms)
    elif create_replicas_by == 'radius':
        replicas = radius_to_replicas(atoms, min_nb_atoms, radius)
    elif create_replicas_by == 'user-defined':
        replicas = target_replicas
    else:
        raise ValueError("Invalid value for create_replicas_by. Possible values are nb_atoms, radius, user-defined.")

    return replicas


def spacegroup_a_to_spacegroup_b(atoms, spgroup_a, spgroup_b, target_b_contribution, create_replicas_by,
                                 min_nb_atoms=None, target_nb_atoms=None, max_diff_nb_atoms=None, radius=None,
                                 target_replicas=None, max_rel_error=0.01, **kwargs):
    """Remove central atoms for bcc to sc"""

    # get number of replicas
    replicas = nb_of_replicas(atoms, create_replicas_by=create_replicas_by, min_nb_atoms=min_nb_atoms,
                              target_nb_atoms=target_nb_atoms, max_diff_nb_atoms=max_diff_nb_atoms, radius=radius,
                              target_replicas=target_replicas)

    atoms = standardize_cell(atoms, **kwargs)

    # make a spgroup_a-type supercell before removing atoms
    atoms_a = atoms.copy()
    atoms_a = atoms_a * replicas

    # check initial spacegroup
    mg_structure = AseAtomsAdaptor.get_structure(atoms)
    finder = SpacegroupAnalyzer(mg_structure)
    init_spgroup = finder.get_space_group_symbol()

    if init_spgroup == spgroup_a:
        logger.debug('Initial spacegroup is {0} as expected'.format(init_spgroup))
    else:
        raise Exception("Initial spacegroup is {0} "
                        "while the expected spacegroup is {1}".format(init_spgroup, spgroup_a))

    # initially the mix structure has all the spgroup_a atoms
    atoms_mix = atoms_a.copy()

    idx_remove_list = []
    TOL = 1e-03

    if spgroup_a == 'Im-3m' and spgroup_b == 'Pm-3m':
        # from bcc to simple cubic
        for idx in range(atoms.get_number_of_atoms()):
            # deleting all atoms from spgroup_a to go in spgroup_b
            # removing the atoms that are in position (0.0, 0.0, 0.0)
            if (abs(atoms.positions[idx][0]) <= TOL and abs(atoms.positions[idx][1]) <= TOL and abs(
                    atoms.positions[idx][2]) <= TOL):
                pass
            else:
                idx_remove_list.append(idx)

    elif spgroup_a == 'Fd-3m' and spgroup_b == 'Fm-3m':
        # from diamond to fcc
        for idx in range(atoms.get_number_of_atoms()):
            # deleting all atoms from spgroup_a to go in spgroup_b
            # removing the atoms that are "inside" the cube
            # keep only the atoms that have one coordinate which is
            # 1/2 of the cell length or position (0.0, 0.0, 0.0)
            cell_length = atoms.get_cell_lengths_and_angles()[0]
            if abs(atoms.positions[idx][0] - cell_length / 2.0) <= TOL or abs(
                atoms.positions[idx][1] - cell_length / 2.0) <= TOL or abs(
                    atoms.positions[idx][2] - cell_length / 2.0) <= TOL:
                pass
            elif (abs(atoms.positions[idx][0]) <= TOL and abs(atoms.positions[idx][1]) <= TOL and abs(
                    atoms.positions[idx][2]) <= TOL):
                pass
            else:
                idx_remove_list.append(idx)
    else:
        raise NotImplementedError("Transformation from spacegroup {0} to spacegroup {1}"
                                  "is not implemented".format(spgroup_a, spgroup_b))

    # delete all the indices added to the list
    del atoms[[atom.index for atom in atoms if atom.index in idx_remove_list]]

    atoms_b = atoms * replicas

    # check final spacegroup
    mg_structure = AseAtomsAdaptor.get_structure(atoms_b)
    finder = SpacegroupAnalyzer(mg_structure)
    final_spgroup = finder.get_space_group_symbol()

    if final_spgroup == spgroup_b:
        logger.debug('Final spacegroup is {0} as expected'.format(final_spgroup))
    else:
        logger.debug("Final spacegroup is {0}".format(final_spgroup))
        logger.debug("Expected final spacegroup is {0}".format(spgroup_b))
        raise Exception("The transformation provided does not give the expected final "
                        " spacegroup. Expected: {0}; obtained: {1}".format(spgroup_b, final_spgroup))

    # find the rows that are in bcc-type supercell and not in sc
    atoms_a_rows = atoms_a.positions.view([('', atoms_a.positions.dtype)] * atoms_a.positions.shape[1])
    atoms_b_rows = atoms_b.positions.view([('', atoms_b.positions.dtype)] * atoms_b.positions.shape[1])
    a_b_diff_pos = np.setdiff1d(atoms_a_rows, atoms_b_rows).view(atoms_a.positions.dtype).reshape(-1,
                                                                                                  atoms_a.positions.shape[
                                                                                                      1])

    atoms_a_only_ids = []
    for idx in range(atoms_a.get_number_of_atoms()):
        for row_idx in range(a_b_diff_pos.shape[0]):
            if np.allclose(atoms_a.positions[idx], a_b_diff_pos[row_idx, :], rtol=1e-03):
                atoms_a_only_ids.append(idx)
                break
            else:
                pass

    # take a random subset of atoms to remove
    nb_atoms_to_rm = int(len(atoms_a_only_ids) * target_b_contribution)
    actual_b_contribution = nb_atoms_to_rm / len(atoms_a_only_ids)

    if target_b_contribution != 0.0:
        rel_error = abs(target_b_contribution - actual_b_contribution) / target_b_contribution

        if rel_error > max_rel_error:
            logger.warning("Difference between target and actual vacancy ratio "
                           "bigger than the threshold ({0}%).\n"
                           "Target/actual vacancy ratio: {1}%/{2}%.".format(max_rel_error * 100.0,
                                                                            target_b_contribution * 100.0,
                                                                            actual_b_contribution * 100.0))

    # random sampling of the list without replacement
    atoms_a_only_ids_subset = random.sample(atoms_a_only_ids, nb_atoms_to_rm)

    # remove atoms from the bcc_atoms_only_ids
    del atoms_mix[[atom.index for atom in atoms_mix if atom.index in atoms_a_only_ids_subset]]

    return atoms_mix


def nb_atoms_to_replicas(atoms, min_nb_atoms, target_nb_atoms, silent=True, max_diff_nb_atoms=100):
    """Calculate the replication that gives a supercell 'naively' as cubic as possible with the number of atoms
        as close as possible to `target_nb_atoms`.

    The algorithm calculates the extension of the structure along the 3 dimensions and add a repetition
    along the direction that has the smaller extension.

    For information regarding the parameters, please refer to
    :py:mod:`ai4materials.utils.utils_crystals.nb_of_replicas`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    nb_atoms = len(atoms)
    atoms_new = copy.deepcopy(atoms)
    replicas = np.array([1, 1, 1])

    while len(atoms_new) < target_nb_atoms:
        max_distance = np.absolute((np.amax(atoms_new.positions, axis=0) - np.amin(atoms_new.positions, axis=0)))
        smallest_idx = np.argmin(max_distance)
        replicas[smallest_idx] = replicas[smallest_idx] + 1
        atoms_new = atoms * replicas

    nb_atoms_replicas = nb_atoms * replicas[0] * replicas[1] * replicas[2]
    diff_nb_atoms = target_nb_atoms - nb_atoms_replicas

    if not silent:
        if diff_nb_atoms > max_diff_nb_atoms:
            logger.warning("Difference between target and actual nb_atoms in "
                           "supercell generation is greater than threshold. \n"
                           "Initial nb_atoms: {0}; target nb_atoms: {1}; actual nb_atoms: {2}; \n"
                           "actual_nb_difference: {3}; max nb_difference: {4} \n"
                           "replicas: {5}".format(nb_atoms, target_nb_atoms, nb_atoms_replicas, diff_nb_atoms,
                                                  max_diff_nb_atoms, replicas))

    if nb_atoms_replicas < min_nb_atoms:
        raise ValueError("Structure has less than the required number of atoms. "
                         "It has {0} atoms instead of {1} ".format(nb_atoms_replicas, min_nb_atoms))

    return replicas


def radius_to_replicas(atoms, min_nb_atoms, radius):
    """Calculate the replication that gives a supercell 'naively' as cubic as possible to the required `radius`.

    It approaches `radius` from below.

    The algorithm calculates how many cells can be fit within the specified radius for each dimension.

    For information regarding the parameters, please refer to
    :py:mod:`ai4materials.utils.utils_crystals.nb_of_replicas`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    cell_lengths = atoms.get_cell_lengths_and_angles()[:3]
    replicas = [int(radius_i // cell_length) for cell_length, radius_i in zip(cell_lengths, radius)]

    nb_atoms = len(atoms)
    nb_atoms_replicas = nb_atoms * replicas[0] * replicas[1] * replicas[2]

    if nb_atoms_replicas < min_nb_atoms:
        raise ValueError("Structure has less than the required number of atoms. "
                         "It has {0} atoms instead of {1} ".format(nb_atoms_replicas, min_nb_atoms))

    return replicas


# def _align_supercell(atoms):
#     """Coherent point drift registration. Only a stub, it does not work."""
#     import pycpd
#     from functools import partial
#
#     phi = np.random.uniform() * 360.0
#     theta = np.random.uniform() * 180.0
#     psi = np.random.uniform() * 360.0
#     atoms = rotate_atoms(atoms, phi=phi, theta=theta, psi=psi)
#
#     logger.debug("Euler angles for rotation. phi: {}, theta {}, psi {}".format(phi, theta, psi))
#
#     positions = atoms.get_positions()
#     atoms_rot = rotate_atoms(atoms, phi=phi, theta=theta, psi=psi)
#     rotated_positions = atoms_rot.get_positions()
#
#     def visualize(iteration, error, coord, coord_rot, ax_plot):
#         plt.cla()
#         ax_plot.scatter(coord[:, 0], coord[:, 1], coord[:, 2], color='red')
#         ax_plot.scatter(coord_rot[:, 0], coord_rot[:, 1], coord_rot[:, 2], color='blue')
#         plt.draw()
#         print("iteration %d, error %.5f" % (iteration, error))
#         plt.pause(0.001)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     callback = partial(visualize, ax_plots=ax)
#
#     reg = pycpd.rigid_registration(positions, rotated_positions, maxIterations=1e6, tolerance=1.e-10)
#     reg.register(callback)
#     plt.show()


def standardize_cell(atoms, cell_type):
    """ Standardize the cell of the atomic structure.

    Parameters:

    atoms: `ase.Atoms`
        Atomic structure.

    cell_type: { 'standard', 'standard_no_symmetries', 'primitive', None}
        Starting from the input cell, creates a standard cell according to same standards
        before the supercell generation. \n
        `cell_type` = 'standard' creates a standard conventional cell.
        See :py:mod:`ai4materials.utils.utils_crystals.get_conventional_std_cell`.  \n
        `cell_type` = 'standard_no_symmetries' creates a standard conventional cell without using symmetries.
        See :py:mod:`ai4materials.utils.utils_crystals.get_conventional_std_cell_no_sym`.  \n
        `cell_type` = 'primitive' creates a standard primitive cell.
        See :py:mod:`ai4materials.utils.utils_crystals.get_primitive_std_cell`. \n
        `cell_type` = `None` does not creates any cell.
        It simply uses the unit cell as input for the supercell generation.

    Returns:

    `ase.Atoms`
        Atomic structure in the standard cell of the selected type.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    if cell_type == 'standard':
        atoms = get_conventional_std_cell(atoms)
    elif cell_type == 'standard_no_symmetries':
        atoms = get_conventional_std_cell_no_sym(atoms)
    elif cell_type == 'primitive':
        atoms = get_primitive_std_cell(atoms)
    elif cell_type is None:
        pass
    else:
        raise ValueError("Unrecognized cell_type value.")

    return atoms


def create_supercell(atoms, create_replicas_by='nb_atoms', min_nb_atoms=None, target_nb_atoms=None,
                     max_diff_nb_atoms=100, random_rotation_before=False, random_rotation=False,
                     cell_type=None,
                     optimal_supercell=False, radius=None, target_replicas=None):
    """Create a supercell specifying using a specified method.

    Parameters:

    atoms: `ase.Atoms`
        Atomic structure.

    create_replicas_by: { 'nb_atoms', 'radius', 'user-defined'}
        Method to calculate the replicas. \n
        `create_replicas_by` = 'nb_atoms' creates a supercell 'naively' as cubic as possible with the number of atoms
        as close as possible to `target_nb_atoms`. \n
        `create_replicas_by` = 'radius' creates a supercell 'naively' as cubic as possible with the number of atoms
        with the given radius. \n
        `create_replicas_by` = 'user-defined' creates a supercell using the user-defined `target_replicas`
        integer of tuple to replicate the cell. See `target_replicas` below.

    min_nb_atoms: int
        Minimum number of atoms for the replicated structure.
        If the resulting structure has less atoms, an error will be raised.

    target_nb_atoms: int, optional
        Target number of atoms in the supercell created. The actual number of atoms might differ from it.
        See also `max_diff_nb_atoms` below.
        Used only if `create_replicas_by`='nb_atoms'.

    max_diff_nb_atoms: int, optional
        Maximum (absolute) difference between the `target_nb_atoms` and the actual number of atoms present
        in the supercell. If the difference is larger, an error will be raised.
        Used only if `create_replicas_by`='nb_atoms'.

    radius: float, optional
        Used only if `create_replicas_by`='radius'.

    target_replicas: int or list/tuple of int, optional
        Number of replicas of the created supercell. \n
        `target_replicas` = 3 replicates the cell 3 times along the 3 directions \n
        `target_replicas` = (4,2,2) replicates the cell 4 times along the 1st direction, and 2 times along the other
        two directions.
        Used only if `create_replicas_by`='user-defined'.

    random_rotation_before: bool, optional (default = `False`)
        Randomly rotate the structure sequentially along the x, y, and z axis by random angles BEFORE the
        supercell is created.
        Both the atomic positions and the cell are rotated so that periodic boundary conditions are respected.

    random_rotation: bool, optional (default = `False`)
        Randomly rotate the structure sequentially along the x, y, and z axis AFTER the supercell is created. \n
        Both the atomic positions and the cell are rotated so that periodic boundary conditions are respected.

    cell_type: { 'standard', 'standard_no_symmetries', 'primitive', None}
        Starting from the input cell, creates a standard cell according to same standards
        before the supercell generation. \n
        `cell_type` = 'standard' creates a standard conventional cell.
        See :py:mod:`ai4materials.utils.utils_crystals.get_conventional_std_cell`.  \n
        `cell_type` = 'standard_no_symmetries' creates a standard conventional cell without using symmetries.
        See :py:mod:`ai4materials.utils.utils_crystals.get_conventional_std_cell_no_sym`.  \n
        `cell_type` = 'primitive' creates a standard primitive cell.
        See :py:mod:`ai4materials.utils.utils_crystals.get_primitive_std_cell`. \n
        `cell_type` = `None` does not creates any cell.
        It simply uses the unit cell as input for the supercell generation.

    optimal_supercell: bool, optional (default = `False`)
        Create a supercell with is 'optimally' as close as possible to a simple cubic supercell. It can be slow. \n
        For more information, please visit:
        https://wiki.fysik.dtu.dk/ase/tutorials/defects/defects.html
        and the associated publication.

    Returns:

    `ase.Atoms`
        Atomic structure of the supercell.

    .. seealso:: Details on the supercell generation can be found in:
        :py:mod:`ai4materials.utils.utils_crystals.nb_of_replicas`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    # the Python random instead of the numpy random is used because numpy does not get reseeded in multiprocessing
    # https://github.com/numpy/numpy/issues/9650
    if random_rotation_before:
        alpha = random.random() * 360.0
        atoms.rotate(alpha, 'x', rotate_cell=True, center='COU')

        beta = random.random() * 360.0
        atoms.rotate(beta, 'y', rotate_cell=True, center='COU')

        gamma = random.random() * 360.0
        atoms.rotate(gamma, 'z', rotate_cell=True, center='COU')

        logger.debug("Structure rotated randomly by:")
        logger.debug("{}° around x-axis; {}° around y-axis; {}° around z-axis".format(alpha, beta, gamma))

    atoms = standardize_cell(atoms, cell_type)

    if optimal_supercell:
        logger.info("Using optimal supercell algorithm for replica determination. ")
        # optimal supercell following https://wiki.fysik.dtu.dk/ase/tutorials/defects/defects.html
        # there is a scipy-based implementation (find_optimal_cell_shape) but it does not work
        # Wave is now a separate package (pip install wave), but ASE still try to import it from scipy
        target_size = int(target_nb_atoms / len(atoms))
        p_opt = find_optimal_cell_shape_pure_python(cell=atoms.cell, target_size=target_size, target_shape='sc',
                                                    verbose=False)
        dev_cubic = get_deviation_from_optimal_cell_shape(np.dot(p_opt, atoms.cell))

        logger.debug("Optimal repetition matrix: {}".format(p_opt))
        logger.debug("Optimality measure (perfect cubic structure -> 0.0): {}".format(dev_cubic))
        atoms = make_supercell(atoms, p_opt)

    else:
        replicas = nb_of_replicas(atoms, create_replicas_by=create_replicas_by, min_nb_atoms=min_nb_atoms,
                                  target_nb_atoms=target_nb_atoms, max_diff_nb_atoms=max_diff_nb_atoms, radius=radius,
                                  target_replicas=target_replicas)
        atoms = atoms * replicas

    if random_rotation:
        alpha = random.random() * 360.0
        atoms.rotate(alpha, 'x', rotate_cell=True, center='COU')

        beta = random.random() * 360.0
        atoms.rotate(beta, 'y', rotate_cell=True, center='COU')

        gamma = random.random() * 360.0
        atoms.rotate(gamma, 'z', rotate_cell=True, center='COU')

        logger.debug("Structure rotated randomly by:")
        logger.debug("{}° around x-axis; {}° around y-axis; {}° around z-axis".format(alpha, beta, gamma))

    # wraps atoms back to the cell
    atoms.wrap()

    return atoms


def rotate_atoms(atoms, phi=0.0, theta=0.0, psi=0.0, center='COU'):
    """Randomly rotate an ASE atoms structure via Euler angles in degree.

    This is simply a wrapper around:
    https://wiki.fysik.dtu.dk/ase/ase/atoms.html#ase.Atoms.euler_rotate

    For more info on Euler angles and for the convention followed by ASE:
    http://mathworld.wolfram.com/EulerAngles.html

    Parameters:

    atoms: `ase.Atoms`
        Atomic structure to be rotated

    center: sequence of length 3 or str
        The point to rotate about. A sequence of length 3 with the
        coordinates of the point, or:
        'COM' to select the center of mass \n
        'COP' to select center of positions \n
        'COU' to select center of cell.

    phi: float
        The 1st rotation angle around the z axis.

    theta: float
        Rotation around the x axis.

    psi: float
        2nd rotation around the z axis.

    Returns:
    `ase.Atoms` object
        The rotated atomic structure.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    atoms.euler_rotate(phi=phi, theta=theta, psi=psi, center=center)

    return atoms


def create_vacancies(atoms, target_vacancy_ratio, max_rel_error=0.25, **kwargs):
    """Make a supercell and then create vacancies in the constructed supercell.

    It is implicitly without replacement because the atoms are deleted
    at each iteration. This is why, for example, the implementation is different w.r.t.
    substitute atoms.

    Parameters:

    atoms: `ase.Atoms`
        Atomic structure.

    target_vacancy_ratio: float
        Target percentage of vacancies. It must be a number between 0.0 (0.0 excluded)
        and 1.0 (all atoms removed). For example, 0.2 will lead to the removal of 20% of the atoms. \n
        The actual number of vacancies might differ from it, especially for
        small supercell and/or small percentage of vacancies. See also `max_rel_error` below.

    max_rel_error: float, optional (default = 0.25)
        Relative (absolute) difference between the `target_vacancy_ratio` and the actual percentage of atoms
        created in the supercell. If the difference is larger, a warning will be raised.

    Returns:

    `ase.Atoms`
        Supercell with vacancies.

    .. seealso:: The supercell is generated by:
        :py:mod:`ai4materials.utils.utils_crystals.create_supercell`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    atoms = create_supercell(atoms, **kwargs)

    nb_atoms = len(atoms)
    if nb_atoms > 0:
        if 0. < target_vacancy_ratio < 1.:
            # calculate the number of vancancies to make given the ratio
            nb_vacancies = int(nb_atoms * target_vacancy_ratio)
            actual_vacancy_ratio = nb_vacancies / nb_atoms

            # randomly remove one atom from the supercell
            for i in range(nb_vacancies):
                idx_atom_to_delete = random.choice([atom.index for atom in atoms])
                del atoms[idx_atom_to_delete]

            rel_error = abs(target_vacancy_ratio - actual_vacancy_ratio) / target_vacancy_ratio

            if rel_error > max_rel_error:
                logger.warning("Difference between target and actual vacancy ratio "
                               "bigger than the threshold ({0}%).\n"
                               "Target/actual vacancy ratio: {1}%/{2}%.".format(max_rel_error * 100.0,
                                                                                target_vacancy_ratio * 100.0,
                                                                                actual_vacancy_ratio * 100.0))
                logger.warning("Number of atoms (before vacancies): {}".format(nb_atoms))
                logger.warning("Number of vacancies: {}".format(nb_vacancies))

        else:
            raise ValueError('The vacancy ratio needs to be between 0 and 1. (0. excluded)')
    else:
        logger.warning('Structure with no atoms. Continuing.')

    return atoms


def substitute_atoms(atoms, target_sub_ratio=0.0, max_n_sub_species=94, max_rel_error=0.25, **kwargs):
    """Make a supercell and then substitute in the constructed supercell.

    Atoms are substituted with other - randomly chosen - chemical species. The positions of the atoms in the
    lattice are not changed, only their chemical identity.

    Parameters:

    atoms: `ase.Atoms`
        Atomic structure.

    target_sub_ratio: float
        Target percentage of atoms to be substituted. It must be a number between 0.0 (no vacancies created)
        and 1.0 (all atoms removed). For example, 0.2 will lead to the substitutions of 20% of the atoms. \n
        The actual number of substitutions might differ from it, especially for
        small supercell and/or small percentage of substitutions. See also `max_rel_error` below.

    max_n_sub_species: int, optional, (default = 94)
        The maximum number of species that can be used in the substitution procedure.
        For example: \n
        - `max_n_sub_species` = 1, will created a disordered binary. \n
        - `substitution_ratio` = 0.50 and `max_n_sub_species` = 1, will create a disordered binary with chemical species
        having the same stoichiometry. \n
        - `max_n_sub_species` = 2, a disordered ternary will be created.

    max_rel_error: float, optional (default = 0.25)
        Relative (absolute) difference between the `target_sub_ratio` and the actual percentage of atoms
        substituted in the supercell. If the difference is larger, a warning will be raised.

    Returns:

    `ase.Atoms`
        Supercell with randomly substituted atoms.

    .. seealso:: The supercell is generated by:
        :py:mod:`ai4materials.utils.utils_crystals.create_supercell`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    nb_atoms = len(atoms)
    if nb_atoms > 0:
        atoms = create_supercell(atoms, **kwargs)

        if 0. < target_sub_ratio < 1.:
            # calculate the number of atoms to substitute to make given the ratio
            nb_atoms = len(atoms)
            nb_subs = int(nb_atoms * target_sub_ratio)
            actual_sub_ratio = nb_subs / nb_atoms

            # pick atomic numbers from 1 to 94
            possible_atom_numbers = np.random.choice(np.arange(1, 95, dtype=np.int16), size=max_n_sub_species)
            new_atomic_numbers = np.random.choice(possible_atom_numbers, size=nb_subs)

            # without replacement
            idx_atom_to_change = np.random.choice(np.arange(0, len(atoms)), size=nb_subs, replace=False)

            for i in range(nb_subs):
                atoms[idx_atom_to_change[i]].number = new_atomic_numbers[i]

            rel_error = abs(target_sub_ratio - actual_sub_ratio) / target_sub_ratio

            if rel_error > max_rel_error:
                logger.warning("Difference between target and actual substitution ratio "
                               "bigger than the threshold ({0}%).\n"
                               "Target/actual substitution ratio: {1}%/{2}%.".format(max_rel_error * 100.0,
                                                                                     target_sub_ratio * 100.0,
                                                                                     actual_sub_ratio * 100.0))
        else:
            raise ValueError('The substitution ratio needs to be comprised between 0.0 and 1.0.')
    else:
        logger.warning('Structure with no atoms. Continuing.')

    return atoms


def random_displace_atoms(atoms, noise_distribution, displacement=None, displacement_scaled=None, **kwargs):
    """Make a supercell and then randomly displace atoms in the constructed supercell.

    Atoms are substituted with other - randomly chosen - chemical species. The positions of the atoms in the
    lattice are not changed, only their chemical identity.

    Parameters:

    atoms: `ase.Atoms`
        Atomic structure.

    displacement: float, optional
        The parameter used in the noise distribution to generate random displacements. In Angstrom.
        Used only if `noise_distribution` = gaussian or `noise_distribution` = uniform.

    displacement_scaled: float, optional
        The parameter used in the noise distribution to generate random displacements. Pure number.
        For example, if `displacement_scaled` = 0.1, atoms will be displaced according to a Gaussian
        distribution with standard deviation equal to 10% of (approximately) the bond length.
        Used only if `noise_distribution` = gaussian_scaled.

    noise_distribution: { 'gaussian', 'uniform', 'gaussian_scaled', 'uniform_scaled'}
        The type of noise distribution used to randomly displace atoms. \n
        - 'gaussian': displace atoms by values sampled from a Gaussian distribution with
        standard deviation equal to `displacement`. \n
        - 'uniform': displace atoms by values sampled from a distribution [-`displacement`, +`displacement`] \n
        - 'gaussian_scaled': displace atoms by a Gaussian distribution with standard deviation
        equal to `displacement` scaled by the nearest neighbors distance. \n
        - 'uniform_scaled': displace atoms by values sampled from a distribution [-`displacement`, +`displacement`]
        scaled by the nearest neighbors distance \n

    max_rel_error: float, optional (default = 0.25)
        Relative (absolute) difference between the `target_sub_ratio` and the actual percentage of atoms
        substituted in the supercell. If the difference is larger, a warning will be raised.

    Returns:

    `ase.Atoms`
        Supercell with randomly displaced atoms.

    .. seealso:: The supercell is generated by:
        :py:mod:`ai4materials.utils.utils_crystals.create_supercell`. \n
        If `noise_distribution` = gaussian_scaled, a quantity related to the average nearest neighbor distance
        is calculated. It is not exactly the average nearest neighbor distance because we use quantiles to be more
        robust with respect to defect. For more details, please go to
        :py:mod:`ai4materials.utils.utils_crystals.get_nn_distance`. \n

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    nb_atoms = len(atoms)
    if nb_atoms > 0:
        atoms = create_supercell(atoms, **kwargs)

        if noise_distribution == 'gaussian':
            noise = np.random.normal(loc=0.0, scale=displacement, size=(len(atoms), 3))
            logger.debug("Gaussian displacement with standard deviation {}".format(displacement))
            logger.debug("Noise realization: min: {}; max: {}".format(noise.min(), noise.max()))
        elif noise_distribution == 'uniform':
            noise = np.random.uniform(low=-displacement, high=displacement, size=(len(atoms), 3))
        elif noise_distribution == 'gaussian_scaled':
            scale_factor = get_nn_distance(atoms)
            displacement = displacement_scaled * scale_factor
            noise = np.random.normal(loc=0.0, scale=displacement, size=(len(atoms), 3))
        elif noise_distribution == 'uniform_scaled':
            scale_factor = get_nn_distance(atoms)
            displacement = displacement_scaled * scale_factor
            noise = np.random.uniform(low=-displacement, high=displacement, size=(len(atoms), 3))
            logger.debug("Noise realization: min: {}; max: {}".format(noise.min(), noise.max()))
        else:
            raise NotImplementedError("The noise distribution chosen is not implemented.")

        atoms.set_positions(atoms.get_positions() + noise)
        # wrap atomic positions inside unit cell
        atoms.wrap()

    else:
        logger.warning('Structure with no atoms. Continuing.')

    return atoms


def grouped(iterable, n):
    """s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."""
    return izip(*[iter(iterable)] * n)


def get_min_distance(atoms, nb_splits=100):
    """Calculate the smallest distance between atoms in a given structure.

    Here we use the scipy implementation. Other implementations could also be used.
    For example: sklearn.metrics.pairwise.pairwise_distances or ase atoms.get_all_distances().
    Calculating all pairwise distances scales as N^2, where N is the number of atoms in the structure.
    This is clearly memory intensive, so we need to split the atoms in batches if we run out of memory.

    Parameters:

    atoms: `ase.Atoms`
        Atomic structure.

    nb_splits: int, optional, (default = 100)
        Number of splits in which to divide the distance matrix in order to calculate the smallest
        distance between atoms in the atomic structure.

    Returns:

    float or None
        Smallest distance between the atoms in the structures. It returns `None` if the shortest distance cannot
        be calculated, for example because there are no atoms in the structure.

    .. seealso:: :py:mod:`ai4materials.utils.utils_crystals.get_nn_distance`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    nb_atoms = len(atoms)
    dist = None

    if nb_atoms > 0:
        pos = atoms.get_positions()
        num_elems = np.prod(pos.shape)
        size_mb = (num_elems ** 2) * 8.0 / (1024 ** 2)

        if size_mb < 1.0 * 1024:
            # less than 1Gb
            dist = scipy.spatial.distance.pdist(pos, 'euclidean')
            dist_sort = np.sort(dist)
            shortest_distance = dist_sort[0]
        else:
            # if the matrix is too big, splits the atoms
            logger.debug("Matrix dimension: {0:.1f} MB".format(size_mb))
            logger.debug('Switching to low-memory requirement algorithm (slower).')
            if nb_splits is None:
                nb_splits = int(size_mb // (1.0 * 1024)) + 1
            pos_list = np.array_split(pos, nb_splits)
            dist_sort = []
            for pos_list_1, pos_list_2 in grouped(pos_list, 2):
                pos_list = np.vstack((pos_list_1, pos_list_2))
                dist = scipy.spatial.distance.pdist(pos_list, 'euclidean')
                dist_sort.append(np.sort(dist)[0])

            shortest_distance = min(dist_sort)

        del dist

    else:
        shortest_distance = None
        logger.warning('Structure with no atoms. Continuing.')

    return shortest_distance


def get_nn_distance(atoms, distribution='quantile_nn', cutoff=4.0,
                    min_nb_nn=5, pbc=True, plot_histogram=False, bins=100, constrain_nn_distances=True, nn_distances_cutoff=0.9):
    """Calculate an "averaged" (actual average or quantile-based) nearest neighbors distance.
    This is a measure of the characteristic structural lengthscale of the system.

    Parameters:

    atoms: `ase.Atoms`
        Atomic structure.

    distribution: { 'avg_nn', 'quantile_nn'}
        Type of statistical function to be used in the lengthscale determination. \n
        - 'avg_nn' simply averages the nearest neighbor distances.
        - 'quantile_nn' used a quantile-based approach to be more robust w.r.t. outliers.
        These two choices are essentially equivalent for pristine structures, while the 'quantile_nn'
        is more robust when defects are included.

    cutoff: float, optional, (default = 4.0)
        Cutoff (in Angstrom) for the radius within which atoms are considered neighbor.
        This neighbors will then be used to identify the nearest atom.

    min_nb_nn: int, optional, (default = 5)
        Minimum number of neighbors for a given atom to be considered.
        If an atom has less than `min_nb_nn` it will not be used in the determination of the system lengthscale.

    pbc: bool, optional, (default = True)
        `True` if periodic boundary conditions are used.

    plot_histogram: bool, optional, (default = True)
        If `True`, plot the histogram on the neighbor distances. It can be useful for debugging, especially
        for heavily defective structures, when the distribution changes substantially (w.r.t. the
        pristine crystal structure) due to disorder.

    bins: int, optional, (default = 100)
        Number of bins used in the histograms of the nearest neighbor distance function.

    constrain_nn_distances: bool, optional, (default = True)
        If `True`, nearest neighbor distances below the cutoff specified by the additional argument
        nn_distances_cutoff will be ignored.
        
    nn_distances_cutoff: float, optional, (default = 0.9)
        Cutoff for nearest neighbor distances.

    Returns:

    float or None
        Characteristic lengthscale of the system based on the nearest neighbors' distance.
        Returns `None` if no characteristic lengthscale could be found.

    .. seealso:: :py:mod:`ai4materials.utils.utils_crystals.get_min_distance`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    if not pbc:
        atoms.set_pbc((False, False, False))

    nb_atoms = atoms.get_number_of_atoms()
    cutoffs = np.ones(nb_atoms) * cutoff
    nl = NeighborList(cutoffs, skin=0.1, self_interaction=False, bothways=False)
    nl.build(atoms)

    nn_dist = []

    for idx in range(nb_atoms):
        logger.debug("List of neighbors of atom number {0}".format(idx))
        indices, offsets = nl.get_neighbors(idx)

        if len(indices) > min_nb_nn:
            coord_central_atom = atoms.positions[idx]
            # get positions of nearest neighbors within the cut-off
            dist_list = []
            for i, offset in zip(indices, offsets):
                # center each neighbors wrt the central atoms
                coord_neighbor = atoms.positions[i] + np.dot(offset, atoms.get_cell())
                # calculate distance between the central atoms and the neighbors
                dist = np.linalg.norm(coord_neighbor - coord_central_atom)
                dist_list.append(dist)

            # dist_list is the list of distances from the central_atoms
            if len(sorted(dist_list)) > 0:
                # get nearest neighbor distance
                nn_dist.append(sorted(dist_list)[0])
            else:
                logger.warning("List of neighbors is empty for some atom. Cutoff must be increased.")
                return None
        else:
            logger.debug("Atom {} has less than {} neighbours. Skipping.".format(idx, min_nb_nn))


    if constrain_nn_distances:
         # Select all nearest neighbor distances larger than nn_distances_cutoff
         threshold_indices = np.array(nn_dist) > nn_distances_cutoff 
         nn_dist = np.extract(threshold_indices , nn_dist)


    if distribution == 'avg_nn':
        length_scale = np.mean(nn_dist)
    elif distribution == 'quantile_nn':
        # get the center of the maximally populated bin
        hist, bin_edges = np.histogram(nn_dist, bins=bins, density=False)

        # scale by r**2 because this is how the rdf is defined
        # the are of the spherical shells grows like r**2
        hist_scaled = []
        for idx_shell, hist_i in enumerate(hist):
            hist_scaled.append(float(hist_i)/(bin_edges[idx_shell]**2))

        length_scale = (bin_edges[np.argmax(hist_scaled)] + bin_edges[np.argmax(hist_scaled) + 1]) / 2.0

        if plot_histogram:
            # this histogram is not scaled by r**2, it is only the count
            plt.hist(nn_dist, bins=bins)  # arguments are passed to np.histogram
            plt.title("Histogram")
            plt.show()
    else:
        raise ValueError("Not recognized option for atoms_scaling. "
                         "Possible values are: 'min_nn', 'avg_nn', or 'quantile_nn'.")

    return length_scale


def scale_structure(atoms, scaling_type, atoms_scaling_cutoffs, min_scale_factor=0.1, max_scale_factor=10.,
                    extrinsic_scale_factor=1.0):
    """Scale an atomic structure by a given scalar determined based on nearest neighbors distance.

    Parameters:

    atoms: `ase.Atoms`
        Atomic structure.

    scaling_type: { 'min_nn', 'avg_nn', 'quantile_nn'}
        Type of scaling used in the atom structure scaling.

    atoms_scaling_cutoffs: list of floats
        List of cutoffs to be used in the determination of the lengthscale of the system in
        :py:mod:`ai4materials.utils.utils_crystals.get_nn_distance`. If the lengthscale calculation is not successful,
        the next cutoff (next elements in the list `atoms_scaling_cutoffs` is used.

    min_scale_factor: float, optional, (default = 0.5)
        If the calculated scale factor is below this value, the scaling will not be performed. In Angstrom.
        The next element in the `atoms_scaling_cutoffs` list will be used.
        This is used as a safety check because the scaling factor should correspond to physically motivated
        nearest neighbor distance in materials.

    max_scale_factor: float, optional, (default = 10.)
        If the calculated scale factor is above this value, the scaling will not be performed. In Angstrom.
        The next element in the `atoms_scaling_cutoffs` list will be used.
        This is used as a safety check because the scaling factor should correspond to physically motivated
        nearest neighbor distance in materials.

    Returns:

    `ase.Atoms`
        Scaled atomic structure.

    .. seealso:: :py:mod:`ai4materials.utils.utils_crystals.get_min_distance`, :py:mod:`ai4materials.utils.utils_crystals.get_nn_distance`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    scale_factor = None
    if scaling_type == 'min_nn':
        scale_factor = get_min_distance(atoms)
    elif scaling_type == 'avg_nn' or scaling_type == 'quantile_nn':
        for idx_cutoff, cutoff in enumerate(atoms_scaling_cutoffs):
            scale_factor = get_nn_distance(atoms=atoms, distribution=scaling_type, cutoff=cutoff)

            if scale_factor is not None:
                if min_scale_factor < scale_factor < max_scale_factor:
                    logger.debug("Cut off of {0} was successful".format(cutoff))
                    logger.info("Scale factor: {}".format(scale_factor))
                    logger.debug(
                        "Scale factor with extrinsic scaling: {} Angstrom".format(
                            scale_factor * extrinsic_scale_factor))
                    break
                else:
                    logger.info("Unable to obtain a physically meaningful scaling factor.")
                    logger.info("Scale factor: {} Angstrom".format(scale_factor))
                    logger.debug(
                        "Scale factor with extrinsic scaling: {} Angstrom".format(
                            scale_factor * extrinsic_scale_factor))
                    logger.info("Increasing cutoff from {} to {} Angstrom".format(
                        atoms_scaling_cutoffs[idx_cutoff], atoms_scaling_cutoffs[idx_cutoff + 1]))
            else:
                logger.info("Unable to obtain a scaling factor.")
                logger.info("Increasing cutoff from {} to {} Angstrom".format(
                    atoms_scaling_cutoffs[idx_cutoff], atoms_scaling_cutoffs[idx_cutoff + 1]))
    else:
        raise ValueError("Not recognized option for scaling_type. "
                         "Possible values are: 'min_nn', 'avg_nn', or 'quantile_nn'.")

    # deep copy otherwise the original atoms structure will be scaled
    # and the spacegroup_number_actual will be wrong
    atoms_tmp = copy.deepcopy(atoms)
    scale_factor = scale_factor * extrinsic_scale_factor
    atoms_tmp.set_positions(atoms_tmp.get_positions() * (1. / scale_factor))

    return atoms_tmp


def get_spacegroup_old(structure, materials_class=None):
    """Get spacegroup from a list of NOMAD structure.

    OBSOLETE. TO BE REMOVED."""

    energy_total = np.random.randn()

    if materials_class == 'binaries':
        # Get chemical formula with two elements and total energy per binary.
        chemical_formula = get_chemical_formula_binaries(structure)
        # energy_total = 2 * structure.energy_total / len(structure)
    else:
        chemical_formula = structure.chemical_formula
        # energy_total = structure.energy_total

    spacegroup_number = ase_get_spacegroup(structure, symprec=1e-03).no

    return chemical_formula, energy_total, spacegroup_number


def get_lattice_type(structure):
    """Get lattice_type from a list of NOMAD structure."""

    energy_total = {}
    chemical_formula = {}
    lattice_type = {}

    for (gIndexRun, gIndexDesc), atoms in iteritems(structure.atoms):
        if atoms is not None:
            energy_total[gIndexRun, gIndexDesc] = structure.energy_total[(gIndexRun, gIndexDesc)]
            chemical_formula[gIndexRun, gIndexDesc] = structure.chemical_formula[(gIndexRun, gIndexDesc)]

            lattice_type[gIndexRun, gIndexDesc] = structure.spacegroup_analyzer[
                gIndexRun, gIndexDesc].get_lattice_type()
            break

    return chemical_formula[0, 0], energy_total[0, 0], lattice_type[0, 0]


def get_target_diff_dic(df, sample_key=None, energy=None, spacegroup=None):
    """ Get a dictionary of dictionaries: samples -> space group tuples -> energy differences.

    Dropping all rows which do not correspond to the minimum energy per sample AND space group,
    then making a new data frame with space groups as columns. Finally constructing the dictionary
    of dictionaries.

    Parameters:

    df: pandas data frame
        with columns=[samples_title, energies_title, SG_title]

    sample_key: string
        Needs to be column title of samples of input df

    energy: string
        Needs to be column title of energies of input df

    spacegroup: string
        Needs to be column title of space groups of input df

    Returns:

    dic_out : dictionary of dictionaries
        In the form:
        {
        sample_a: { (SG_1,SG_2):E_diff_a12, (SG_1,SG_3):E_diff_a13,...},
        sample_b: { (SG_1,SG_2):E_diff_b12, (SG_1,SG_3):E_diff_b13,... },
        ...
        }
        E_diff_a12 = energy_SG_1 - energy_SG_2   of sample a.
        Both (SG_1,SG_2) and (SG_2,SG_1) are considered.
        If SG_1 or SG_2 is NaN, energy difference to it is ignored.

    .. codeauthor:: Emre Ahmetcik <ahmetcik@fhi-berlin.mpg.de>

    """

    # use only rows with minimum energies
    idx = df.groupby([sample_key, spacegroup])[energy].transform(min) == df[energy]
    df = df[idx]
    df = df.drop_duplicates()

    # make new table with the different supgroups as columns
    df = df.pivot_table(energy, [sample_key], spacegroup)

    # make dictionary of dictionaries
    SG_list = df.columns.values
    Samples_list = df.index.values
    matrix = np.array(df)
    dic_out = dict.fromkeys(Samples_list)
    for i, sample in enumerate(Samples_list):
        row = matrix[i]
        not_nan_indices = np.argwhere(~np.isnan(row)).flatten()
        sample_dic = {}
        for j_1, j_2 in permutations(not_nan_indices, 2):
            SG_1, SG_2 = SG_list[j_1], SG_list[j_2]
            Energy_diff = row[j_1] - row[j_2]
            sample_dic.update({(SG_1, SG_2): Energy_diff})
        if sample_dic:
            dic_out[sample] = sample_dic
    return dic_out


def select_diff_from_dic(dic, spacegroup_tuples, sample_key='Mat', drop_nan=None):
    """ Get data frame of selected spacegroup_tuples from dictionary of dictionaries.

        Creating a pandas data frame with columns of samples and selected space group tuples (energy differnces).

        Parameters:

        dic: dict {samples -> space group tuples -> energy differences.}

        spacegroup_tuples: tuple, list of tuples, tuples of tuples
            Each tuple has to contain two space groups numbers,
            to be looked up in the input dic.

        sample_key: string
            Will be the column title of the samples of the created data frame

        drop_nan: string, optional {'rows', 'SG_tuples'}
            Drops all rows or columns (SG_tuples) containing NaN.

    .. codeauthor:: Emre Ahmetcik <ahmetcik@fhi-berlin.mpg.de>

    """

    if isinstance(spacegroup_tuples, tuple) and all(isinstance(item, (float, int)) for item in spacegroup_tuples):
        spacegroup_tuples = [spacegroup_tuples]
    df_out = pd.DataFrame(dic, index=spacegroup_tuples).T

    if drop_nan is not None:
        if drop_nan == 'rows':
            df_out.dropna(axis=0, inplace=True)
        elif drop_nan == 'SG_tuples':
            df_out.dropna(axis=1, inplace=True)
        else:
            raise ValueError("Argument 'drop_nan' has to be 'None', 'rows' or 'SG_tuples'.")

    # check if df_out is empty
    len_columns = len(df_out.columns)
    len_rows = len(df_out.index)
    if len_columns == 0 or len_rows == 0:
        if len_rows == 0:
            string = 'rows'
        else:
            string = 'spacegroup_tuples'
        logger.error('Dropping {0} with NaNs leads to empty data frame.'.format(string))
        logger.error('Hint: Select different spacegroup_tuples or set drop_nan=None')
        sys.exit(1)
    df_out.reset_index(inplace=True)
    df_out.rename(columns={'index': sample_key}, inplace=True)
    return df_out


def convert_energy_substance(unit, value, energy_unit=None, length_unit=None):
    """Convert energy to energy/substance and viceversa.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    ureg_path = get_data_filename('utils/units.txt')
    ureg = UnitRegistry(ureg_path)
    unit_def = ureg(unit)
    energy_unit_def = ureg(energy_unit)

    if unit_def.dimensionality == '[length] ** 2 * [mass] / [time] ** 2':
        if energy_unit_def.dimensionality == '[length] ** 2 * [mass] / [substance] / [time] ** 2':
            # kj/mol and kcal/mol includes also [substance]
            # divide value by mol (multiply by Avogadro number)
            if isinstance(value, list):
                value = [x * constants.Avogadro for x in value]
            else:
                value = value * constants.Avogadro

            unit = unit + '/mol'
            value = uc.convert_unit(value, unit, target_unit=energy_unit)  # print feature, value, self.energy_unit
        else:
            # already an actual energy
            value = uc.convert_unit(value, unit, target_unit=energy_unit)  # print feature, value, self.energy_unit
    elif unit_def.dimensionality == '[length]':
        value = uc.convert_unit(value, unit, target_unit=length_unit)  # print feature, value, self.length_unit
    else:
        # keep original units (or no units, e.g. Zvalence)
        # raise ValueError("Unit needs to be [energy], [energy]/[substance] or [length]")
        pass

    return value


def format_e(n):
    """Transform a float in a string to reduce the number of decimal places shown."""
    try:
        a = '{0:.3E}'.format(Decimal(n))
    except TypeError as err:
        # if it cannot convert, keep the original format
        logger.debug(err)
        a = n
    return a


def filter_ase_list_by_label(ase_list, folder_name, cell_type='standard_no_symmetries', main_folder=None,
                             filter_by=None, accepted_labels=None, write_to_file=False,
                             symprec=None, all_symprec_consistent=True):
    """ Filter the ase_list for the descriptor according to the spgroup value.

    Example:
    filter_by=['lattice_type', 'spacegroup_symbol'],
    filter_by=['spacegroup_number'],

    filter_by=['lattice_type'],
    cell_type=cell_type,
    accepted_labels=accepted_label,
    accepted_labels=[['cubic'], ['Fd-3m', 'Fm-3m', 'Im-3m', 'Pm-3m']],
    ['I-43m', 'P2_13', 'Pa3']

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    folder_to_write = os.path.abspath(os.path.normpath(os.path.join(main_folder, folder_name)))

    if write_to_file:
        if not os.path.exists(folder_to_write):
            os.makedirs(folder_to_write)

    labels_filtered = {}
    filtered_ase_list = []

    # create empty list for each filter_by value to append values later and check which unique values are selected
    for filter_ in filter_by:
        labels_filtered.update({str(filter_): []})

    for idx, atoms in enumerate(ase_list):
        if idx % (int(len(ase_list) / 10) + 1) == 0:
            logger.info("Reading: file {0}/{1} "
                        "to filter the atomic structure list".format(idx + 1, len(ase_list)))

        atoms = standardize_cell(atoms, cell_type)

        atoms.info['spacegroup_nb'] = {}
        atoms.info['crystal_system'] = {}
        atoms.info['lattice_type'] = {}

        space_group_analyzer = get_spacegroup_analyzer(atoms, symprec=symprec)
        for key, value in space_group_analyzer.items():
            atoms.info['spacegroup_nb'][str(key)] = value.get_space_group_number()
            atoms.info['crystal_system'][str(key)] = value.get_crystal_system()
            atoms.info['lattice_type'][str(key)] = value.get_lattice_type()
            # labels.update({'json_file': json_file})

        if filter_by is not None:
            filter_count = 0

            for id_filter, filter_ in enumerate(filter_by):
                if all_symprec_consistent:
                    # consider only systems with a consistent classification
                    filter_values = atoms.info[str(filter_)].values()
                    if filter_values.count(filter_values[0]) == len(filter_values):
                        filter_value = str(filter_values[0])
                        # look if the property which we want to filter by
                        # is included in the atoms.info dictionary with the
                        # correct value
                        if str(filter_value) not in accepted_labels[id_filter]:
                            break
                        else:
                            labels_filtered[str(filter_)].append(filter_value)
                            filter_count += 1
                    else:
                        logger.debug("Values of {} are not unique w.r.t. the symmetry precision used.".format(filter_))
                        logger.debug("Values are {} for precision {}, respectively.".format(filter_values, symprec))

            # append to the filtered json only if everything is matching
            if filter_count == len(filter_by):
                filtered_ase_list.append(atoms)

    for filter_ in filter_by:
        logger.info("List of unique values for filter '{0}': "
                    "{1}".format(str(filter_), list(set(labels_filtered[str(filter_)]))))

    logger.info("List of 'accepted labels': "
                "{0}".format(accepted_labels))

    logger.info("Length of the whole ASE structure list: "
                "{0}".format(len(ase_list)))

    logger.info("Length of the filtered ASE structure list: "
                "{0}".format(len(filtered_ase_list)))

    if write_to_file:
        logger.info("Writing filtered ase list to file in folder: {0}".format(folder_to_write))

        for idx, atoms in enumerate(filtered_ase_list):
            ase_db_filename = os.path.abspath(os.path.normpath(os.path.join(folder_to_write, str(idx))))
            atoms.write(ase_db_filename + '.json', format='json')

    # write also a recap file
    #     results = {"filtered_json_list": filtered_json_list, "spacegroup_list": spacegroup_list,
    #                "unique_spacegroup_list": list(set(spacegroup_list)), "accepted_labels": accepted_labels,
    #                "values_filter": labels_filtered, "filter_by": filter_by, "symprec": symprec,
    #                "angle_tolerance": angle_tolerance}
    #
    #     with open(filtered_file, "w") as f:
    #         f.write("""
    # {
    #       "data":[""")
    #
    #         json.dump(results, f, indent=2)
    #
    #         f.write("""
    # ] }""")
    #         f.flush()

    return filtered_ase_list


def rot_mat_x(angle):
    """Return the rotation matrix for a rotation around the x axis"""
    return np.array([[1, 0, 0], [0, math.cos(np.radians(angle)), math.sin(np.radians(angle))],
                     [0, -math.sin(np.radians(angle)), math.cos(np.radians(angle))]]).astype(float)


def rot_mat_y(angle):
    """Return the rotation matrix for a rotation around the y axis"""
    return np.array([[math.cos(np.radians(angle)), 0, math.sin(np.radians(angle))], [0, 1, 0],
                     [-math.sin(np.radians(angle)), 0, math.cos(np.radians(angle))]]).astype(float)


def rot_mat_z(angle):
    """Return the rotation matrix for a rotation around the z axis"""
    return np.array([[math.cos(np.radians(angle)), math.sin(np.radians(angle)), 0],
                     [-math.sin(np.radians(angle)), math.cos(np.radians(angle)), 0], [0, 0, 1]]).astype(float)


def interpolate_parameters(initial_params, final_params, nb_steps=10, include_final=True):
    """Given a list of initial and final parameters, linearly interpolated between them."""
    step_params = (final_params - initial_params) / nb_steps

    if include_final:
        nb_param_sets = nb_steps + 1
    else:
        nb_param_sets = nb_steps

    params_list = []
    for idx in range(nb_param_sets):
        curr_params = initial_params + step_params * idx
        params_list.append(curr_params)
    return params_list


def get_boxes_from_xyz(filename, sliding_volume, stride_size, adapt=True, element_agnostic=False,
                       give_atom_density=False, plot_atom_density=False, padding_ratio=None):
    """Determine boxes for strided pattern analysis.
    
    Parameters:
    
    filename: string 
    Name of xyz input file specifying the polycrystal.
    
    sliding_volume: 1D list
    3 floats specifying sliding volume.
    
    stride_size: 1D list
    3 floats specifying stride size in x,y and z direction.
    
    adapt: bool,optional (default=True)
    If true, adapt minimum and possibly stride size such that distortions in heatmaps are reduced.
    
    element_agnostic: bool, optional (default=False)
    If true, consider all atoms to be of the same chemical species and return boxes containing only 'Fe' atoms.
    
    give_atom_density: bool, optional (default=False)
    If true, atom density is returned.
    
    plot_atom_density: bool, optional (default=False)
    If true, density of number of atoms is plotted as 2D heatmap for each z.
    
    Returns:
    
    Depending on the parameter give_atom_density, return will be 3D list of xyz boxes plus 3D list of atomic number
    density or only the 3D list of xyz boxes.

    .. codeauthor:: Andreas Leitherer <leitherer@fhi-berlin.mpg.de>

    """

    if padding_ratio is not None:
        padding = np.multiply(padding_ratio, sliding_volume)
    else:
        padding = np.array((0.0, 0.0, 0.0))

    # Get x,y,z coordinate vectors and coordinate range
    f = open(filename, 'r')
    x = []
    y = []
    z = []
    coordinate_vector_and_element_list = []  # 2D list with each element = [coordinate vector(1D list),'element name']
    lines = f.readlines()[2:]  # skip first two lines since they do not contain coordinates,
    # only number of total atoms and a comment line
    
    for line in lines:
        # Save all x,y and z coordinates in separate arrays
        # to determine min,max and range later on
        x.append(float(line.split()[1]))
        y.append(float(line.split()[2]))
        z.append(float(line.split()[3]))
        # Define list of coordinate vectors used for determining those atoms
        # within a certain sliding box
        coordinate_vector = [float(line.split()[1]), float(line.split()[2]), float(line.split()[3])]
        coordinate_vector_and_element_list.append([coordinate_vector, line.split()[0]])

    x_max = max(x) + padding[0]
    x_min = min(x) - padding[0]
    y_max = max(y) + padding[1]
    y_min = min(y) - padding[1]
    z_max = max(z) + padding[2]
    z_min = min(z) - padding[2]

    x_range = x_max-x_min
    y_range = y_max-y_min
    z_range = z_max-z_min

    logger.info("x range: {}".format(x_range))
    logger.info("y range: {}".format(y_range))
    logger.info("z range: {}".format(z_range))
    
    # Size of sliding window
    x_sliding_volume_edge_length = sliding_volume[0]
    y_sliding_volume_edge_length = sliding_volume[1]
    z_sliding_volume_edge_length = sliding_volume[2]
    
    # Step size for sliding window
    step_size_x = stride_size[0]
    step_size_y = stride_size[1]
    step_size_z = stride_size[2]

    # Check if step size exceeds coordinate range - if true, set range equal to step
    # size in that direction such that number of strides, which is computed
    # via range/step size, is equal to one and at least one step is made.
    # if step_size_x > x_range:
    #     logger.warning('x stride size exceeds x-coordinate range')
    #     x_range = step_size_x
    #     # this ensures int(x_range/step_size_x)=1 to be nonempty such that make one step in x direction
    # if step_size_y > y_range:
    #     logger.warning('y stride size exceeds y-coordinate range')
    #     y_range = step_size_y
    # if step_size_z > z_range:
    #     logger.warning('z stride size exceeds z-coordinate range')
    #     z_range = step_size_z

    # In the following, shift minimum value such that last sliding box does not overlap too
    # much into empty space and distortion in heatmaps is reduced.

    min_vector = np.array([x_min, y_min, z_min])
    max_vector = np.array([x_max, y_max, z_max])

    number_of_strides_vector = np.array(
        [int(math.floor(x_range/step_size_x))+1, int(math.floor(y_range/step_size_y))+1,
         int(math.floor(z_range/step_size_z))+1])

    stride_size_vector = np.array([step_size_x, step_size_y, step_size_z])
    sliding_volume_vector = np.array(
        [x_sliding_volume_edge_length, y_sliding_volume_edge_length, z_sliding_volume_edge_length])

    if adapt:
        # take number_of_strides_vector-1 because first step at minimum taken into account
        overhang = min_vector+np.multiply(number_of_strides_vector-1,
                                          stride_size_vector)+sliding_volume_vector-max_vector

        logger.debug("Minimum: {}".format(min_vector))
        logger.debug("Maximum: {}".format(max_vector))
        logger.debug("Number of strides: {}".format(number_of_strides_vector))
        logger.debug("Stride size: {}".format(stride_size_vector))
        logger.debug("Sliding volume: {}".format(sliding_volume_vector))
        logger.debug("Resulting overhang: {}".format(overhang))
            
        for i in range(0, 3):  # go through x,y,z coordinates of overhang vector
            
            # If overhang is negative, i.e. sliding box is so small that in the last step structure is potentially lost,
            # increase the number of strides automatically until overhang becomes positive (or zero)
            if overhang[i] < 0:
                while overhang[i] < 0:
                    logger.debug("In direction x_{}: overhang<0".format(str(i)))
                    number_of_strides_vector[i] += 1
                    # need to recompute overhang
                    overhang[i] = min_vector[i] + (number_of_strides_vector[i]-1)*stride_size_vector[i] + \
                        sliding_volume_vector[i] - max_vector[i]
                    logger.debug("In direction x_{}: residual overhang = {}".format(str(i), str(overhang[i])))
            
            if overhang[i] == 0.0:
                logger.debug("In direction x_{}: overhang=0".format(str(i)))
    
            # At this point the overhang is either zero or positive.
            # In the latter case, half of the overhang is subtracted from the minimum vector such that
            # boundary effects in the heatmaps are reduced
            if overhang[i] > 0:
                logger.debug("In direction x_{}: overhang>0".format(str(i)))
                min_vector[i] = min_vector[i]-(overhang[i]/2.0)

        logger.debug("New min: {}".format(min_vector))
        logger.debug("New number of strides: {}".format(number_of_strides_vector))

    x_min = min_vector[0]
    y_min = min_vector[1]
    z_min = min_vector[2]

    # Now determine boxes
    
    # start vector
    start = [x_min+x_sliding_volume_edge_length, y_min+y_sliding_volume_edge_length, z_min+z_sliding_volume_edge_length]
    logger.debug("Start vector: {}".format(start))

    list_of_xyz_boxes = []
    number_of_atoms_xyz = []

    # if z_range=step_size_z (see if branches above) then range(...)=range(1)=[0]!
    # So z=str(k*step_sze_z) gives correct z position, namely z=0 and sliding box is oriented to the positive side
    for k in range(number_of_strides_vector[2]):

        list_of_xy_boxes = []
        number_of_atoms_xy = []
        
        for i in range(number_of_strides_vector[1]):
            
            list_of_x_boxes = []
            number_of_atoms_x = []
            
            for j in range(number_of_strides_vector[0]):
                
                # Determine atoms within sliding box
                positionvectors_within_sliding_volume = []
                element_names_within_sliding_volume = ''

                for vector, element_name in coordinate_vector_and_element_list:
                    condition = vector[0] <= start[0] and vector[1] <= start[1] and vector[2] <= start[2] \
                                and vector[0] >= (start[0]-x_sliding_volume_edge_length) \
                                and vector[1] >= (start[1]-y_sliding_volume_edge_length) \
                                and vector[2] >= (start[2]-z_sliding_volume_edge_length)
                    
                    if condition:
                        positionvectors_within_sliding_volume.append(vector)
                        element_names_within_sliding_volume += element_name
                
                if len(positionvectors_within_sliding_volume) == 0:
                    number_of_atoms_x.append(0)
                    
                    element_name = element_names_within_sliding_volume  # should be ''
                    # create ase Atoms object Atoms(symbols='',pbc=False)
                    atoms_within_sliding_volume = ase.Atoms(element_name)
                    # Optional: assign label
                    # atoms_within_sliding_volume.info['label']='box_label_'+str(i)+str(j)+str(k)
                    list_of_x_boxes.append(atoms_within_sliding_volume)
                    
                else:
                    number_of_atoms_x.append(len(positionvectors_within_sliding_volume))
                    
                    if element_agnostic:
                        element_name = 'Fe'+str(len(positionvectors_within_sliding_volume))
                    else:
                        element_name = element_names_within_sliding_volume
                    atoms_within_sliding_volume = ase.Atoms(element_name, positionvectors_within_sliding_volume)
                    # Optional: assign label
                    # atoms_within_sliding_volume.info['label']='box_label_'+str(i)+str(j)+str(k)
                    list_of_x_boxes.append(atoms_within_sliding_volume)
                
                start[0] += step_size_x

            number_of_atoms_xy.append(number_of_atoms_x)
            list_of_xy_boxes.append(list_of_x_boxes)
            start[0] = x_min+x_sliding_volume_edge_length  # Reset x_value after most inner for loop finished
            start[1] += step_size_y  # next y
        
        number_of_atoms_xyz.append(number_of_atoms_xy)
        list_of_xyz_boxes.append(list_of_xy_boxes)
        start[1] = y_min+y_sliding_volume_edge_length  # Reset y_value for next z coordinate
        start[2] += step_size_z
        
    if give_atom_density:
        
        if plot_atom_density:
            
            z = 0
            for xy_density in number_of_atoms_xyz:
                plt.xlabel('x $[\mathrm{\AA}]$')
                plt.ylabel('y $[\mathrm{\AA}]$')
                plt.title('Atom density for z=' + str(z))
                plt.imshow(xy_density, interpolation='hanning', cmap='viridis',
                           extent=[0, (len(xy_density[0]))*stride_size[0], (len(xy_density))*stride_size[1], 0])
                plt.colorbar()
                plt.show()
                plt.savefig(filename[:-4] + '_Atom_density_for_z=' + str(z) + '.png')
                plt.close()
                
                z += stride_size[2]

        return list_of_xyz_boxes, number_of_atoms_xyz
    else:
        return list_of_xyz_boxes


def rename_material(chem_f):
    """Patch to rename certain binaries with the more electronegative element first."""
    for idx, el in enumerate(chem_f):
        if el == 'FLi':
            chem_f[idx] = 'LiF'
        if el == 'ClLi':
            chem_f[idx] = 'LiCl'
        if el == 'BrLi':
            chem_f[idx] = 'LiBr'
        if el == 'ILi':
            chem_f[idx] = 'LiI'
        if el == 'AsB':
            chem_f[idx] = 'BAs'
        if el == 'FNa':
            chem_f[idx] = 'NaF'
        if el == 'ClNa':
            chem_f[idx] = 'NaCl'
        if el == 'BrNa':
            chem_f[idx] = 'NaBr'
        if el == 'INa':
            chem_f[idx] = 'NaI'
        if el == 'CSi':
            chem_f[idx] = 'SiC'
        if el == 'FK':
            chem_f[idx] = 'KF'
        if el == 'ClK':
            chem_f[idx] = 'KCl'
        if el == 'BrK':
            chem_f[idx] = 'KBr'
        if el == 'IK':
            chem_f[idx] = 'KI'
        if el == 'ClCu':
            chem_f[idx] = 'CuCl'
        if el == 'BrCu':
            chem_f[idx] = 'CuBr'
        if el == 'OZn':
            chem_f[idx] = 'ZnO'
        if el == 'SZn':
            chem_f[idx] = 'ZnS'
        if el == 'SeZn':
            chem_f[idx] = 'ZnSe'
        if el == 'TeZn':
            chem_f[idx] = 'ZnTe'
        if el == 'AsGe':
            chem_f[idx] = 'GeAs'
        if el == 'FRb':
            chem_f[idx] = 'RbF'
        if el == 'ClRb':
            chem_f[idx] = 'RbCl'
        if el == 'BrRb':
            chem_f[idx] = 'RbBr'
        if el == 'IRb':
            chem_f[idx] = 'RbI'
        if el == 'OSr':
            chem_f[idx] = 'SrO'
        if el == 'SSr':
            chem_f[idx] = 'SrS'
        if el == 'SeSr':
            chem_f[idx] = 'SrSe'
        if el == 'AsIn':
            chem_f[idx] = 'InAs'
        if el == 'ClCs':
            chem_f[idx] = 'CsCl'
        if el == 'BrCs':
            chem_f[idx] = 'CsBr'
        if el == 'SiSn':
            chem_f[idx] = 'SnSi'
        if el == 'CGe':
            chem_f[idx] = 'GeC'
        if el == 'GeSn':
            chem_f[idx] = 'SnGe'
        if el == 'CSn':
            chem_f[idx] = 'SnC'

    return chem_f
