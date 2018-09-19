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

from ase.data import chemical_symbols
from future.utils import viewitems
import logging
import math
from ai4materials.descriptors.base_descriptor import Descriptor
from ai4materials.descriptors.base_descriptor import is_descriptor_consistent
import numpy as np
import os
import scipy.sparse
logger = logging.getLogger('ai4materials')


class PRDF(Descriptor):
    """Compute the partial radial distribution of a given crystal structure.

    Cell vectors v1,v2,v3 with values in the columns: [[v1x,v2x,v3x],[v1y,v2y,v3x],[v1z,v2z,v3z]]

    Parameters:

    cutoff_radius: float, optional (default=20)
        Atoms within a sphere of cut-off radius (in Angstrom) are considered.

    rdf_only: bool, optional (defaults=`False`)
        If `False` calculates partial radial distribution function.
        If `True` calculates radial distribution function (all atom types are considered as the same)

    .. codeauthor:: Fawzi Mohamed <mohamed@fhi-berlin.mpg.de> and Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    def __init__(self, configs=None, cutoff_radius=20, rdf_only=False):
        super(PRDF, self).__init__(configs=configs)

        self.cutoff_radius = cutoff_radius
        self.rdf_only = rdf_only

        if not self.rdf_only:
            logger.info("Calculating partial radial distribution function.")
        else:
            logger.info("Calculating radial distribution function; all atomic species are considered as the same.")

    def calculate(self, structure, **kwargs):
        """Calculate the descriptor for the given ASE structure.

        Parameters:

        structure: `ase.Atoms` object
            Atomic structure.

        .. codeauthor:: Fawzi Mohamed <mohamed@fhi-berlin.mpg.de>

        """

        if not np.all(structure.pbc):
            raise NotImplementedError("At present, the partial radial distribution function is implemented only for "
                                      "three-dimensional periodic structures.")
        else:
            cell = structure.get_cell()
            cutoff_radius2 = self.cutoff_radius * self.cutoff_radius

            radD = {}
            for ii, a1 in enumerate(structure):
                for jj, a2 in enumerate(structure[ii:]):

                    # write atomic numbers if prdf, otherwise set them to zero
                    if not self.rdf_only:
                        n1 = a1.number
                        n2 = a2.number
                    else:
                        n1 = 0
                        n2 = 0

                    label = "%d_%d" % (min(n1, n2), max(n1, n2))
                    if label not in radD:
                        radD[label] = {"particle_atom_number_1": n1, "particle_atom_number_2": n2, "arr": []}
                    arr = radD[label]["arr"]
                    r0 = np.array([a1.x - a2.x, a1.y - a2.y, a1.z - a2.z])
                    r02 = np.dot(r0, r0)
                    m = np.dot(cell.transpose(), cell)
                    l = np.dot(r0, cell)
                    r = np.dot(r0, r0) - cutoff_radius2
                    mii = m[0, 0]
                    mij = m[0, 1]
                    mik = m[0, 2]
                    mjj = m[1, 1]
                    mjk = m[1, 2]
                    mkk = m[2, 2]
                    li = l[0]
                    lj = l[1]
                    lk = l[2]

                    c = (
                        mjj ** 2 * mkk ** 3 * r - 2 * mjj * mjk ** 2 * mkk ** 2 * r + mjk ** 4 * mkk * r - lj ** 2 * mjj * mkk ** 3 + lj ** 2 * mjk ** 2 * mkk ** 2 + 2 * lj * lk * mjj * mjk * mkk ** 2 - lk ** 2 * mjj ** 2 * mkk ** 2 - 2 * lj * lk * mjk ** 3 * mkk + lk ** 2 * mjj * mjk ** 2 * mkk)

                    a = (
                        mii * mjj ** 2 * mkk ** 3 - mij ** 2 * mjj * mkk ** 3 - 2 * mii * mjj * mjk ** 2 * mkk ** 2 + mij ** 2 * mjk ** 2 * mkk ** 2 + 2 * mij * mik * mjj * mjk * mkk ** 2 - mik ** 2 * mjj ** 2 * mkk ** 2 + mii * mjk ** 4 * mkk - 2 * mij * mik * mjk ** 3 * mkk + mik ** 2 * mjj * mjk ** 2 * mkk)

                    b = (
                        2 * li * mjj ** 2 * mkk ** 3 - 2 * lj * mij * mjj * mkk ** 3 - 4 * li * mjj * mjk ** 2 * mkk ** 2 + 2 * lj * mij * mjk ** 2 * mkk ** 2 + 2 * lj * mik * mjj * mjk * mkk ** 2 + 2 * lk * mij * mjj * mjk * mkk ** 2 - 2 * lk * mik * mjj ** 2 * mkk ** 2 + 2 * li * mjk ** 4 * mkk - 2 * lj * mik * mjk ** 3 * mkk - 2 * lk * mij * mjk ** 3 * mkk + 2 * lk * mik * mjj * mjk ** 2 * mkk)

                    delta = b * b - 4 * a * c

                    if a == 0 or delta < 0:
                        continue
                    sDelta = math.sqrt(delta)
                    imin = int(math.ceil((-b - sDelta) / (2 * a)))
                    imax = int(math.floor((-b + sDelta) / (2 * a)))
                    for i in range(imin, imax + 1):
                        cj = (
                            mkk * r + i ** 2 * mii * mkk + 2 * i * li * mkk - i ** 2 * mik ** 2 - 2 * i * lk * mik - lk ** 2)
                        aj = (mjj * mkk - mjk ** 2)
                        bj = (2 * i * mij * mkk + 2 * lj * mkk - 2 * i * mik * mjk - 2 * lk * mjk)
                        deltaj = bj * bj - 4 * aj * cj
                        if aj == 0 or deltaj < 0:
                            continue
                        sDeltaj = math.sqrt(deltaj)
                        jmin = int(math.ceil((-bj - sDeltaj) / (2 * aj)))
                        jmax = int(math.floor((-bj + sDeltaj) / (2 * aj)))
                        for j in range(jmin, jmax + 1):
                            ck = r + j ** 2 * mjj + 2 * i * j * mij + i ** 2 * mii + 2 * j * lj + 2 * i * li
                            ak = mkk
                            bk = (2 * j * mjk + 2 * i * mik + 2 * lk)
                            deltak = bk * bk - 4 * ak * ck
                            if ak == 0 or deltak < 0:
                                continue
                            sDeltak = math.sqrt(deltak)
                            kmin = int(math.ceil((-bk - sDeltak) / (2 * ak)))
                            kmax = int(math.floor((-bk + sDeltak) / (2 * ak)))
                            for k in range(kmin, kmax + 1):
                                if jj != 0 or i != 0 or j != 0 or k != 0:
                                    rr = r02 + k ** 2 * mkk + k * (
                                        2 * j * mjk + 2 * i * mik + 2 * lk) + j ** 2 * mjj + 2 * i * j * mij + i ** 2 * mii + 2 * j * lj + 2 * i * li
                                    arr.append(math.sqrt(rr))
            wFact = 4 * math.pi * len(structure) / abs(np.linalg.det(cell))
            for k, v in radD.items():
                v["arr"].sort()
                v["weights"] = map(lambda r: 1.0 / (wFact * r * r), v["arr"])

        # add results in ASE structure info
        if self.rdf_only:
            descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self), rdf=radD)
        else:
            descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self), prdf=radD)

        structure.info['descriptor'] = descriptor_data

        return structure

    def write(self, structure, tar, op_id=0, write_geo=True, format_geometry='aims'):
        """Write the descriptor to file.

        Parameters:

        structure: `ase.Atoms` object
            Atomic structure.

        tar: TarFile object
            TarFile archive where the descriptor is added. This is created internally with `tarfile.open`.

        op_id: int, optional (default=0)
            Number of the applied operation to the descriptor. At present always set to zero in the code.

        write_geo: bool, optional (default=`True`)
            If `True`, write a coordinate file of the structure for which the diffraction pattern is calculated.


        .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

        """

        if not is_descriptor_consistent(structure, self):
            raise Exception('Descriptor not consistent. Aborting.')

        desc_folder = self.configs['io']['desc_folder']

        if self.rdf_only:
            rdf = structure.info['descriptor']['rdf']

            rdf_filename_npy = os.path.abspath(os.path.normpath(
                os.path.join(desc_folder, structure.info['label'] + self.desc_metadata.ix['rdf']['file_ending'])))
            np.save(rdf_filename_npy, rdf)
            structure.info['rdf_filename_npy'] = rdf_filename_npy
            tar.add(structure.info['rdf_filename_npy'])

        else:
            prdf = structure.info['descriptor']['prdf']

            prdf_filename_npy = os.path.abspath(os.path.normpath(
                os.path.join(desc_folder, structure.info['label'] + self.desc_metadata.ix['prdf']['file_ending'])))
            np.save(prdf_filename_npy, prdf)
            structure.info['prdf_filename_npy'] = prdf_filename_npy
            tar.add(structure.info['prdf_filename_npy'])

        if write_geo:
            # to have the file accessible by the Beaker notebook image we need to put them
            # in a special folder ('/user/tmp')
            if self.configs['runtime']['isBeaker']:
                # only for Beaker Notebook
                coord_filename_in = os.path.abspath(os.path.normpath(os.path.join('/user/tmp/',
                                                                                  structure.info['label'] +
                                                                                  self.desc_metadata.ix[
                                                                                      'prdf_coordinates'][
                                                                                      'file_ending'])))
            else:
                coord_filename_in = os.path.abspath(os.path.normpath(os.path.join(desc_folder, structure.info['label'] +
                                                                                  self.desc_metadata.ix[
                                                                                      'prdf_coordinates'][
                                                                                      'file_ending'])))

            structure.write(coord_filename_in, format=format_geometry)
            structure.info['prdf_coord_filename_in'] = coord_filename_in
            tar.add(structure.info['prdf_coord_filename_in'])


def get_unique_chemical_species(structures):
    """Get the set of unique chemical species from a list of atomic structures.

    The list of structures must contain the calculated :py:class:`ai4materials.descriptors.prdf.PRDF`.

    Parameters:

    structures: ``ase.Atoms`` object or list of ``ase.Atoms`` objects
        Atomic structure or list of atomic structure.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    atomic_nbs_set = set()
    for structure in structures:
        prdfs = structure.info['descriptor']['prdf']

        # find the set of unique number of chemical species in the partial radial distribution functions
        for (key, value) in viewitems(prdfs):
            atom_type_1 = value['particle_atom_number_1']
            atom_type_2 = value['particle_atom_number_2']
            atomic_nbs_set.add(atom_type_1)
            atomic_nbs_set.add(atom_type_2)

        unique_chem_species = [chemical_symbols[item] for item in atomic_nbs_set]

        max_rdf_length = 0
        for name, rdf in viewitems(prdfs):
            rdf_length = len(rdf['weights'])
            if max_rdf_length < rdf_length:
                max_rdf_length = rdf_length

            assert len(rdf['arr']) == len(rdf['weights']), "Wrong number of distances at c={}.".format(name)

    largest_atomic_nb = max(atomic_nbs_set) + 1

    logger.debug("Setting up dictionary of chemical species.")
    logger.debug("Longest rdf list: {0}".format(max_rdf_length))
    logger.debug("Number of different chemical species in the set: {0}".format(len(atomic_nbs_set)))
    logger.debug("Highest atomic number in the set: {0}".format(largest_atomic_nb))
    logger.info("Actual chemical species set: {0}".format(unique_chem_species))

    return largest_atomic_nb, unique_chem_species


def get_design_matrix(structures, total_bins=50, max_dist=25):
    """Starting from atomic structures calculate the design matrix for the partial radial distribution function.

    The list of structures must contain the calculated :py:class:`ai4materials.descriptors.prdf.PRDF`.
    The discretization is performed using a logarithmic grid as follows:
        bins = np.logspace(0, np.log10(max_dist), num=total_bins + 1) - 1

    Parameters:

    structures: ``ase.Atoms`` object or list of ``ase.Atoms`` object
        Atomic structure or list of atomic structure.

    total_bins: int, optional (default=50)
        Total number of bins to be used in the discretization of the partial radial distribution function.

    max_dist: float, optional (default=25)
        Maximum distance to consider in the partial radial distribution function when the design matrix is
        calculated. Unit in Angstrom.
        The unit of measure is the same as :py:class:`ai4materials.descriptors.prdf.PRDF`.

    Return:

    scipy.sparse.csr.csr_matrix, shape [n_samples, largest_atomic_nb * largest_atomic_nb * total_bins]
        Returns a sparse row-compressed matrix.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    largest_atomic_nb, _ = get_unique_chemical_species(structures)

    design_matrix = scipy.sparse.lil_matrix((len(structures), largest_atomic_nb * largest_atomic_nb * total_bins))
    logger.debug("Feature matrix shape: {0}".format(design_matrix.shape))

    bins = np.logspace(0, np.log10(max_dist), num=total_bins + 1) - 1

    for idx_structure, structure in enumerate(structures):
        prdfs = structure.info['descriptor']['prdf']

        def chem_species_sorted(partial_rdfs):
            atom_nbs = [x.split("_") for x in list(partial_rdfs)]
            sorted_atom_nbs = sorted([(int(a), int(b)) for a, b in atom_nbs])
            return ["{0}_{1}".format(a, b) for a, b in sorted_atom_nbs]

        # indices must increase monotonically, other lil_matrix has bad performance
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.lil_matrix.html
        for name in chem_species_sorted(prdfs):
            prdf = prdfs[name]
            i_atom = prdf['particle_atom_number_1']
            j_atom = prdf['particle_atom_number_2']

            if i_atom < j_atom:
                i_atom, j_atom = j_atom, i_atom

            arr = np.array(prdf['arr'], dtype=np.float64).reshape((-1,))
            weights = np.array(prdf['weights'], dtype=np.float64).reshape((-1,))

            binned, foo = np.histogram(arr, bins=bins, weights=weights, density=True)

            idx = (i_atom * largest_atomic_nb * total_bins +
                   j_atom * total_bins)

            for i in range(total_bins):
                design_matrix[idx_structure, idx] = binned[i]
                idx += 1

    design_matrix = design_matrix.asformat("csr")
    logger.debug("Sparse feature matrix needs {0} bytes".format(design_matrix.data.nbytes))
    logger.debug("Feature matrix shape: {0}".format(design_matrix.shape))

    return design_matrix
