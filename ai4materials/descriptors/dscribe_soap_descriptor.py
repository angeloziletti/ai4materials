import os
import os.path

from ai4materials.descriptors.base_descriptor import Descriptor
from ai4materials.descriptors.base_descriptor import is_descriptor_consistent
from ai4materials.utils.utils_crystals import scale_structure

from dscribe.descriptors import SOAP
from dscribe.utils.geometry import get_extended_system

import numpy as np

import matplotlib.pyplot as plt

from collections import Counter

import itertools

from copy import deepcopy

import logging
logger = logging.getLogger('ai4materials')


class dscribe_SOAP_descriptor(Descriptor):
    """
    SOAP descriptor from dscribe package
    """

    def __init__(self, configs=None, p_b_c=False, cutoff=4.0,
                 l_max=6, n_max=9, atom_sigma=0.1, central_weight=0.0,
                 average=True, average_over_permuations=False,
                 number_averages=200, atoms_scaling='quantile_nn',
                 atoms_scaling_cutoffs=[10.], extrinsic_scale_factor=1.0,
                 n_Z=1, Z=26, n_species=1, species_Z=26,
                 scale_element_sensitive=False, return_binary_descriptor=False,
                 average_binary_descriptor=False, min_atoms=1, shape_soap=316,
                 constrain_nn_distances=False, rbf="polynomial",
                 use_mixed_coefficients_implementation=False):
        super(dscribe_SOAP_descriptor, self).__init__(configs=configs)

        self.p_b_c = p_b_c
        self.cutoff = cutoff
        self.l_max = l_max
        self.n_max = n_max
        self.atom_sigma = atom_sigma
        self.central_weight = central_weight
        self.average = average

        self.average_over_permuations = average_over_permuations
        self.number_averages = number_averages

        self.atoms_scaling = atoms_scaling
        self.atoms_scaling_cutoffs = atoms_scaling_cutoffs
        self.extrinsic_scale_factor = extrinsic_scale_factor

        # From quippy documentation:
        # https://libatoms.github.io/QUIP/Tutorials/quippy-descriptor-tutorial.html#A-many-body-descriptor:-SOAP
        # Atomic numbers to be considered for central atom, e.g. Z={1 6}
        self.Z = Z
        # How many different types of central atoms to consider
        self.n_Z = n_Z
        # Number of species for the descriptor
        self.n_species = n_species
        # Atomic number of species, e.g. species_Z={1 6}
        self.species_Z = species_Z
        self.scale_element_sensitive = scale_element_sensitive

        # If return_binary_descriptor=True, then
        # return descriptor (11,12,21,22),
        # with ij being the (averaged) SOAP descriptor
        # where sit on atoms of
        # species i and only consider atoms with species j as neighbors.
        self.return_binary_descriptor = return_binary_descriptor
        # if average_binary_descriptor=True, average over soap
        # vectors from different species/ the corresponding chem. envs.
        # Default = False, i.e.,
        # [(1,1)-SOAP vector, (1,2) SOAP_vector), ... ] is returned
        self.average_binary_descriptor = average_binary_descriptor

        # minimum number of atoms (important for
        # polycrystal application, otherwise will get lots of errors)
        self.min_atoms = min_atoms
        # important if nber aotms < min_atoms because then need shape!
        self.shape_soap = shape_soap

        # for some structures, nn may be
        # smaller than default thershold in utils_crystals'
        # get_nndistance function
        self.constrain_nn_distances = constrain_nn_distances

        #descriptor_options = 'soap '+'cutoff='+str(self.cutoff)+' l_max='+str(self.l_max)+' n_max='+str(self.n_max)+' atom_sigma='+str(self.atom_sigma)+\
        #                     ' n_Z='+str(self.n_Z)+' Z={'+str(self.Z)+'} n_species='+str(self.n_species)+' species_Z={'+str(self.species_Z)+'} central_weight='+str(self.central_weight)+' average='+str(self.average)              
        #self.descriptor_options=descriptor_options
        self.rbf = rbf
        
        self.use_mixed_coefficients_implementation = use_mixed_coefficients_implementation

    def calculate(self, structure, **kwargs):

        # get all atomic numbers and the unique values
        atomic_numbers = structure.get_atomic_numbers()
        atomic_numbers_unique = np.unique(atomic_numbers)

        occurences_species = Counter(atomic_numbers)
        species_to_delete = []
        for species in atomic_numbers_unique:
            if occurences_species[species] < self.min_atoms:
                species_to_delete.append(species)
        del structure[[atom.index for atom in structure if atom.number in species_to_delete]]

        # case that had to delete all species, return nan
        if len(structure) == 0:
            soap_desc = np.full(self.shape_soap, np.nan)
            descriptor_data = dict(descriptor_name=self.name,
                                   descriptor_info=str(self),
                                   SOAP_descriptor=soap_desc)
            structure.info['descriptor'] = descriptor_data
            return structure

        # now actual SOAP calculation

        """
        substructure_dict = {}
        for z in atomic_numbers:
            mask = atomic_numbers == z
            substructure = structure[mask]
            substructure_dict[z] = {'structure': substructure,
                                    'positions': substructure.positions}
        """
        pbc_structure = structure.get_pbc()
        print(pbc_structure)
        #if # not working: (not pbc_structure.all()==True) or (not pbc_structure.all()==False):
        if len(list(set(pbc_structure)))>1:
            raise NotImplementedError("Mixed periodic boundary conditions not supported at the moment")
        elif pbc_structure.all()==True:
            self.p_b_c = True
        elif pbc_structure.all()==False:
            self.p_b_c = False
        
        
        if self.use_mixed_coefficients_implementation:
            print("Using mixed coefficients.")
            soap = SOAP(species=atomic_numbers_unique,
                        periodic=self.p_b_c,
                        rcut=self.cutoff, nmax=self.n_max, lmax=self.l_max,
                        sigma=self.atom_sigma, average=self.average,
                        rbf=self.rbf, sparse=False) #, crossover=False)
                        
            atoms = scale_structure(structure,
                                    scaling_type=self.atoms_scaling,
                                    atoms_scaling_cutoffs=self.atoms_scaling_cutoffs,
                                    extrinsic_scale_factor=self.extrinsic_scale_factor,
                                    element_sensitive=False, # !!
                                    central_atom_species=1,
                                    neighbor_atoms_species=1,
                                    constrain_nn_distances=self.constrain_nn_distances)
            soap_desc = soap.create(atoms, positions=atoms.positions)
            norm = np.linalg.norm(soap_desc.flatten())
            soap_desc = soap_desc.flatten() / norm
            all_descriptors = []
            def split_list_(sequence, nb_splits):
                """ Split l in n_split. It can return unevenly sized chunks.
            
                Parameters:
            
                sequence: iterable
                    Iterable object to be split in `nb_splits`.
            
                nb_splits: int
                    Number of splits.
            
                Returns:
            
                iterable
                    `sequence` splits in `nb_splits`.
            
                .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>
            
                """
            
                return [sequence[i::nb_splits] for i in range(nb_splits)]
            all_descriptors = split_list_(soap_desc, len(list(itertools.combinations_with_replacement(atomic_numbers_unique, 2))))
            """            
            for Z_1 in atomic_numbers_unique:
                for Z_2 in atomic_numbers_unique:
                    # only with py3 version:                    
                    #location = soap.get_location((Z_1, Z_2))
                    #reduced_soap_desc = soap_desc[location.start:location.stop]
                    #all_descriptors.append(reduced_soap_desc)
            """
            if self.average_binary_descriptor:
                all_descriptors = np.mean(np.array(all_descriptors), axis=0)
                descriptor_data = dict(descriptor_name=self.name,
                                       descriptor_info=str(self),
                                       SOAP_descriptor=all_descriptors)
            else:
                descriptor_data = dict(descriptor_name=self.name,
                                       descriptor_info=str(self),
                                       SOAP_descriptor=np.array(all_descriptors))
            structure.info['descriptor'] = descriptor_data
            return structure
        else:
            print("Not using mixed coefficients.")
            # not working for dscribe:
            #self.p_b_c = structure.get_pbc()
            # Need either True or False
            #rbf = "polynomial"
            all_descriptors = []
            for Z_centers in atomic_numbers_unique:
                for Z_neighbors in atomic_numbers_unique:
                    soap = SOAP(species=[Z_neighbors],
                                # below, give substructure which
                                # has only Z_neighbors atoms,
                                # but sit at positions corresponding to Z_centers
                                # only problem left: for multiple species,
                                # no atoms should be at the centers,
                                # but for mono-species compounds, don't
                                # have a central weight parameter to
                                # switch off contributions from the central atom.
                                periodic=self.p_b_c,
                                rcut=self.cutoff, nmax=self.n_max, lmax=self.l_max,
                                sigma=self.atom_sigma, average=self.average,
                                rbf=self.rbf, sparse=False) #, crossover=False)
                    print(Z_centers, Z_neighbors)
    
                    # atoms = substructure_dict[Z_neighbors]['structure']
    
                    atoms = scale_structure(structure,
                                            scaling_type=self.atoms_scaling,
                                            atoms_scaling_cutoffs=self.atoms_scaling_cutoffs,
                                            extrinsic_scale_factor=self.extrinsic_scale_factor,
                                            element_sensitive=self.scale_element_sensitive,
                                            central_atom_species=Z_centers,
                                            neighbor_atoms_species=Z_neighbors,
                                            constrain_nn_distances=self.constrain_nn_distances)
    
                    # Filter out substructure and corresponding atomic positions
                    mask_neighbors = atomic_numbers == Z_neighbors
                    mask_centers = atomic_numbers == Z_centers
    
                    positions_centers = atoms[mask_centers].positions
                    substructure = atoms[mask_neighbors]
                    print(substructure)
                    #from ase.visualize import view
                    #view(substructure)
                    
                    if Z_centers==Z_neighbors: #- this segment won't work - if pbc=True, then remove atom and appears spuriously in replicas!!! For pbc=False, it may work
                    # also if substitue species, won't work, still gives spurious replicas!
                        soap_descs = []
                        #if self.p_b_c:
                        #    substructure = get_extended_system(substructure, self.cutoff)
                            #replica = 1
                            #while len(substructure*(replica, replica, replica))<=1:
                            #    replica += 1
                            #substructure *= (replica, replica, replica)
                        #elif len(substructure)<=1:
                        #    raise NotImplementedError("pbc False for only one atom -> not implemented at the moment.")
                        soap = SOAP(species=[Z_neighbors],
                                    # below, give substructure which
                                    # has only Z_neighbors atoms,
                                    # but sit at positions corresponding to Z_centers
                                    # only problem left: for multiple species,
                                    # no atoms should be at the centers,
                                    # but for mono-species compounds, don't
                                    # have a central weight parameter to
                                    # switch off contributions from the central atom.
                                    periodic=False, # get extended system, treat as non-periodic!
                                    rcut=self.cutoff, nmax=self.n_max, lmax=self.l_max,
                                    sigma=self.atom_sigma, average=self.average,
                                    rbf=self.rbf, sparse=False)
                        for idx, atom in enumerate(substructure):
                            central_pos = atom.position
                            atoms_without_center = substructure.copy()

                            if self.p_b_c:
                                atoms_without_center = get_extended_system(atoms_without_center, self.cutoff, centers=[central_pos])[0]
                            
                            del atoms_without_center[idx]
                            soap_descs.append(soap.create(atoms_without_center, positions=[central_pos]).flatten())
                            # substitute species:
                            """
                            tmp_substructure = deepcopy(substructure)
                            new_species = 1
                            while new_species in atomic_numbers_unique:
                                new_species += 1
                            #if len(substructure)>1:
                            #    del tmp_substructure[idx]
                            species_substructure = tmp_substructure.get_atomic_numbers()
                            species_substructure[idx] = new_species
                            tmp_substructure.set_atomic_numbers(species_substructure)
                            print(tmp_substructure)
                            soap_desc = soap.create(tmp_substructure, positions=[atom.position])
                            soap_descs.append(soap_desc.flatten()) # WILL make problems if self.average=False
                            #print(soap_desc.shape)
                            """
                        # this won't work when taking the norm... 
                        #if self.average:
                        #    #print(np.array(soap_descs).shape)
                        #    soap_descs = np.mean(soap_descs, axis=0)
                        #    #print(soap_descs.shape)
                        soap_descs = np.mean(soap_descs, axis=0)
                        norm = np.linalg.norm(soap_descs.flatten())
                        soap_descs = soap_descs.flatten() / norm
                        all_descriptors.append(soap_descs)
                        
                    else:
                        soap_desc = soap.create(substructure,
                                                positions=positions_centers)
                        #print(soap_desc)
                        # print(soap_desc.shape)
                        norm = np.linalg.norm(soap_desc.flatten())
                        soap_desc = soap_desc.flatten() / norm
                        all_descriptors.append(soap_desc)
                    # print(Z_centers, Z_neighbors)
                    # plt.plot(soap_desc.flatten())
                    # plt.show()
    
            if self.average_binary_descriptor:
                all_descriptors = np.mean(np.array(all_descriptors), axis=0)
                descriptor_data = dict(descriptor_name=self.name,
                                       descriptor_info=str(self),
                                       SOAP_descriptor=all_descriptors)
            else:
                descriptor_data = dict(descriptor_name=self.name,
                                       descriptor_info=str(self),
                                       SOAP_descriptor=np.array(all_descriptors))
            structure.info['descriptor'] = descriptor_data
            return structure
                             
    def write(self,structure,tar,write_soap_npy=True,write_soap_png=True,op_id=0,write_geo=True,format_geometry='aims'):
        """Write the descriptor to file.

        Parameters:

        structure: `ase.Atoms` obejct
            Atomic structure.

        tar: TarFile object
            TarFile archive where the descriptor is added. This is created internally with `tarfile.open`. 
            
        write_soap_npy: bool,optional (default=True)
            If true, write SOAP descriptor to binary file 
            
        write_soap_png: bool,optional (default=True)
            If True, write to file a png file showing the SOAP descriptor
            
        op_id: int, optional (default=0)
            To be added        
        
        write_geo: bool, optional( default='True')
            If 'true', write a coordinate file of the structure for which the SOAP descriptor is calculated.
            
        format_geometry: string, optional (default=`aims`)
            Output format of the geometry file. All ASE valid output formats are accepted.
            For a complete list see: https://wiki.fysik.dtu.dk/ase/ase/io/io.html        
        """
        
        if not is_descriptor_consistent(structure, self):
            raise Exception('Descriptor not consistent. Aborting.')        
        
        desc_folder = self.configs['io']['desc_folder']
        descriptor_info = structure.info['descriptor']['descriptor_info']
        
        soap_descriptor=structure.info['descriptor']['SOAP_descriptor']
        
        #tar=tarfile.open(desc_folder+tar,'w')
        
        if write_soap_npy:
            
            soap_filename_npy = os.path.abspath(os.path.normpath(os.path.join(desc_folder,
                                                                                   structure.info['label'] +
                                                                                   self.desc_metadata.ix[
                                                                                       'dscribe_SOAP'][
                                                                                       'file_ending'])))
            only_file=structure.info['label'] + self.desc_metadata.ix['dscribe_SOAP']['file_ending']
            
            np.save(soap_filename_npy, soap_descriptor)
            structure.info['dscribe_SOAP_descriptor_filename_npy'] = soap_filename_npy
            tar.add(structure.info['dscribe_SOAP_descriptor_filename_npy'],arcname=only_file) 
            
        if write_soap_png:

            image_soap_filename_png = os.path.abspath(os.path.normpath(os.path.join(desc_folder,
                                                                                   structure.info['label'] +
                                                                                   self.desc_metadata.ix[
                                                                                       'dscribe_SOAP_image'][
                                                                                       'file_ending'])))
            only_file=structure.info['label'] + self.desc_metadata.ix['dscribe_SOAP_image']['file_ending']
            
            plt.ioff()
            plt.title(structure.info['label']+' SOAP descriptor ')
            plt.xlabel('SOAP component')
            plt.ylabel('SOAP value')
            plt.plot(soap_descriptor)
            # adjust y axis limit
            #ax = plt.gca()
            #ax.set_ylim([-0.5,0.7])
            plt.savefig(image_soap_filename_png)
            plt.close()
            structure.info['dscribe_SOAP_descriptor_filename_png'] = image_soap_filename_png
            tar.add(structure.info['dscribe_SOAP_descriptor_filename_png'],arcname=only_file)             
        
        if write_geo:
            
            coord_filename_in = os.path.abspath(os.path.normpath(os.path.join(desc_folder, structure.info['label'] +
                                                                                  self.desc_metadata.ix['dscribe_SOAP_coordinates'][
                                                                                      'file_ending'])))
            
            only_file=structure.info['label'] +self.desc_metadata.ix['dscribe_SOAP_coordinates']['file_ending']

            structure.write(coord_filename_in, format=format_geometry)
            structure.info['dscribe_SOAP_coord_filename_in'] = coord_filename_in
            tar.add(structure.info['dscribe_SOAP_coord_filename_in'],arcname=only_file)
        