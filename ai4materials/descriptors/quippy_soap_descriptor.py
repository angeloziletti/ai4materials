import os,os.path

from ai4materials.descriptors.base_descriptor import Descriptor
from ai4materials.descriptors.base_descriptor import is_descriptor_consistent
from ai4materials.utils.utils_crystals import scale_structure

from quippy import descriptors


from ase.io import write as ase_write

import numpy as np

import matplotlib.pyplot as plt

from collections import Counter

import logging
logger = logging.getLogger('ai4materials')

class quippy_SOAP_descriptor(Descriptor):
    """SOAP descriptor as implemented in quippy package, which is available for non-commercial use at https://github.com/libAtoms/QUIP
    Please refer to the followin publications for more details on the SOAP descriptor:
    [1] A. P. Bartok et al. Physical Review Letters 104 136403 (2010) 
    [2] A. P. Bartok et al. Physical Review B 87 184115 (2013) 
    [3] S. De et al. Physical Chemistry Chemical Physics 18, 13754 (2016)
    While this code can be used to calculate the SOAP descriptor in its most general form, we 
    use an averaged version of it (see also Leitherer et. al. 2021) such that a given atomic structure is represented with a fixed length 
    vector, independent from the number of atoms and chemical species. In particular, no cross-terms are included, i.e., Eq. 16 in [3] 
    is only evaluated for alpha=beta, while for a compound AB we inspect all environments where A or B are considered as center
    and only A or B atoms as neighbors (this is accomplished via the keywords 'Z' and 'species_Z' in the definition of the descriptor options).
    
    Parameters:
    
    configs: dict
    Contains configuration information such as folders for input and output
    (e.g. `desc_folder`, `tmp_folder`), logging level, and metadata location.
    See also :py:mod:`ai4materials.utils.utils_config.set_configs`.
    
    atoms_scaling: string, optional (default='quantile_nn')
        Type of scaling used in the atom structure scaling. See :py:mod:`ai4materials.utils.utils_crystals.scale_structure`
        for more details.

    atoms_scaling_cutoffs: list of float, optional (default=[10.0])
        List of cutoffs to be used in the determination of the lengthscale of the system to be used in
        :py:mod:`ai4materials.utils.utils_crystals.scale_structure`.

    extrinsic_scale_factor: float, optional (default=1.0)
        Scale the structure by another factor on top of the one obtained by isotropic scaling.
        This can be used to do data augmentation on the scaling factor. A factor of 0.95 will scale the structure by
        0.95, thus enlarging it. The default is 1.0, i.e. no extrinsic scaling.
    """
    
    def __init__(self,configs=None,p_b_c=False,cutoff=4.0,l_max=6,n_max=9,atom_sigma=0.1,central_weight=0.0,
                 average=True,average_over_permuations=False,number_averages=200,atoms_scaling='quantile_nn',atoms_scaling_cutoffs=[10.], extrinsic_scale_factor=1.0,
                 n_Z=1, Z=26, n_species=1, species_Z=26, scale_element_sensitive=True, return_binary_descriptor=True, average_binary_descriptor=True, min_atoms=1, shape_soap = 316,
                 constrain_nn_distances=False, version='py3'):
        super(quippy_SOAP_descriptor, self).__init__(configs=configs)
        
        self.p_b_c=p_b_c
        self.cutoff=cutoff
        self.l_max=l_max
        self.n_max=n_max
        self.atom_sigma=atom_sigma
        self.central_weight=central_weight
        self.average=average
        
        self.average_over_permuations=average_over_permuations
        self.number_averages=number_averages
        
        self.atoms_scaling=atoms_scaling
        self.atoms_scaling_cutoffs=atoms_scaling_cutoffs
        self.extrinsic_scale_factor = extrinsic_scale_factor
        
        # From quippy documentation: https://libatoms.github.io/QUIP/Tutorials/quippy-descriptor-tutorial.html#A-many-body-descriptor:-SOAP
        # Atomic numbers to be considered for central atom, e.g. Z={1 6}
        self.Z = Z
        # How many different types of central atoms to consider
        self.n_Z = n_Z
        # Number of species for the descriptor
        self.n_species = n_species
        # Atomic number of species, e.g. species_Z={1 6}
        self.species_Z = species_Z
        self.scale_element_sensitive = scale_element_sensitive
        
        # If return_binary_descriptor=True, then return descriptor (11,12,21,22),
        # with ij being the (averaged) SOAP descriptor where sit on atoms of 
        # species i and only consider atoms with species j as neighbors.
        self.return_binary_descriptor = return_binary_descriptor
        # if average_binary_descriptor=True, average over soap vectors from different species/ the corresponding chem. envs.
        # Default = False, i.e., [(1,1)-SOAP vector, (1,2) SOAP_vector), ... ] is returned
        self.average_binary_descriptor = average_binary_descriptor
        
        # minimum number of atoms (important for polycrystal application, otherwise will get lots of errors)
        self.min_atoms = min_atoms
        self.shape_soap = shape_soap # important if nber aotms < min_atoms because then need shape!
        
        # for some structures, nn may be smaller than default thershold in utils_crystals' get_nndistance function
        self.constrain_nn_distances = constrain_nn_distances
        
        descriptor_options = 'soap '+'cutoff='+str(self.cutoff)+' l_max='+str(self.l_max)+' n_max='+str(self.n_max)+' atom_sigma='+str(self.atom_sigma)+\
                             ' n_Z='+str(self.n_Z)+' Z={'+str(self.Z)+'} n_species='+str(self.n_species)+' species_Z={'+str(self.species_Z)+'} central_weight='+str(self.central_weight)+' average='+str(self.average)              
        self.descriptor_options=descriptor_options
        self.version = version
        
        
    def calculate(self,structure,**kwargs):
        
        # HACK to get right PBC for 2D materials and Nanotubes
        # use pbc as specified in the structure ITSELF
        self.p_b_c = structure.get_pbc()
        
        # the following code gives errors!
        #if (self.p_b_c).any()==False:
        #    structure.cell = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        
        #if (self.p_b_c).all()==False:
        #    structure.cell = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        
        if type(self.p_b_c)==list or type(self.p_b_c)==np.ndarray:
            for idx,p_b_c_component in enumerate(self.p_b_c):
                if p_b_c_component == False:
                    #structure.cell[self.p_b_c[idx]] = [0.0,0.0,0.0]
                    structure.cell[idx] = [0.0, 0.0, 0.0]
        elif self.p_b_c==False:
            structure.set_cell( [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]] )
        else:
            raise ValueError("Format of cell not known.")
        
        structure.set_pbc(self.p_b_c)
        #print(self.p_b_c)
        logger.info("Structure: "+str(structure))
        # Important for  avoiding crash of polycrystal code
        # Uncommented part: look at TOTAL number of atoms. Now do it element-specific (i.e. each species needs to be there min_atoms # times, otherwise this species will not be considered and treated as vacancy)
        """
        if len(structure)<self.min_atoms:
            
            soap_desc = np.full(self.shape_soap, np.nan)
            descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self), SOAP_descriptor=soap_desc)
            structure.info['descriptor'] = descriptor_data
            return structure 
        """
        # get all atomic numbers and the unique values
        atomic_numbers = structure.get_atomic_numbers()
        atomic_numbers_unique = list(set(atomic_numbers))
        
        occurences_species = Counter(atomic_numbers)
        species_to_delete = []
        for species in atomic_numbers_unique:
            if occurences_species[species]<self.min_atoms:
                species_to_delete.append(species)
        del structure[[atom.index for atom in structure if atom.number in species_to_delete]]
        
        if len(structure)==0: # case that had to delete all species, return nan
            logger.info("Structure empty after deleting all under-represented species")
            soap_desc = np.full(self.shape_soap, np.nan)
            descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self), SOAP_descriptor=soap_desc)
            structure.info['descriptor'] = descriptor_data
            return structure        
        
        
            
        if self.return_binary_descriptor:
            """
            if self.scale_element_sensitive:
                pass
            else:
                raise ValueError('Need to scale element-sensitive for binary descriptor!')
                return []
            """
            atomic_numbers = list(set(structure.get_atomic_numbers()))
            all_descriptors = []
            for Z in atomic_numbers:
                for species_Z in atomic_numbers:
                    #
                    #if Z==species_Z and len(structure[structure.get_atomic_numbers()==species_Z])<=1:
                    #    all_descriptors.append(np.full(316, 0.0))
                    #    continue
                    #
                    n_Z = 1
                    n_species = 1
                    #print(Z, species_Z)
                    atoms = scale_structure(structure, scaling_type=self.atoms_scaling,
                                            atoms_scaling_cutoffs=self.atoms_scaling_cutoffs, extrinsic_scale_factor=self.extrinsic_scale_factor,
                                            element_sensitive=self.scale_element_sensitive, central_atom_species=Z, neighbor_atoms_species=species_Z,
                                            constrain_nn_distances=self.constrain_nn_distances)
                                            
                    #Define descritpor - all options stay untouched, i.e., as provided by the intial call, but the species parameter are changed
                    descriptor_options = 'soap '+'cutoff='+str(self.cutoff)+' l_max='+str(self.l_max)+' n_max='+str(self.n_max)+' atom_sigma='+str(self.atom_sigma)+\
                                         ' n_Z='+str(n_Z)+' Z={'+str(Z)+'} n_species='+str(n_species)+' species_Z={'+str(species_Z)+'} central_weight='+str(self.central_weight)+' average='+str(self.average)  
                    desc=descriptors.Descriptor(descriptor_options)
                    
                    if self.version == 'py3':
                        SOAP_descriptor = desc.calc(atoms)['data'].flatten()
                    else:
                        #Define structure as quippy Atoms object
                        #filename=str(atoms.info['label'])+'.xyz'
                        #ase_write(filename,atoms,format='xyz')
                        #struct=quippy_Atoms(filename)
                        from quippy import Atoms as quippy_Atoms
                        struct=quippy_Atoms(atoms)     # Seems to work fine like this. (rather than creating xyz file first)
                        struct.set_pbc(self.p_b_c)
                        
                        #Remove redundant files that have been created
                        #if os.path.exists(filename):
                        #    os.remove(filename)
                        #if os.path.exists(filename+'.idx'):
                        #    os.remove(filename+'.idx')
                        
                        #Compute SOAP descriptor
                        struct.set_cutoff(desc.cutoff())
                        struct.calc_connect()
                        SOAP_descriptor=desc.calc(struct)['descriptor']
                        #print 'SOAP '+str(SOAP_descriptor.flatten().shape)                
                    
                    if any(np.isnan(SOAP_descriptor.flatten())):
                        #plt.plot(SOAP_descriptor.flatten())
                        #print np.nan_to_num(SOAP_descriptor, copy=True)
                        #plt.plot(np.nan_to_num(SOAP_descriptor, copy=True).flatten())
                        raise ValueError('Nan value encountered in SOAP descriptor.')
                    
                    
                    if self.average_over_permuations:
                        #average over different orders
                        SOAP_proto_averaged=np.zeros(SOAP_descriptor.size)
                        SOAP_proto_copy=SOAP_descriptor 
                        # To do: SOAP_proto_copy is not a real copy...
                        # The right way: http://henry.precheur.org/python/copy_list.html
                        # or just a = ..., b = np.array(a)
                        for i in range(self.number_averages):
                            np.random.shuffle(SOAP_proto_copy)
                            SOAP_proto_averaged=np.add(SOAP_proto_averaged,SOAP_proto_copy.flatten())
                        SOAP_proto_averaged=np.array([x/float(self.number_averages) for x in SOAP_proto_averaged])
                        SOAP_descriptor=SOAP_proto_averaged              
                        
                    
                    #if self.average:
                    #    SOAP_descriptor=SOAP_descriptor.flatten() # if get averaged LAE, then default output shape is (1,316), hence flatten()
                    all_descriptors.append(SOAP_descriptor.flatten())
            
            #if len(all_descriptors)==0: # if choose to skip environments with only one atom --> all_descriptors may be empty. That's why append nan array to avoid 
            # mistakes eg in make_strided_pattern_matching_dataset when structure.info['descriptor'][desc_metadata][:] = np.nan is used!
            #    all_descriptors.append(np.full(self.shape_soap, np.nan))
            
            if self.average_binary_descriptor:
                all_descriptors = np.mean(np.array(all_descriptors), axis=0)
                descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self), SOAP_descriptor=all_descriptors)
            else:
                descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self), SOAP_descriptor=np.array(all_descriptors))
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
                                                                                       'quippy_SOAP'][
                                                                                       'file_ending'])))
            only_file=structure.info['label'] + self.desc_metadata.ix['quippy_SOAP']['file_ending']
            
            np.save(soap_filename_npy, soap_descriptor)
            structure.info['quippy_SOAP_descriptor_filename_npy'] = soap_filename_npy
            tar.add(structure.info['quippy_SOAP_descriptor_filename_npy'],arcname=only_file) 
            
        if write_soap_png:

            image_soap_filename_png = os.path.abspath(os.path.normpath(os.path.join(desc_folder,
                                                                                   structure.info['label'] +
                                                                                   self.desc_metadata.ix[
                                                                                       'quippy_SOAP_image'][
                                                                                       'file_ending'])))
            only_file=structure.info['label'] + self.desc_metadata.ix['quippy_SOAP_image']['file_ending']
            
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
            structure.info['quippy_SOAP_descriptor_filename_png'] = image_soap_filename_png
            tar.add(structure.info['quippy_SOAP_descriptor_filename_png'],arcname=only_file)             
        
        if write_geo:
            
            coord_filename_in = os.path.abspath(os.path.normpath(os.path.join(desc_folder, structure.info['label'] +
                                                                                  self.desc_metadata.ix['quippy_SOAP_coordinates'][
                                                                                      'file_ending'])))
            
            only_file=structure.info['label'] +self.desc_metadata.ix['quippy_SOAP_coordinates']['file_ending']

            structure.write(coord_filename_in, format=format_geometry)
            structure.info['quippy_SOAP_coord_filename_in'] = coord_filename_in
            tar.add(structure.info['quippy_SOAP_coord_filename_in'],arcname=only_file)
        
