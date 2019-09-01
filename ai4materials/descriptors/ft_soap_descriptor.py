import os,os.path

# from ai4materials.descriptors.quippy_soap_descriptor import quippy_SOAP_descriptor
from quippy_soap_descriptor import quippy_SOAP_descriptor

from ai4materials.descriptors.base_descriptor import Descriptor
from ai4materials.descriptors.base_descriptor import is_descriptor_consistent

import collections

from scipy.fftpack import fft,fftfreq

import numpy as np

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger('ai4materials')

class FT_SOAP_harmonics(Descriptor):

    def __init__(self,configs=None,p_b_c=False,cutoff=3.0,l_max=6,n_max=9,atom_sigma=0.1,central_weight=0.0,
                 average_over_permuations=False,number_averages=200,atoms_scaling='quantile_nn',
                 atoms_scaling_cutoffs=[10.],number_of_harmonics=158,real_or_imag_or_spec='power_spectrum',
                 discard_full_spectrum=True, extrinsic_scale_factor=1.0, 
                 n_Z=1, Z=26, n_species=1, species_Z=26, scale_element_sensitive=False, return_binary_descriptor=False, unit_norm=False, average_harmonics = False):
        super(FT_SOAP_harmonics, self).__init__(configs=configs)
        
        
        self.p_b_c=p_b_c
        self.cutoff=cutoff
        self.l_max=l_max
        self.n_max=n_max
        self.atom_sigma=atom_sigma
        self.central_weight=central_weight
        average=False # Always set to false s.t. get unflattened SOAP descriptor in calculate() method below
        self.average=average 

        
        self.average_over_permuations=average_over_permuations 
        self.number_averages=number_averages
        self.atoms_scaling=atoms_scaling
        self.atoms_scaling_cutoffs=atoms_scaling_cutoffs
        self.p_b_c=p_b_c
        self.number_of_harmonics=number_of_harmonics
        self.real_or_imag_or_spec=real_or_imag_or_spec
        self.discard_full_spectrum=discard_full_spectrum
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
        
        # analogous to quippy SOAP binary descriptor
        self.return_binary_descriptor = return_binary_descriptor
            
        
        descriptor_options = 'soap '+'cutoff='+str(self.cutoff)+' l_max='+str(self.l_max)+' n_max='+str(self.n_max)+' atom_sigma='+str(self.atom_sigma)+\
                             ' n_Z='+str(self.n_Z)+' Z={'+str(self.Z)+'} n_species='+str(self.n_species)+' species_Z={'+str(self.species_Z)+'} central_weight='+str(self.central_weight)+' average='+str(self.average)             
        self.descriptor_options=descriptor_options
        
        self.unit_norm = unit_norm
        self.average_harmonics = average_harmonics
        
        
    def calculate(self,structure,**kwargs):
        
        if self.return_binary_descriptor:
            
            # split up binary SOAP descriptor into 4 parts (11,12,21,22) (or more in case of higher number of chem. species),
            # and then compute FT SOAP for each of these descriptors
            
            all_harmonics = []
            all_spectra = [] # full spectra (pos and neg freq, Im and Re)
            
            # get occurrences of each atomic numbers, used below to get the number of atoms for each chemical
            # species, which is used for normalization
            all_atomic_numbers = structure.get_atomic_numbers()
            occurences_atomic_numbers = collections.Counter(all_atomic_numbers)
            # Sorted list of unique species
            atomic_numbers = list(set(structure.get_atomic_numbers()))
            #print occurences_atomic_numbers
            for Z in atomic_numbers:
                for species_Z in atomic_numbers:
                    n_Z = 1
                    n_species = 1 
                    return_binary_descriptor = False # elemental solid case
                    average_binary_descriptor = False # can be True or False
                    soap_object=quippy_SOAP_descriptor(self.configs,self.p_b_c,self.cutoff,self.l_max,self.n_max,self.atom_sigma,self.central_weight,
                                                       self.average,self.average_over_permuations,self.number_averages,self.atoms_scaling,self.atoms_scaling_cutoffs, self.extrinsic_scale_factor, 
                                                       n_Z, Z, n_species, species_Z, self.scale_element_sensitive, return_binary_descriptor, 
                                                       average_binary_descriptor)
                    soap_descriptor=soap_object.calculate(structure).info['descriptor']['SOAP_descriptor'] # shape =  (N_atoms, #SOAP components)

                    if any(np.isnan(soap_descriptor.flatten())):
                        #plt.plot(SOAP_descriptor.flatten())
                        #print np.nan_to_num(SOAP_descriptor, copy=True)
                        #plt.plot(np.nan_to_num(SOAP_descriptor, copy=True).flatten())
                        raise ValueError('Nan value encountered in SOAP descriptor.')
                    
                    N_atoms = occurences_atomic_numbers[Z]
                    number_soap_components_one_LAE = soap_descriptor.size/N_atoms # check if this works!
                    #print N_atoms
                    #print number_soap_components_one_LAE
                    #Calculate ft soap 
                    spec=fft(soap_descriptor.flatten(),soap_descriptor.flatten().size)
            
                    if self.real_or_imag_or_spec=='real_part':
                        spec=np.real(spec)
                        normalization_factor=float(number_soap_components_one_LAE*N_atoms)
                    elif self.real_or_imag_or_spec=='imaginary_part':
                        spec=np.imag(spec)
                        normalization_factor=float(number_soap_components_one_LAE*N_atoms)
                    elif self.real_or_imag_or_spec=='power_spectrum':
                        spec=np.real(np.multiply(spec,np.conj(spec)))
                        normalization_factor=np.power(float(number_soap_components_one_LAE*N_atoms),2) # ! squared normalization factor for power spectrum !
                    else:
                        raise NotImplementedError("Parameter real_or_imag_or_spec must be either real_part, imaginary_part or power_spectrum")
                    
                    
                    
                    spec_size=spec.size        
                        
                        
                    """    
                    if spec_size%2==0:
                        pos_part=spec[0:spec_size/2] #take zero component to positive part, see also https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html
                    else:
                        pos_part=spec[0:(spec_size-1)/2]      
                    """
                    freq = fftfreq(spec_size) # inspired by https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
                    pos_freq = freq>=0
                    pos_part = spec[pos_freq] 
                    #print pos_part.shape
                    
                    
                    #Select Fourier coefficients X_k for k=integer * N_atoms    
                    indices_to_select_pos_part=np.arange(0,pos_part.size,N_atoms) 
                    #print N_atoms
                    #print indices_to_select_pos_part
                    new_pos_part=np.take(pos_part,indices_to_select_pos_part)
                    #print new_pos_part.shape
                    
                    ####### ATTENTION
                    # CHANGE NEXT LINE TO [0:self.number_of_harmonics] IF WANT DC COMPONENT!!
                    harmonics_of_spectrum=new_pos_part[1:self.number_of_harmonics]
                    #print harmonics_of_spectrum.shape
                    #Normalize
                    harmonics_of_spectrum=harmonics_of_spectrum/normalization_factor
                    
                    if self.discard_full_spectrum:
                        spec=np.array([])
                    """
                    if any(np.isnan(harmonics_of_spectrum)):
                        print harmonics_of_spectrum
                        print normalization_factor
                        print structure.info['label']
                        print structure
                        raise ValueError('harmonics_of_spectrum falsely equals '+str(harmonics_of_spectrum))
                    """
                    all_harmonics.append(harmonics_of_spectrum)
                    all_spectra.append(spec)
            
            """
            if self.unit_norm:
                norm = np.linalg.norm(concatenated_spectra)
                concatenated_spectra = concatenated_spectra/norm
            """
            if self.average_harmonics:
                all_harmonics = np.mean(np.array(all_harmonics), axis=0)
                descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self), FT_SOAP_full=all_spectra, FT_SOAP_harmonics=all_harmonics)
            else:
                descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self), FT_SOAP_full=all_spectra, FT_SOAP_harmonics=np.array(all_harmonics))
            structure.info['descriptor'] = descriptor_data
            
            return structure

        
        elif False:
            print('OLD IMPLEMENTATION USED')
            # NEED TO  FIND ERROR IN THIS CODE SEGMENT
            
            # split up binary SOAP descriptor into 4 parts (11,12,21,22), and then compute FT SOAP for each of these descriptors
            one_descriptor_length = int( ((soap_descriptor.flatten()).size)/4 ) # WRONG! don't have even number of atom for each species!!
            soap_descriptor_reshaped = (soap_descriptor.flatten()).reshape(4,one_descriptor_length)
            
            concatenated_spectra = np.array([])
            concatenated_full_spectra = np.array([])

            for soap_descriptor in soap_descriptor_reshaped:
                # run through all (partial) soap descriptors and compute FT SOAP                
                
                # The following is used later on for normalization
                N_atoms=structure.get_number_of_atoms()   
                
                if self.average_over_permuations:
                    number_soap_components_one_LAE = soap_descriptor.size/N_atoms # maybe redundant
                else:
                    number_soap_components_one_LAE=one_descriptor_length
                
                #Calculate ft soap 
                spec=fft(soap_descriptor.flatten(),soap_descriptor.flatten().size)
        
                if self.real_or_imag_or_spec=='real_part':
                    spec=np.real(spec)
                    normalization_factor=float(number_soap_components_one_LAE*N_atoms)
                elif self.real_or_imag_or_spec=='imaginary_part':
                    spec=np.imag(spec)
                    normalization_factor=float(number_soap_components_one_LAE*N_atoms)
                elif self.real_or_imag_or_spec=='power_spectrum':
                    spec=np.real(np.multiply(spec,np.conj(spec)))
                    normalization_factor=np.power(float(number_soap_components_one_LAE*N_atoms),2) # ! squared normalization factor for power spectrum !
                else:
                    raise NotImplementedError("Parameter real_or_imag_or_spec must be either real_part, imaginary_part or power_spectrum")
                
                
                
                spec_size=spec.size        
                    
                    
                """    
                if spec_size%2==0:
                    pos_part=spec[0:spec_size/2] #take zero component to positive part, see also https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html
                else:
                    pos_part=spec[0:(spec_size-1)/2]      
                """
                freq = fftfreq(spec_size) # inspired by https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
                pos_freq = freq>=0
                pos_part = spec[pos_freq] 
                print pos_part.shape
                
                
                #Select Fourier coefficients X_k for k=integer * N_atoms    
                indices_to_select_pos_part=np.arange(0,pos_part.size,N_atoms) 
                print N_atoms
                print indices_to_select_pos_part
                new_pos_part=np.take(pos_part,indices_to_select_pos_part)
                print new_pos_part.shape
                harmonics_of_spectrum=new_pos_part[0:self.number_of_harmonics]
                print harmonics_of_spectrum.shape
                #Normalize
                harmonics_of_spectrum=harmonics_of_spectrum/normalization_factor
                
                if self.discard_full_spectrum:
                    spec=np.array([])
                """
                if any(np.isnan(harmonics_of_spectrum)):
                    print harmonics_of_spectrum
                    print normalization_factor
                    print structure.info['label']
                    print structure
                    raise ValueError('harmonics_of_spectrum falsely equals '+str(harmonics_of_spectrum))
                """
                concatenated_full_spectra = np.append(concatenated_full_spectra, spec)
                concatenated_spectra = np.append(concatenated_spectra, harmonics_of_spectrum)
                
            if self.unit_norm:
                norm = np.linalg.norm(concatenated_spectra)
                concatenated_spectra = concatenated_spectra/norm
            
            descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self), FT_SOAP_full=concatenated_full_spectra, FT_SOAP_harmonics=concatenated_spectra)
            structure.info['descriptor'] = descriptor_data
            
            return structure            
            
        else:
            # FT-sOAP for given species
            #Calculate soap
            # average_binary_descriptor=True/False does not matter
            # since are in elemental solid case
            average_binary_descriptor = False
            
            soap_object = quippy_SOAP_descriptor(self.configs,self.p_b_c,self.cutoff,self.l_max,self.n_max,self.atom_sigma,self.central_weight,
                                                self.average,self.average_over_permuations,self.number_averages,self.atoms_scaling,self.atoms_scaling_cutoffs, self.extrinsic_scale_factor, 
                                                self.n_Z, self.Z, self.n_species, self.species_Z, self.scale_element_sensitive, self.return_binary_descriptor, average_binary_descriptor)
            soap_descriptor=soap_object.calculate(structure).info['descriptor']['SOAP_descriptor']
            print 'Concatenated SOAP '+str(soap_descriptor.shape)
            
            # The following is used later on for normalization
            N_atoms=structure.get_number_of_atoms()        
            if self.average_over_permuations:
                number_soap_components_one_LAE = soap_descriptor.size/N_atoms
            else:
                number_soap_components_one_LAE=soap_descriptor.shape[1] 
            
            #Calculate ft soap 
            spec=fft(soap_descriptor.flatten(),soap_descriptor.flatten().size)
    
            if self.real_or_imag_or_spec=='real_part':
                spec=np.real(spec)
                normalization_factor=float(number_soap_components_one_LAE*N_atoms)
            elif self.real_or_imag_or_spec=='imaginary_part':
                spec=np.imag(spec)
                normalization_factor=float(number_soap_components_one_LAE*N_atoms)
            elif self.real_or_imag_or_spec=='power_spectrum':
                spec=np.real(np.multiply(spec,np.conj(spec)))
                normalization_factor=np.power(float(number_soap_components_one_LAE*N_atoms),2) # ! squared normalization factor for power spectrum !
            else:
                raise NotImplementedError("Parameter real_or_imag_or_spec must be either real_part, imaginary_part or power_spectrum")
            
            
            
            spec_size=spec.size        
                
                
            """    
            if spec_size%2==0:
                pos_part=spec[0:spec_size/2] #take zero component to positive part, see also https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.fft.html
            else:
                pos_part=spec[0:(spec_size-1)/2]      
            """
            freq = fftfreq(spec_size) # inspired by https://stackoverflow.com/questions/25735153/plotting-a-fast-fourier-transform-in-python
            pos_freq = freq>=0
            pos_part = spec[pos_freq] 
            
            
            
            #Select Fourier coefficients X_k for k=integer * N_atoms    
            indices_to_select_pos_part=np.arange(0,pos_part.size,N_atoms) 
            new_pos_part=np.take(pos_part,indices_to_select_pos_part)
            
            ####### ATTENTION
            # CHANGE NEXT LINE TO [0:self.number_of_harmonics] IF WANT DC COMPONENT!!
            harmonics_of_spectrum=new_pos_part[1:self.number_of_harmonics]
            #Normalize
            harmonics_of_spectrum=harmonics_of_spectrum/normalization_factor
            
            if self.discard_full_spectrum:
                spec=[]
            """
            if any(np.isnan(harmonics_of_spectrum)):
                print harmonics_of_spectrum
                print normalization_factor
                print structure.info['label']
                print structure
                raise ValueError('harmonics_of_spectrum falsely equals '+str(harmonics_of_spectrum))
            """
            descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self), FT_SOAP_full=spec, FT_SOAP_harmonics=harmonics_of_spectrum)
            structure.info['descriptor'] = descriptor_data
            
            return structure


    def write(self,structure,tar,write_ft_soap_npy=True,write_ft_soap_png=True,write_ft_soap_full_npy=True,write_geo=True,op_id=0,format_geometry='aims'):
        """Write the descriptor to file.

        Parameters:

        structure: `ase.Atoms` obejct
            Atomic structure.

        tar: TarFile object
            TarFile archive where the descriptor is added. This is created internally with `tarfile.open`. 
            
        write_ft_soap_npy: bool,optional (default=True)
            If true, write FT-SOAP descriptor to binary file 
            
        write_ft_soap_png: bool,optional (default=True)
            If True, write to file a png file showing the FT-SOAP descriptor
        
        write_ft_soap_full_npy: bool,optional (default=False)
            If True, write full fft of SOAP descriptor to binary file

        op_id: int, optional (default=0)
            To be done.
            
        write_geo: bool, optional (default=`True`)
            If `True`, write a coordinate file of the structure for which the FT-SOAP descriptor is calculated.

        format_geometry: string, optional (default=`aims`)
            Output format of the geometry file. All ASE valid output formats are accepted.
            For a complete list see: https://wiki.fysik.dtu.dk/ase/ase/io/io.html
        """
        
        if not is_descriptor_consistent(structure, self):
            raise Exception('Descriptor not consistent. Aborting.')        
        
        desc_folder = self.configs['io']['desc_folder']
        descriptor_info = structure.info['descriptor']['descriptor_info']
        
        ft_soap_descriptor=structure.info['descriptor']['FT_SOAP_harmonics']
        
        
        
        if write_ft_soap_npy:
            
            ft_soap_filename_npy = os.path.abspath(os.path.normpath(os.path.join(desc_folder,
                                                                                   structure.info['label'] +
                                                                                   self.desc_metadata.ix[
                                                                                       'FT_SOAP_harmonics'][
                                                                                       'file_ending'])))
            only_file=structure.info['label'] + self.desc_metadata.ix['FT_SOAP_harmonics']['file_ending']
            
            np.save(ft_soap_filename_npy, ft_soap_descriptor)
            structure.info['FT_SOAP_harmonics_filename_npy'] = ft_soap_filename_npy
            tar.add(structure.info['FT_SOAP_harmonics_filename_npy'],arcname=only_file) 
            
        if write_ft_soap_png:

            image_ft_soap_filename_png = os.path.abspath(os.path.normpath(os.path.join(desc_folder,
                                                                                   structure.info['label'] +
                                                                                   self.desc_metadata.ix[
                                                                                       'FT_SOAP_harmonics_image'][
                                                                                       'file_ending'])))
            only_file=structure.info['label'] + self.desc_metadata.ix['FT_SOAP_harmonics_image']['file_ending']
            
            plt.title(structure.info['label']+' FT SOAP descriptor ')
            plt.xlabel('FT SOAP component')
            plt.ylabel('FT SOAP value')
            plt.plot(ft_soap_descriptor)
            plt.savefig(image_ft_soap_filename_png)
            plt.close()
            structure.info['FT_SOAP_harmonics_filename_png'] = image_ft_soap_filename_png
            tar.add(structure.info['FT_SOAP_harmonics_filename_png'],arcname=only_file)             
        
        if write_ft_soap_full_npy:
            
            full_fft=structure.info['descriptor']['FT_SOAP_full']
            
            ft_soap_full_filename_npy = os.path.abspath(os.path.normpath(os.path.join(desc_folder,
                                                                                   structure.info['label'] +
                                                                                   self.desc_metadata.ix[
                                                                                       'FT_SOAP_full_fft'][
                                                                                       'file_ending'])))
            only_file=structure.info['label'] + self.desc_metadata.ix['FT_SOAP_full_fft']['file_ending']
            
            np.save(ft_soap_full_filename_npy, full_fft)
            structure.info['FT_SOAP_full_filename_npy'] = ft_soap_full_filename_npy
            tar.add(structure.info['FT_SOAP_full_filename_npy'],arcname=only_file)   
            
        if write_geo:
            
            coord_filename_in = os.path.abspath(os.path.normpath(os.path.join(desc_folder, structure.info['label'] +
                                                                                  self.desc_metadata.ix['FT_SOAP_harmonics_coordinates'][
                                                                                      'file_ending'])))
                                                                                          
            only_file=structure.info['label']+self.desc_metadata.ix['FT_SOAP_harmonics_coordinates']['file_ending']
            
            structure.write(coord_filename_in, format=format_geometry)
            structure.info['FT_SOAP_harmonics_coord_filename_in'] = coord_filename_in
            tar.add(structure.info['FT_SOAP_harmonics_coord_filename_in'],arcname=only_file)        
        

        
