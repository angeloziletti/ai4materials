import os,os.path

from nomadml.descriptors.quippy_soap_descriptor import quippy_SOAP_descriptor

from nomadml.descriptors.base_descriptor import Descriptor
from nomadml.descriptors.base_descriptor import is_descriptor_consistent


from scipy.fftpack import fft

import numpy as np

import matplotlib.pyplot as plt

import logging
logger = logging.getLogger('ai4materials')

class FT_SOAP_harmonics(Descriptor):

    def __init__(self,configs=None,p_b_c=False,cutoff=3.0,l_max=6,n_max=9,atom_sigma=0.1,central_weight=0.0,average_over_permuations=False,number_averages=200,atoms_scaling='quantile_nn',atoms_scaling_cutoffs=[10.],number_of_harmonics=158,real_or_imag_or_spec='spectrum'):
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
        
        descriptor_options='soap '+'cutoff='+str(cutoff)+' l_max='+str(l_max)+' n_max='+str(n_max)+' atom_sigma='+str(atom_sigma)+' n_Z='+str(1)+' Z={'+str(26)+'} central_weight='+str(central_weight)+' average='+str(average)              
        self.descriptor_options=descriptor_options
        
        
        
        
    def calculate(self,structure,**kwargs):
        #self,configs=None,p_b_c=False,cutoff=3.0,l_max=6,n_max=9,atom_sigma=0.1,central_weight=0.0,average=True,average_over_permuations=False,number_averages=200,atoms_scaling='quantile_nn',atoms_scaling_cutoffs=[10.]
        #Calculate soap
        soap_object=quippy_SOAP_descriptor(self.configs,self.p_b_c,self.cutoff,self.l_max,self.n_max,self.atom_sigma,self.central_weight,self.average,self.average_over_permuations,self.number_averages,self.atoms_scaling,self.atoms_scaling_cutoffs)
        soap_descriptor=soap_object.calculate(structure).info['descriptor']['SOAP_descriptor']
        number_soap_components_one_LAE=soap_descriptor.shape[1] # used later on for normalization
        
        
        #Calculate ft soap 
        spec=fft(soap_descriptor.flatten(),soap_descriptor.flatten().size)

        if self.real_or_imag_or_spec=='real_part':
            spec=np.real(spec)
        elif self.real_or_imag_or_spec=='imaginary_part':
            spec=np.imag(spec)
        elif self.real_or_imag_or_spec=='power_spectrum':
            spec=np.real(np.multiply(spec,np.conj(spec)))
        else:
            raise NotImplementedError("Parameter real_or_imag_or_spec must be either real_part, imaginary_part or power_spectrum")
        
        
        
        spec_size=spec.size        
            
        if spec_size%2==0:
            pos_part=spec[0:spec_size/2] #take zero component to positive part
        else:
            pos_part=spec[0:(spec_size-1)/2]      
        
        #Select Fourier coefficients X_k for k=integer * N_atoms    
        N_atoms=structure.get_number_of_atoms()
        indices_to_select_pos_part=np.arange(0,pos_part.size,N_atoms) 
        new_pos_part=np.take(pos_part,indices_to_select_pos_part)
        harmonics_of_spectrum=new_pos_part[:self.number_of_harmonics]
        #Normalize
        harmonics_of_spectrum=harmonics_of_spectrum/float(number_soap_components_one_LAE*N_atoms)
        
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
        

        
