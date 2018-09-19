import os,os.path

from nomadml.descriptors.base_descriptor import Descriptor
from nomadml.descriptors.base_descriptor import is_descriptor_consistent
from nomadml.utils.utils_crystals import scale_structure

from quippy import descriptors
from quippy import Atoms as quippy_Atoms

from ase.io import write as ase_write

import numpy as np


import matplotlib.pyplot as plt

import logging
logger = logging.getLogger('ai4materials')

class quippy_SOAP_descriptor(Descriptor):
    """SOAP descriptor from quippy package
    
    
    """
    
    def __init__(self,configs=None,p_b_c=False,cutoff=3.0,l_max=6,n_max=9,atom_sigma=0.1,central_weight=0.0,average=True,average_over_permuations=False,number_averages=200,atoms_scaling='quantile_nn',atoms_scaling_cutoffs=[10.]):
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
        
        descriptor_options='soap '+'cutoff='+str(cutoff)+' l_max='+str(l_max)+' n_max='+str(n_max)+' atom_sigma='+str(atom_sigma)+' n_Z='+str(1)+' Z={'+str(26)+'} central_weight='+str(central_weight)+' average='+str(average)              
        self.descriptor_options=descriptor_options
        
        
    def calculate(self,structure,**kwargs):
        
        atoms = scale_structure(structure, scaling_type=self.atoms_scaling,
                                atoms_scaling_cutoffs=self.atoms_scaling_cutoffs)

                       
        #Define descritpor
        desc=descriptors.Descriptor(self.descriptor_options)
        
        #Define structure as quippy Atoms object
        filename=str(atoms.info['label'])+'.xyz'
        ase_write(filename,atoms,format='xyz')
        struct=quippy_Atoms(filename)
        struct.set_pbc(self.p_b_c)
        
        #Remove redundant files that have been created
        if os.path.exists(filename):
            os.remove(filename)
        if os.path.exists(filename+'.idx'):
            os.remove(filename+'.idx')
        
        #Compute SOAP descriptor
        struct.set_cutoff(desc.cutoff())
        struct.calc_connect()
        SOAP_descriptor=desc.calc(struct)['descriptor']
        
        if self.average_over_permuations:
            #average over different orders
            SOAP_proto_averaged=np.zeros(SOAP_descriptor.size)
            SOAP_proto_copy=SOAP_descriptor
            for i in range(self.number_averages):
                np.random.shuffle(SOAP_proto_copy)
                SOAP_proto_averaged=np.add(SOAP_proto_averaged,SOAP_proto_copy.flatten())
            SOAP_proto_averaged=np.array([x/float(self.number_averages) for x in SOAP_proto_averaged])
            SOAP_descriptor=SOAP_proto_averaged              
            
        
        if self.average:
            SOAP_descriptor=SOAP_descriptor.flatten() # if get averaged LAE, then default output shape is (1,316), hence flatten()
        
        
        descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self), SOAP_descriptor=SOAP_descriptor)

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
            
            plt.title(structure.info['label']+' SOAP descriptor ')
            plt.xlabel('SOAP component')
            plt.ylabel('SOAP value')
            plt.plot(soap_descriptor)
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
        