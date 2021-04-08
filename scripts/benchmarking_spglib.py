### Script seems to be correct after recheck. One thing that has to be redone for paper: recalculate tables in the appendix. Excluded nanotubes by default
# since have if condition for non-pbc down in the code. 

from collections import Counter

# Import standard Python and NumPy modules.
import sys
import numpy
import os
from ase.atoms import Atoms
from ase.db import connect
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
import json
from collections import defaultdict

from ai4materials.utils.utils_crystals import get_spacegroup
from ase.spacegroup import get_spacegroup as ase_get_spacegroup
from pymatgen.symmetry.groups import sg_symbol_from_int_number
##################
# run with
# conda activate ovito
# ~/apps/ovito-3.0.0-dev284-x86_64/bin/ovitos benchmarking_ovito.py
####################

def read_ase_db(db_path):
    """From the path to an ASE database file, return a list of ASE atom object contained in it.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    db = connect(db_path)

    ase_list = []
    for idx_db in range(len(db)):
        atoms = db.get_atoms(selection=idx_db + 1, add_additional_information=True)
        # put info from atoms.info['data'] back at their original place (atoms.info)
        # this is because when an ASE atoms object is saved into the SQLite database,
        # ASE does not save automatically atoms.info but instead to
        # atoms.info are saved in atoms.info['data']
        #if 'data' in atoms.info.keys():
        #    atoms.info = atoms.info['data']
        if 'key_value_pairs' in atoms.info.keys():
            atoms.info = atoms.info['key_value_pairs']

        ase_list.append(atoms)

    return ase_list


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()
    
    
    
if __name__=='__main__':
    
    
    prototypes_path = '/home/leitherer/nomad/nomad-lab-base/apt/PROTOTYPES'
    datasets = os.listdir(prototypes_path)
    targets_ordered_by_materialtype = defaultdict()
    for dataset in datasets:
        all_protos = os.listdir(os.path.join(prototypes_path,dataset))
        targets_ordered_by_materialtype[dataset] = all_protos    
    
    
    
    
    
    
    
    
    
    spacegroup_true_labels_file = '/home/leitherer/nomad/nomad-lab-base/apt/BENCHMARKING/spacegroups_all_classes.json'
    
    with open(spacegroup_true_labels_file) as jsonfile:
        spacegroup_true_labels = json.load(jsonfile)
    


    spacegroups_all_classes = spacegroup_true_labels
    
    
    
    ase_db_dataset_dir = '/home/leitherer/EVENTS/manuscript/RESULTS/DATASETS_DESCFOLDERS/all_classes/defective_special_parameter/datasets'#'/home/leitherer/nomad/nomad-lab-base/apt/BENCHMARKING/db_folder_paper'#'/home/ziletti/Documents/calc_nomadml/rot_inv_3d/db_ase'
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_pristine' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-0.1%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-0.2%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-0.6%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-1%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-2%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-4%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-5%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-8%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-10%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-12%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-20%' + '.db')
    #ase_db = os.path.join(ase_db_dataset_dir, 'soap_displacements_0.02.db')#'hcp-sc-fcc-diam-bcc_displacement-30%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-50%' + '.db')
    
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_vacancies-1%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_vacancies-2%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_vacancies-5%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_vacancies-10%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_vacancies-20%' + '.db')
    # ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_vacancies-50%' + '.db')
    ase_databases = os.listdir(ase_db_dataset_dir)
    ase_databases = [os.path.join(ase_db_dataset_dir,x) for x in ase_databases]
    ase_databases = [x for x in ase_databases if x[-3:]=='.db']
    
    results_dict_all_classes = {'spglib_tight': defaultdict(list), 'spglib_loose': defaultdict(list)}
    results_dict_designed_classes = {'spglib_tight': defaultdict(list), 'spglib_loose': defaultdict(list)}
    
    
    spglib_tight_accuracies = defaultdict(dict)
    spglib_loose_accuracies = defaultdict(dict)
    
    defect_types = []
    nanotube_classes = ['armchair', 'zigzag', 'chiral']

    # take only 2D mateirals which are not classified as P1
    _2D_mats = targets_ordered_by_materialtype['2D_materials'] 
    _2D_mats_filtered = [_ for _ in _2D_mats if not spacegroups_all_classes[_]=='P1_excluded']
    
    materials_being_allowed = targets_ordered_by_materialtype['Elemental_solids'] + targets_ordered_by_materialtype['Binaries'] #+ _2D_mats_filtered+ targets_ordered_by_materialtype['Quaternaries']
    
    
    allowed_defects = [0.001, 0.006, 0.01, 0.02, 0.04, 0.05, 0.1, 0.2]
    allowed_defects = [str(_) for _ in allowed_defects]
    for ase_db in ase_databases:
        #if not 'pristine' in ase_db:
        #    continue
        if 'pristine' in ase_db:
            pass
        elif not (('displacements' in ase_db) or ('vacancies' in ase_db)):
            continue
        elif not np.isin(ase_db[:-3].split('_')[-1], allowed_defects):
            continue
    
        print('Database '+ase_db)
        current_defect = (ase_db.split('/')[-1])[:-3]
        defect_types.append(current_defect)
        file_key = current_defect
    
        structure_list = read_ase_db(db_path=ase_db)
        


        # SPGLIB
        
        y_true_spglib = []
        y_true_spglib_all_classes = []
        
        y_pred_loose = [] # symprec=0.1
        y_pred_loose_all_classes = []
        y_pred_tight = [] # symprec=1e-3
        y_pred_tight_all_classes = []
        
        for atoms in structure_list:
            if atoms.pbc.all()==False:
                continue
            
            #if not atoms.info['target'] in targets_ordered_by_materialtype['2D_materials']:
            #    continue
            #if 'pristine' in file_key:
            #    atoms = random_displace_atoms(atoms=atoms,noise_distribution='uniform_scaled',displacement_scaled=0.000001,**kwargs)
            atoms.info['spacegroup_nb'] = {}
            atoms.set_pbc(True)
            
            if atoms.info['crystal_structure'].split('_')[0] in nanotube_classes:
                target = atoms.info['label']
                y_true_spglib_all_classes.extend([target])
                y_pred_loose_all_classes.extend(['cannot_classify_nanotubes'])
                y_pred_tight_all_classes.extend(['cannot_classify_nanotubes'])
            else:
                target = spacegroups_all_classes[str(atoms.info['target'])]
                
                y_true_spglib_all_classes.extend([target])
                
                #atom_class_true = spacegroup_dict[target]
                # y_true.extend(atom_class_true)
                
                #mg_structure = AseAtomsAdaptor.get_structure(prototypes[0]*(4,4,4))
                #finder = SpacegroupAnalyzer(mg_structure)
                #symb = finder.get_space_group_symbol()
                
                # loose
                #spno = ase_get_spacegroup(atoms, symprec=0.1).no
                spno = get_spacegroup(atoms, symprec=0.1, angle_tolerance=5)[0]
                sg_sym = sg_symbol_from_int_number(spno,hexagonal=False) # hexagonal should be false, otherwise geth H at the end for rhombohedral groups 160,166,155,146,148,167 etc
                # only include structures from aflow for testing spglib
                if atoms.info['target'] in materials_being_allowed: #targets_ordered_by_materialtype['Binaries'] or atoms.info['target'] in targets_ordered_by_materialtype['Elemental_solids']:
                    y_pred_loose.extend([sg_sym])
                    y_true_spglib.extend([target])
                y_pred_loose_all_classes.extend([sg_sym])
                # tight
                #spno = ase_get_spacegroup(atoms, symprec=1e-3).no
                spno = get_spacegroup(atoms, symprec=1e-4, angle_tolerance=1)[0]
                sg_sym = sg_symbol_from_int_number(spno,hexagonal=False) # hexagonal should be false, otherwise geth H at the end for rhombohedral groups 160,166,155,146,148,167 etc
                # only include structures from aflow for testing spglib
                if atoms.info['target'] in materials_being_allowed:#targets_ordered_by_materialtype['Binaries'] or atoms.info['target'] in targets_ordered_by_materialtype['Elemental_solids']:
                    y_pred_tight.extend([sg_sym])
                    #y_true_spglib.extend([target]) don't need this since already do it above!
                y_pred_tight_all_classes.extend([sg_sym])
                
            atoms.set_pbc(False)
            
        acc_loose_spglib = accuracy_score(y_true_spglib, y_pred_loose)
        acc_loose_spglib_all_classes = accuracy_score(y_true_spglib_all_classes, y_pred_loose_all_classes)
        
        acc_tight_spglib = accuracy_score(y_true_spglib, y_pred_tight)
        acc_tight_spglib_all_classes = accuracy_score(y_true_spglib_all_classes, y_pred_tight_all_classes)
            
        print("Loose: ") 
        print(file_key+' Accuracy spglib classes: {}'.format(acc_loose_spglib))  
        print(file_key+' Accuracy all classes: {}'.format(acc_loose_spglib_all_classes))  
        
        print("Tight: ")  
        print(file_key+' Accuracy spglib classes: {}'.format(acc_tight_spglib))  
        print(file_key+' Accuracy all classes: {}'.format(acc_tight_spglib_all_classes))  
        
        
        spglib_tight_accuracies[file_key]['spglib'] = acc_tight_spglib
        spglib_tight_accuracies[file_key]['all_classes'] = acc_tight_spglib_all_classes
        
        spglib_loose_accuracies[file_key]['spglib'] = acc_loose_spglib
        spglib_loose_accuracies[file_key]['all_classes'] = acc_loose_spglib_all_classes
        
    with open('spglib_tight_accuracies.json', 'w') as outfile:
        json.dump(spglib_tight_accuracies, outfile) 
        
    with open('spglib_loose_accuracies.json', 'w') as outfile:
        json.dump(spglib_loose_accuracies, outfile) 