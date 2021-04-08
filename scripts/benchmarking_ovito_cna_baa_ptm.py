import ovito
print("Hello, this is OVITO %i.%i.%i" % ovito.version)

# Import OVITO modules.
from ovito.io import *
from ovito.modifiers import *
from ovito.data import *
from collections import Counter

# Import standard Python and NumPy modules.
import sys
import numpy
import os
from ovito.pipeline import StaticSource, Pipeline
from ovito.io.ase import ase_to_ovito
from ase.atoms import Atoms
from ase.db import connect
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import itertools
import json
from collections import defaultdict
#import pandas as pd
##################
# Angelo: run with
# conda activate ovito
# ~/apps/ovito-3.0.0-dev284-x86_64/bin/ovitos benchmarking_ovito.py
#
# My local machine:
# conda activate ovito maybe not necessary
# /home/leitherer/ovito-3.0.0-dev564-x86_64/bin/ovitos benchmarking_ovito_generalized.py
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
    spacegroup_true_labels_file = '/home/leitherer/nomad/nomad-lab-base/apt/BENCHMARKING/spacegroups_all_classes.json'
    
    with open(spacegroup_true_labels_file) as jsonfile:
        spacegroup_true_labels = json.load(jsonfile)
    

    # PTM
    """
    0 = Other, unknown coordination structure
    1 = FCC, face-centered cubic
    2 = HCP, hexagonal close-packed
    3 = BCC, body-centered cubic
    4 = ICO, icosahedral coordination
    5 = SC, simple cubic
    6 = Cubic diamond
    7 = Hexagonal diamond
    8 = Graphene
    """
    # ack_joines
    """
    0 = Other, unknown coordination structure
    1 = FCC, face-centered cubic
    2 = HCP, hexagonal close-packed
    3 = BCC, body-centered cubic
    4 = ICO, icosahedral coordination
    """
    # CNA
    """
    0 = Other, unknown coordination structure
    1 = FCC, face-centered cubic
    2 = HCP, hexagonal close-packed
    3 = BCC, body-centered cubic
    4 = ICO, icosahedral coordination
    """
        
    classes_int = dict(ack_jones=['None', '225', '194', '229', 'Ic'], cna=['None', '225', '194', '229', 'Ic'],
                       a_cna=['None', '225', '194', '229', 'Ic'], 
                       ptm=['None', '225', '194', '229', 'Ic', '221', '227', '194', '191'],
                       baa=['None', '225', '194', '229', 'Ic'])
                   
    classes = dict(ack_jones=['None', 'Fm-3m', 'P6_3/mmc', 'Im-3m', 'Ic'], cna=['None', 'Fm-3m', 'P6_3/mmc', 'Im-3m', 'Ic'],
                   a_cna=['None', 'Fm-3m', 'P6_3/mmc', 'Im-3m', 'Ic'], 
                   ptm=['None', 'Fm-3m', 'P6_3/mmc', 'Im-3m', 'Ic', 'Pm-3m', 'Fd-3m', 'P6_3/mmc', 'P6/mmm'],
                   baa=['None', 'Fm-3m', 'P6_3/mmc', 'Im-3m', 'Ic'])
                   
    classes_allowed = dict(ack_jones=['fcc_Cu_A_cF4_225_a', 'hcp_Mg_A_hP2_194_c', 'bcc_W_A_cI2_229_a'], a_cna=['fcc_Cu_A_cF4_225_a', 'hcp_Mg_A_hP2_194_c', 'bcc_W_A_cI2_229_a'],
                           cna=['fcc_Cu_A_cF4_225_a', 'hcp_Mg_A_hP2_194_c', 'bcc_W_A_cI2_229_a'], 
                            ptm=['fcc_Cu_A_cF4_225_a', 'hcp_Mg_A_hP2_194_c', 'bcc_W_A_cI2_229_a', 'sc_alphaPo_A_cP1_221_a', 'diam_C_A_cF8_227_a', 'HexDiam_C_A_hP4_194_f', 'C',
                                 'L10_CuAu', 'L12_Cu3Au', 'CsCl', 'Zincblende_ZnS', 'Wurtzite_ZnS']) # C; stands for graphene
    
    # ORDERING TYPES - ONLY FOR PTM
    ptm_labels = ['other', 'fcc', 'hcp', 'bcc', 'Ic', 'sc', 'diam', 'hexdiam', 'graph']
    ordering_types = ['None', 'pure', 'L10', 'L12_A', 'L12_B', 'B2', 'Zincblende_Wurtzite', 'Boron_Nitride'] #, 'Boron_Nitride'] # TODO : treatment of Boron nitride.. was not clear from the documentation...
    ordering_types_true_labels = {0: 'None', 1: 'pure', 2: 'L10_CuAu', 3: 'L12_Cu3Au', 4:'L12_Cu3Au', 5: 'Cscl', 6:'Zincblende_Wurtzite', 7: 'Boron_Nitride'} # BN set to none...
    ptm_spacegroups = classes['ptm']
    ptm_labels_to_sg = dict(list(zip(ptm_labels, ptm_spacegroups)))
    
    ptm_classes_to_sg = {}
    
    for ptm_label in ptm_labels:
        for ordering_type in ordering_types:
            if ordering_type=='Boron_Nitride':
                ptm_classes_to_sg[(ptm_label, ordering_type)] = 'None'
            if ptm_label=='other' or ordering_type=='None':
                ptm_classes_to_sg[(ptm_label, ordering_type)] = 'None'
            elif ordering_type=='pure':
                ptm_classes_to_sg[(ptm_label, ordering_type)] = ptm_labels_to_sg[ptm_label] # all elemental solids
            elif ptm_label=='fcc' and (ordering_type=='L12_A' or ordering_type=='L12_B'):
                ptm_classes_to_sg[(ptm_label, ordering_type)] = 'Pm-3m' # sg 221 - L12
            elif ptm_label=='bcc' and ordering_type=='B2':
                ptm_classes_to_sg[(ptm_label, ordering_type)] = 'Pm-3m' # sg 221 - CsCl
            elif ptm_label=='fcc' and ordering_type=='L10':
                ptm_classes_to_sg[(ptm_label, ordering_type)] = 'P4/mmm' # sg 123 - L10
            elif ptm_label=='hexdiam' and ordering_type=='Zincblende_Wurtzite':
                ptm_classes_to_sg[(ptm_label, ordering_type)] = 'P6_3mc' # sg 186 - Wurtzite
            elif ptm_label=='diam' and ordering_type=='Zincblende_Wurtzite':
                ptm_classes_to_sg[(ptm_label, ordering_type)] = 'F-43m' # sg 216 - Zincblende
            elif ptm_label=='graph' and ordering_type=='Boron_Nitride':
                ptm_classes_to_sg[(ptm_label, ordering_type)] = 'P6_3/mmc'
            
            
    print(ptm_labels_to_sg)        
            
        
    """
    0 = Other, unknown or no ordering

    1 = Pure (all neighbors like central atom)

    2 = L10

    3 = L12 (A-site)

    4 = L12 (B-site)

    5 = B2

    6 = zincblende / wurtzite
    """
    #ordering_types_true_labels = {0: 'None', 1: 'pure', 2: 'L10_CuAu', 3: 'L12_Cu3Au', 4:'L12_Cu3Au', 5: 'Cscl', 6:'None', 7: 'None'} # ZB and BN are interpreted as None
    # the following two arrays are used below to check if the current proto is elemental or binary:
    #ptm_elemental_classes = ['fcc_Cu_A_cF4_225_a', 'hcp_Mg_A_hP2_194_c', 'bcc_W_A_cI2_229_a', 'sc_alphaPo_A_cP1_221_a', 'diam_C_A_cF8_227_a', 'HexDiam_C_A_hP4_194_f', 'C']
    #ptm_binary_classes = ['L10_CuAu', 'L12_Cu3Au', 'CsCl']

    # for CNA with fixed cutoff: 
    # fcr detecting fcc and hcp, choose 0.854 * a_fcc, which is 3.61491*1.207 = 4.36
    # for bcc: 3.155*1.207 = 3.808

    #filepath = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/structures_for_paper/four_grains/four_grains_poly.xyz'
    #node = import_file(filepath, columns=["Particle Type", "Position.X", "Position.Y", "Position.Z"])
    
    ase_db_dataset_dir = '/home/leitherer/nomad/nomad-lab-base/apt/BENCHMARKING/db_folder_paper_old'#'/home/ziletti/Documents/calc_nomadml/rot_inv_3d/db_ase'
    ase_db_dataset_dir = '/home/leitherer/EVENTS/manuscript/RESULTS/DATASETS_DESCFOLDERS/all_classes/defective_special_parameter/datasets'
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
    
    results_dict_all_classes = {'cna': defaultdict(list), 'a_cna': defaultdict(list), 'ack_jones': defaultdict(list), 'ptm': defaultdict(list)}
    results_dict_designed_classes = {'cna': defaultdict(list), 'a_cna': defaultdict(list), 'ack_jones': defaultdict(list), 'ptm': defaultdict(list)}
    
    modifiers = {'cna': CommonNeighborAnalysisModifier(mode=CommonNeighborAnalysisModifier.Mode.FixedCutoff, cutoff=3.2),
                 'a_cna': CommonNeighborAnalysisModifier(mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff),
                'ack_jones': AcklandJonesModifier(),
                'ptm': PolyhedralTemplateMatchingModifier(rmsd_cutoff=0.1, output_ordering=True)} # Angelo set this to zero... standard value is 0.1
    defect_types = []
    allowed_defects = [0.001, 0.006, 0.01, 0.02, 0.04, 0.05, 0.1, 0.2]
    allowed_defects = [str(_) for _ in allowed_defects]
    for ase_db in ase_databases:
        if 'pristine' in ase_db:
            pass
        elif not (('displacements' in ase_db) or ('vacancies' in ase_db)):
            continue
        elif not np.isin(ase_db[:-3].split('_')[-1], allowed_defects):
            continue
        
        print('Database '+ase_db)
        current_defect = (ase_db.split('/')[-1])[:-3]
        defect_types.append(current_defect)
        
    
        ase_atoms_list = read_ase_db(db_path=ase_db)
        
        
        for method in modifiers:
            #if not method=='cna':
            #    continue
            print('Method '+str(method))   
            
            modifier = modifiers[method]
            if method=='ptm':
                # Enable the identification of cubic and hexagonal diamond structures:
                modifier.structures[PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND].enabled = True
                modifier.structures[PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND].enabled = True
                modifier.structures[PolyhedralTemplateMatchingModifier.Type.GRAPHENE].enabled = True
                modifier.structures[PolyhedralTemplateMatchingModifier.Type.SC].enabled = True
                
                modifier.structures[PolyhedralTemplateMatchingModifier.OrderingType.NONE].enabled = True
                modifier.structures[PolyhedralTemplateMatchingModifier.OrderingType.PURE].enabled = True
                modifier.structures[PolyhedralTemplateMatchingModifier.OrderingType.L10].enabled = True
                modifier.structures[PolyhedralTemplateMatchingModifier.OrderingType.L12_A].enabled = True
                modifier.structures[PolyhedralTemplateMatchingModifier.OrderingType.L12_B].enabled = True
                modifier.structures[PolyhedralTemplateMatchingModifier.OrderingType.B2].enabled = True
                modifier.structures[PolyhedralTemplateMatchingModifier.OrderingType.ZINCBLENDE_WURTZITE].enabled = True
                modifier.structures[PolyhedralTemplateMatchingModifier.OrderingType.BORON_NITRIDE].enabled = True
            
            y_pred = []
            y_true = []
            y_pred_designed_classes = []
            y_true_designed_classes = []
            atom_classes_list = []
            
            for idx, atoms in enumerate(ase_atoms_list):
                #if atoms.pbc.all()==False:
                #    continue
                #print(atoms.info)
                #print(atoms.info['key_value_pairs']['label'])
                #atoms_label = atoms.info['key_value_pairs']['target']
                #print(atoms_label)
                atoms_label = atoms.info['key_value_pairs']['label'] # label longer and more informative, allows to filter out graphene for ptm
                if idx % 1000 == 0:
                    print(idx)
                    
                
                atoms.info['target'] = atoms.info['key_value_pairs']['target']
                if ('armchair' in atoms_label) or \
                   ('zigzag' in atoms_label) or ('chiral' in atoms_label):
                    y_pred_i = ['None']*len(atoms)
                    atom_class_true = [str(atoms.info['target'])] * len(atoms)
                    y_true.extend(atom_class_true)
                    y_pred.extend(y_pred_i)
                else:
                    if atoms.pbc.all()==False: # skip non-periodic structures
                        continue
                    atoms.set_pbc(True)
                    if 'pristine' in ase_db:
                        #atoms *= (3,3,3)
                        replica = 0
                        min_nb_atoms = 100
                        while (atoms*(replica,replica,replica)).get_number_of_atoms()<min_nb_atoms:
                            replica+=1
                        atoms *= (replica, replica, replica)
                    # if str(atoms.info['target']) == '227':
                    """
                    if str(atoms.info['target']) == '227' or str(atoms.info['target']) == '221':
                        pass
                    """
                    #if False:
                    #    pass
                    #else:
                    # atoms = atoms*(2, 2, 2)
                    data = ase_to_ovito(atoms)
                    node = Pipeline(source=StaticSource(data=data))
            
                    node.modifiers.append(modifier)
                    #node.modifiers.append(CommonNeighborAnalysisModifier(mode=CommonNeighborAnalysisModifier.Mode.FixedCutoff))
                    # node.modifiers.append(CommonNeighborAnalysisModifier(mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff))
                    #node.modifiers.append(AcklandJonesModifier())
            
                    # node.modifiers.append(BondAngleAnalysisModifier())
                    # node.modifiers.append(PolyhedralTemplateMatchingModifier(rmsd_cutoff=0.0))
            
                    # Let OVITO's data pipeline do the heavy work.
                    node.compute()
            
                    # A two-dimensional array containing the three CNA indices
                    # computed for each bond in the system.
                    atom_classes = list(node.output.particle_properties['Structure Type'].array)
            
            
                    #AcklandJonesModifier.Type.OTHER(0)
                    #AcklandJonesModifier.Type.FCC(1)
                    #AcklandJonesModifier.Type.HCP(2)
                    #AcklandJonesModifier.Type.BCC(3)
                    #AcklandJonesModifier.Type.ICO(4)
            
                    # CommonNeighborAnalysisModifier.Type.OTHER(0)
                    # CommonNeighborAnalysisModifier.Type.FCC(1)
                    # CommonNeighborAnalysisModifier.Type.HCP(2)
                    # CommonNeighborAnalysisModifier.Type.BCC(3)
                    # CommonNeighborAnalysisModifier.Type.ICO(4)
                    #
                    """
                    classes = dict(ack_jones=['None', '225', '194', '229', 'Ic'], cna=['None', '225', '194', '229', 'Ic'],
                                   ptm=['None', '225', '194', '229', 'Ic', '221', '227', '227'],
                                   baa=['None', '225', '194', '229', 'Ic'])
                    """
                    # ovito 3.0.0
                    # Type.OTHER(0)
                    # PolyhedralTemplateMatchingModifier.Type.FCC(1)
                    # PolyhedralTemplateMatchingModifier.Type.HCP(2)
                    # PolyhedralTemplateMatchingModifier.Type.BCC(3)
                    # PolyhedralTemplateMatchingModifier.Type.ICO(4)
                    # PolyhedralTemplateMatchingModifier.Type.SC(5)
                    # PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND(6)
                    # PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND(7)
                    if method=='ptm':
                        # first get ordering types
                        ordering_type = [ordering_types[item] for item in list(node.output.particle_properties['Ordering Type'].array)]
                        # now get prediction of class
                        class_prediction = [ptm_labels[item] for item in atom_classes]
                        # combine ordering type and prediction to get final space group label
                        zipped_order_class = list(zip(class_prediction, ordering_type))
                        y_pred_i = [ptm_classes_to_sg[z_o_c] for z_o_c in zipped_order_class]
                        if 'NiAs' in atoms.info['target']:
                            print(atoms.info['target'])
                            print(y_pred_i)
                            print(list(node.output.particle_properties['Ordering Type'].array))
                        """
                        # unfinished idea of checking within loop - but may be way too costly
                        y_pred_i = []
                        for y_ in zipped_order_class:
                            class_ = zipped_order_class[0]
                            ordering = zipped_order_class[1]
                            if class_=='pure':
                                y_pred_i.append(0)
                        """
                        
                    else:
                        y_pred_i = [classes[method][item] for item in atom_classes]
            
                    #y_pred_acna = [acna_classes[item] for item in y_pred]
                    # y_pred_baa = [baa_classes[item] for item in y_pred]
                    #print(y_pred_this)
                    #atoms = atoms * (2, 2, 2)
                    atom_class_true = [spacegroup_true_labels[str(atoms.info['target'])]] * len(atoms)
                    print('Accuracy '+str(accuracy_score(atom_class_true, y_pred_i)))
                    """
                    if method=='ptm':
                        if atoms.info['target'] in ptm_elemental_classes:
                            y_pred_i_before = deepcopy(y_pred_i)
                            y_pred_i_before = np.array(y_pred_i_before, dtype=object)
                            true_ordering_type = ['pure'] * len(atoms)
                            pred_ordering_type = np.array([ordering_types_true_labels[item] for item in list(node.output.particle_properties['Ordering Type'].array)], dtype=object)
                            mask = pred_ordering_type!='pure'
                            print(mask)
                            y_pred_i[mask] = 'None' # if for elemental classes the ordering type is guessed wrong - set prediction to False
                            print(y_pred_i==y_pred_i_before)
                        elif atoms.info['target'] in ptm_binary_classes:
                            atom_class_true = [atoms.info['target']] * len(atoms)
                            y_pred_i = np.array([ordering_types_true_labels[item] for item in list(node.output.particle_properties['Ordering Type'].array)], dtype=object)                    
                    """
                    
                    
                    y_true.extend(atom_class_true)
                    y_pred.extend(y_pred_i)
                    atom_classes_list.extend(atom_classes)
                    
                    # only save prediction if the given structure is contained in the classes
                    # the method was defined for
                    method_classes = classes_allowed[method]
                    #print(str(atoms.info['target']))
                    if str(atoms.info['target']) in method_classes:
                        print('################################################################## For '+method+' - Enter designed classes')
                        print('Class '+atoms.info['target'])
                        #print('Prediction '+str(y_pred_i))
                        
                        #print([ordering_types[item] for item in list(node.output.particle_properties['Ordering Type'].array)])
                        #print(y_pred_i)
                        print('Accuracy '+str(accuracy_score(atom_class_true, y_pred_i)))
                        y_pred_designed_classes.extend(y_pred_i)
                        y_true_designed_classes.extend(atom_class_true)
            
            #print(len(y_true))
            print('All classes')
            print('y_true', Counter(y_true))
            print('y_pred', Counter(y_pred))
            #print(Counter(y_true), Counter(y_pred))
            
            
            
            print('Accuracy: {}'.format(accuracy_score(y_true, y_pred)))
            #cnf_matrix = confusion_matrix(y_true, y_pred)
            #np.set_printoptions(precision=4)
            
            #print(cnf_matrix)
            
            print('Designed classes')
            print('y_true', Counter(y_true_designed_classes))
            print('y_pred', Counter(y_pred_designed_classes))
            print('Accuracy: {}'.format(accuracy_score(y_true_designed_classes, y_pred_designed_classes)))
            
            results_dict_all_classes[method][current_defect] = round(accuracy_score(y_true, y_pred), 4)
            results_dict_designed_classes[method][current_defect] = round(accuracy_score(y_true_designed_classes, y_pred_designed_classes), 4)
            
            # y_pred Counter({'194': 583828, '229': 116999, '225': 115152, 'None': 968})
            #ack_jones_classes = ['194', '229', '225', 'None']
            
            # plot_confusion_matrix(cnf_matrix, classes=ack_jones_classes,
            #                       normalize=False, title='Confusion matrix, without normalization')
            # Loop over particles and print their CNA indices.
            #for idx_particle, particle_index in enumerate(range(node.output.number_of_particles)):
                #pass
                # Print particle index (1-based).
                #sys.stdout.write("%i " % (particle_index + 1))
            
                #outname = 'BondAngleAnalysis.counts.'
            
                #print(node.output.particle_properties['Structure Type'].array[idx_particle])
            #    print(y_pred[idx_particle])
            
                # Create local list with CNA indices of the bonds of the current particle.
                #bond_index_list = list(bond_enumerator.bonds_of_particle(particle_index))
                #local_cna_indices = cna_indices[bond_index_list]
            
                # Count how often each type of CNA triplet occurred.
                #unique_triplets, triplet_counts = row_histogram(local_cna_indices)
            
                # Print list of triplets with their respective counts.
                #for triplet, count in zip(unique_triplets, triplet_counts):
                #    sys.stdout.write("%s:%i " % (triplet, count))
            
                # End of particle line
                #sys.stdout.write("\n")
    # SAVE
    save_path = '/home/leitherer/nomad/nomad-lab-base/apt/BENCHMARKING/results_OVITO'
    for method_ in results_dict_all_classes:
        with open(os.path.join(save_path, 'results_dict_all_classes_'+method_+'.json'), 'w') as f:
            json.dump(results_dict_all_classes[method_], f)
    for method_ in results_dict_designed_classes:
        with open(os.path.join(save_path, 'results_dict_designed_classes_'+method_+'.json'), 'w') as f:
            json.dump(results_dict_designed_classes[method_], f)
    
    """
    for dict_ in [results_dict_all_classes, results_dict_designed_classes]:
        methods = dict_.keys()
        defects = dict_[methods[0]].keys()
        
        for method in methods:
    """  
    """    
    # save as pandas dataframe
    all_resultds_dict = {method_:[] for method_ in results_dict_all_classes}
    all_resultds_dict.update()
    for method_ in results_dict_all_classes:
        all_results_dict[method_] = [method_] + results_dict_all_classes[method_]
    """