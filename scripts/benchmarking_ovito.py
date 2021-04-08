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
        if 'data' in atoms.info.keys():
            atoms.info = atoms.info['data']

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

#filepath = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/structures_for_paper/four_grains/four_grains_poly.xyz'
#node = import_file(filepath, columns=["Particle Type", "Position.X", "Position.Y", "Position.Z"])

ase_db_dataset_dir = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/db_ase'
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
ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-30%' + '.db')
# ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_displacement-50%' + '.db')

# ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_vacancies-1%' + '.db')
# ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_vacancies-2%' + '.db')
# ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_vacancies-5%' + '.db')
# ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_vacancies-10%' + '.db')
# ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_vacancies-20%' + '.db')
# ase_db = os.path.join(ase_db_dataset_dir, 'hcp-sc-fcc-diam-bcc_vacancies-50%' + '.db')

ase_atoms_list = read_ase_db(db_path=ase_db)


y_pred = []
y_true = []
atom_classes_list = []

for idx, atoms in enumerate(ase_atoms_list):
    if idx % 1000 == 0:
        print(idx)

    # if str(atoms.info['target']) == '227':
    if str(atoms.info['target']) == '227' or str(atoms.info['target']) == '221':
        pass
    # if False:
    #     pass
    else:
        # atoms = atoms*(2, 2, 2)
        data = ase_to_ovito(atoms)
        node = Pipeline(source=StaticSource(data=data))


        # node.modifiers.append(CommonNeighborAnalysisModifier(mode=CommonNeighborAnalysisModifier.Mode.FixedCutoff))
        # node.modifiers.append(CommonNeighborAnalysisModifier(mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff))
        node.modifiers.append(AcklandJonesModifier())

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
        classes = dict(ack_jones=['None', '225', '194', '229', 'Ic'], cna=['None', '225', '194', '229', 'Ic'],
                       ptm=['None', '225', '194', '229', 'Ic', '221', '227', '227'],
                       baa=['None', '225', '194', '229', 'Ic'])

        # ovito 3.0.0
        # Type.OTHER(0)
        # PolyhedralTemplateMatchingModifier.Type.FCC(1)
        # PolyhedralTemplateMatchingModifier.Type.HCP(2)
        # PolyhedralTemplateMatchingModifier.Type.BCC(3)
        # PolyhedralTemplateMatchingModifier.Type.ICO(4)
        # PolyhedralTemplateMatchingModifier.Type.SC(5)
        # PolyhedralTemplateMatchingModifier.Type.CUBIC_DIAMOND(6)
        # PolyhedralTemplateMatchingModifier.Type.HEX_DIAMOND(7)

        y_pred_i = [classes['cna'][item] for item in atom_classes]

        #y_pred_acna = [acna_classes[item] for item in y_pred]
        # y_pred_baa = [baa_classes[item] for item in y_pred]
        #print(y_pred_this)
        #atoms = atoms * (2, 2, 2)
        atom_class_true = [str(atoms.info['target'])] * len(atoms)
        y_true.extend(atom_class_true)
        y_pred.extend(y_pred_i)
        atom_classes_list.extend(atom_classes)

print(len(y_true))
print('y_true', Counter(y_true))
print('y_pred', Counter(y_pred))
#print(Counter(y_true), Counter(y_pred))



print('Accuracy: {}'.format(accuracy_score(y_true, y_pred)))
cnf_matrix = confusion_matrix(y_true, y_pred)
np.set_printoptions(precision=4)

print(cnf_matrix)

# y_pred Counter({'194': 583828, '229': 116999, '225': 115152, 'None': 968})
ack_jones_classes = ['194', '229', '225', 'None']

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
