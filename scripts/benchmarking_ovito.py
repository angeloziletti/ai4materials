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

filepath = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/structures_for_paper/four_grains/four_grains_poly.xyz'
# Load the simulation dataset to be analyzed.
node = import_file(filepath, columns=["Particle Type", "Position.X", "Position.Y", "Position.Z"])

# Create bonds.
#node.modifiers.append(BondAngleAnalysisModifier())
#node.modifiers.append(PolyhedralTemplateMatchingModifier())
node.modifiers.append(CommonNeighborAnalysisModifier(mode=CommonNeighborAnalysisModifier.Mode.AdaptiveCutoff))
#node.modifiers.append(CommonNeighborAnalysisModifier(mode=CommonNeighborAnalysisModifier.Mode.FixedCutoff))

# Let OVITO's data pipeline do the heavy work.
node.compute()

# A two-dimensional array containing the three CNA indices
# computed for each bond in the system.
y_pred = list(node.output.particle_properties['Structure Type'].array)

acna_classes = ['None', '225', '194', '229', 'Ic', '221']
ptm_classes = ['None', '225', '194', '229', 'Ic', '221']
baa_classes = ['None', '225', '194', '229', 'Ic']

#y_pred = [classes[item] for item in y_pred]


print(Counter(y_pred))

import sys
sys.exit()
# Loop over particles and print their CNA indices.
for idx_particle, particle_index in enumerate(range(node.output.number_of_particles)):
    #pass
    # Print particle index (1-based).
    #sys.stdout.write("%i " % (particle_index + 1))

    #outname = 'BondAngleAnalysis.counts.'

    #print(node.output.particle_properties['Structure Type'].array[idx_particle])
    print(y_pred[idx_particle])

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
