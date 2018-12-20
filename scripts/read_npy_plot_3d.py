from mayavi import mlab
import numpy as np
from scipy import ndimage
from mayavi import mlab
import numpy as np
from scipy import ndimage

filenames = []
# filenames.append('Probability_prob_class0.npy')
# filenames.append('Probability_prob_class1.npy')
# filenames.append('Probability_prob_class2.npy')
# filenames.append('Probability_prob_class3.npy')
# filenames.append('Probability_prob_class4.npy')

# filenames.append('Uncertainty (predictive_entropy)_uncertainty_class.npy')
# filenames.append('Uncertainty (mutual_information)_uncertainty_class.npy')
# filenames.append('Uncertainty (variation_ratio)_uncertainty_class.npy')

filenames.append('prob_class0_pristine.npy')
filenames.append('prob_class1_pristine.npy')
filenames.append('prob_class2_pristine.npy')
filenames.append('prob_class3_pristine.npy')
filenames.append('prob_class4_pristine.npy')
filenames.append('uncertainty_class_pristine.npy')
# filenames.append('power_spect.npy')

filenames = ['/home/ziletti/Documents/calc_nomadml/rot_inv_3d/' + item for item in filenames]

for filename in filenames:
    prob = np.load(filename)

    prob = np.nan_to_num(prob)
    # prob = np.abs(ndimage.zoom(prob, (6, 6, 6)))
    # s = (prob - prob.min()) / (prob.max() - prob.min())
    s = prob

    print(prob.shape)
    min = s.min()
    max = s.max()

    mlab.options.offscreen = False
    mlab.clf()
    src = mlab.pipeline.scalar_field(s)

    # mlab.pipeline.volume(src, vmin=0.0, vmax=min + .5 * (max - min))
    mlab.pipeline.iso_surface(src, contours=[s.min() + 0.5 * s.ptp(), ], opacity=0.1)
    # mlab.pipeline.iso_surface(src, contours=[s.min() + 0.5 * (max - min), ], opacity=0.1)
    # mlab.pipeline.iso_surface(src, contours=[max], opacity=0.1)
    # mlab.pipeline.iso_surface(src, contours=[s.max() - 0.5 * s.ptp(), ], )
    # mlab.volume_slice(s, plane_orientation='x_axes', slice_index=88)
    # mlab.volume_slice(s, plane_orientation='y_axes', slice_index=88)
    # mlab.volume_slice(s, plane_orientation='z_axes', slice_index=88)
    obj = mlab.contour3d(s, contours=20, vmin=0.0, vmax=None, opacity=0.5)
    # obj.scene.disable_render = True
    # obj.scene.anti_aliasing_frames = 0
    mlab.colorbar(title='Field intensity', orientation='vertical')

    mlab.outline()
    mlab.show()

mlab.close(all=True)

