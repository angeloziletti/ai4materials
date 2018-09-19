# coding=utf-8
# Copyright 2016-2018 Angelo Ziletti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2018, Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "23/03/18"

import condor
import logging
import os
import matplotlib
# force matplotlib to not use any Xwindows backend - needed for gitlab continous integration
matplotlib.use('Agg')
os.system("export DISPLAY=:0")
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from ai4materials.descriptors.base_descriptor import Descriptor
from ai4materials.descriptors.base_descriptor import is_descriptor_consistent
from ai4materials.utils.utils_crystals import scale_structure
from ai4materials.utils.utils_vol_data import interp_theta_phi_surfaces
from ai4materials.utils.utils_vol_data import get_shells_from_indices
from ai4materials.utils.utils_vol_data import get_slice_volume_indices
from ai4materials.utils.utils_neural_networks import get_activations
import numpy as np
import os
from pyshtools.expand import SHExpandDH
from pyshtools.expand import MakeGridDH
from pyshtools.shclasses import SHCoeffs
from scipy import ndimage
from scipy import fftpack
from scipy.signal import get_window

logger = logging.getLogger('ai4materials')


class DISH(Descriptor):
    """Calculation of the spherical harmonics expansion of diffraction intensity.

    This is the descriptor introduced in Ziletti et al., in preparation (2018). The default values
    of the parameters used are the one used in the reference above.

    Parameters:

    configs: dict
        Contains configuration information such as folders for input and output
        (e.g. `desc_folder`, `tmp_folder`), logging level, and metadata location.
        See also :py:mod:`ai4materials.utils.utils_config.set_configs`.

    param_source: dict, optional (default={'wavelength': 5.0E-12, 'pulse_energy': 1E-6, 'focus_diameter': 1E-6})
        Contains parameters that are passed to the underlying package used for the diffraction computation (Condor).
        This parameters are passed to Condor via: \n
        `src = condor.Source(**self.param_source)` \n
        See http://condor.readthedocs.io/en/latest/config.html#source for the parameters that can be used.

    param_detector: dict, optional (default={'distance': 0.1, 'pixel_size': 4E-4, 'nx': 64, 'ny': 64})
        Contains parameters that are passed to the underlying package used for the diffraction computation (Condor).
        This parameters are passed to Condor via: \n
        `det = condor.Detector(**self.param_detector)` \n
        See http://condor.readthedocs.io/en/latest/config.html#detector for the parameters that can be used.

    atoms_scaling: string, optional (default='avg_nn')
        Type of scaling used in the atom structure scaling. See :py:mod:`ai4materials.utils.utils_crystals.scale_structure`
        for more details.

    atoms_scaling_cutoffs: list of float, optional (default=[4.0, 5.0, 7.0, 9.0, 11.0, 12.0])
        List of cutoffs to be used in the determination of the lengthscale of the system to be used in
        :py:mod:`ai4materials.utils.utils_crystals.scale_structure`.

    extrinsic_scale_factor: float, optional (default=1.0)
        Scale the structure by another factor on top of the one obtained by isotropic scaling.
        This can be used to do data augmentation on the scaling factor. A factor of 0.95 will scale the structure by
        0.95, thus enlarging it. The default is 1.0, i.e. no extrinsic scaling.

    use_mask: bool, optional (default=True)
        If `True`, a mask such that only (`mask_r_min` < points < `mask_r_max`) are kept in the calculated
        diffraction pattern. This is done because the central peak in diffraction is very intense
        but does not carry useful information regarding the atomic structure.

    mask_r_min: int, optional (default=12)
        The intensity of the points in the diffraction pattern with a radius smaller than `mask_r_min` are set to zero.

    mask_r_max: int, optional (default=92)
        The intensity of the points in the diffraction pattern with a radius larger than `mask_r_max` are set to zero.

    theta_bins: int, optional (default=50)
        Bins to be used in the :py:mod:`ai4materials.utils.utils_vol_data.slice_3d_volume` for the theta angle of the
        spherical coordinates. This is used to extract the spherical shells from the 3d volume in cartesian
        coordinates. See :py:mod:`ai4materials.utils.utils_vol_data.slice_3d_volume` for more details.

    phi_bins: int, optional (default=100)
        Bins to be used in the :py:mod:`ai4materials.utils.utils_vol_data.slice_3d_volume` for the phi angle of the
        spherical coordinates. This is used to extract the spherical shells from the 3d volume in cartesian
        coordinates. See :py:mod:`ai4materials.utils.utils_vol_data.slice_3d_volume` for more details.

    theta_bins_fine: int, optional (default=256)
        Bins to be used in the :py:mod:`ai4materials.utils.utils_vol_data.interp_theta_phi_surfaces` for the theta
        angle of the spherical coordinates. This is used to interpolate on a finer grid the spherical shells
        extracted from the 3d volumes - now in spherical coordinates. It can be much higher than ``theta_bins`` because
        the spherical shells are represented in spherical coordinates (2D) and not in cartesian coordinates (3D).
        See :py:mod:`ai4materials.utils.utils_vol_data.interp_theta_phi_surfaces` for more details.

    phi_bins_fine: int, optional (default=512)
        Bins to be used in the :py:mod:`ai4materials.utils.utils_vol_data.interp_theta_phi_surfaces` for the theta
        angle of the spherical coordinates. This is used to interpolate on a finer grid the spherical shells
        extracted from the 3d volumes - now in spherical coordinates. It can be much higher than ``theta_bins`` because
        the spherical shells are represented in spherical coordinates (2D) and not in cartesian coordinates (3D).
        See :py:mod:`ai4materials.utils.utils_vol_data.interp_theta_phi_surfaces` for more details.

    sph_l_cutoff: int, optional (default=32)
        Cut-off for the spherical harmonics expansion. Only spherical harmonics with degree l up to sph_l_cutoff
        will be kept. The reconstruction of the sperical shells for different values of the ``sph_l_cutoff``
        can be visually checked (for each slice) by plotting the reconstructed images via
        ai4materials.descriptors.diffraction3d.plot_concentric_shells_spherical_coords.

    window: str, optional (default='hanning')
        Window to be used when performing the Fourier transform. In principle any window from
        ``scipy.signal.get_window`` can be used.

    nx_fft: int, optional (default=128)
        Size of the Fourier transform along the x axis.

    ny_fft: int, optional (default=128)
        Size of the Fourier transform along the y axis.

    nz_fft: int, optional (default=128)
        Size of the Fourier transform along the z axis.


    .. note::
        This descriptor was introduced in the following article:

        A. Ziletti, A. Leitherer, M. Scheffler, and L. M. Ghiringhelli, "Crystal-structure classification
        via Bayesian deep learning - towards superhuman performance", to be submitted (2018).

        Please cite this manuscript if you find this descriptor useful in your work.

    .. seealso::
        To calculate the three-dimensional diffraction pattern we use the open source software Condor.
        See the Condor webpage for more details: http://condor.readthedocs.io/en/.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    def __init__(self, configs, param_source=None, param_detector=None, user_param_source=None,
                 user_param_detector=None, atoms_scaling='quantile_nn', extrinsic_scale_factor=1.0,
                 atoms_scaling_cutoffs=None, use_mask=True, mask_r_min=12, mask_r_max=92, phi_bins=100, theta_bins=50,
                 phi_bins_fine=512, theta_bins_fine=256, sph_l_cutoff=32, window='hanning', nx_fft=128, ny_fft=128,
                 nz_fft=128, **params):
        super(DISH, self).__init__(configs=configs)

        params = Descriptor.params(self)

        if param_source is None:
            param_source = {'wavelength': 4.0E-12, 'pulse_energy': 1E-3, 'focus_diameter': 1E-3}

        if param_detector is None:
            param_detector = {'distance': 0.1, 'pixel_size': 3E-4, 'nx': 64, 'ny': 64}

        if user_param_source is not None:
            param_source.update(user_param_source)

        if user_param_detector is not None:
            param_detector.update(user_param_detector)

        if atoms_scaling_cutoffs is None:
            atoms_scaling_cutoffs = [5.0, 7.0, 9.0, 11.0, 13.0, 15.0]

        self.atoms_scaling = atoms_scaling
        self.extrinsic_scale_factor = extrinsic_scale_factor
        self.atoms_scaling_cutoffs = atoms_scaling_cutoffs
        self.param_source = param_source
        self.param_detector = param_detector
        self.use_mask = use_mask
        self.mask_r_min = mask_r_min
        self.mask_r_max = mask_r_max
        self.phi_bins = phi_bins
        self.theta_bins = theta_bins
        self.phi_bins_fine = phi_bins_fine
        self.theta_bins_fine = theta_bins_fine
        self.sph_l_cutoff = sph_l_cutoff
        self.n_px = param_detector["nx"]
        self.n_py = param_detector["ny"]
        self.window = window

        assert self.n_px == self.n_py, "Images needs to have equal width and height while we have {0} and {1}".format(
            self.n_px, self.n_py)

        # we assume that the pz is the same as n_px and n_py
        # in condor it is not defined, but from the output we see that it is the same as n_px and n_py
        self.n_pz = self.n_px

        params = {}
        params.update({"param_source": param_source, "param_detector": param_detector, "use_mask": use_mask})

        # add derived parameters
        if nx_fft is None:
            nx_fft = self.n_px * 2
        if ny_fft is None:
            ny_fft = self.n_py * 2
        if nz_fft is None:
            nz_fft = self.n_pz * 2

        self.nx_fft = nx_fft
        self.ny_fft = ny_fft
        self.nz_fft = nz_fft

        assert self.nx_fft == self.ny_fft, "The fft sampling should be cubic while we have {0} and {1}".format(
            self.nx_fft, self.ny_fft)
        assert self.ny_fft == self.nz_fft, "The fft sampling should be cubic while we have {0} and {1}".format(
            self.ny_fft, self.nz_fft)

        self.max_r = self.nx_fft / 2.0

    def calculate(self, structure, min_nb_atoms=20, plot_3d=False, plot_slices=False, plot_slices_sph_coords=False,
                  **kwargs):
        """Calculate the descriptor for the given ASE structure.

        Parameters:

        structure: `ase.Atoms` object
            Atomic structure.

        min_nb_atoms: int, optional (default=20)
            If the structure contains less than ``min_nb_atoms``, the descriptor is not calculated and an array with
            zeros is return as descriptor. This is because the descriptor is expected to be no longer meaningful for
            such a small amount of atoms present in the chosen structure.

        """

        if len(structure) > min_nb_atoms - 1:

            atoms = scale_structure(structure, scaling_type=self.atoms_scaling,
                                    atoms_scaling_cutoffs=self.atoms_scaling_cutoffs,
                                    extrinsic_scale_factor=self.extrinsic_scale_factor)

            # Source
            src = condor.Source(**self.param_source)

            # Detector
            # solid_angle_correction are meaningless for 3d diffraction
            det = condor.Detector(solid_angle_correction=False, **self.param_detector)

            # Atoms
            atomic_numbers = map(lambda el: el.number, atoms)
            atomic_numbers = [atomic_number + 5 for atomic_number in atomic_numbers]
            # atomic_numbers = [82 for atomic_number in atomic_numbers]

            # convert Angstrom to m (CONDOR uses meters)
            atomic_positions = map(lambda pos: [pos.x * 1E-10, pos.y * 1E-10, pos.z * 1E-10], atoms)

            par = condor.ParticleAtoms(atomic_numbers=atomic_numbers, atomic_positions=atomic_positions)

            s = "particle_atoms"
            condor_exp = condor.Experiment(src, {s: par}, det)
            res = condor_exp.propagate3d()

            # retrieve some physical quantities that might be useful for users
            intensity = res["entry_1"]["data_1"]["data"]
            fourier_space = res["entry_1"]["data_1"]["data_fourier"]
            phases = np.angle(fourier_space) % (2 * np.pi)

            # 3D diffraction calculation
            real_space = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(res["entry_1"]["data_1"]["data_fourier"])))
            window = get_window(self.window, self.n_px)
            tot_density = window * real_space.real
            center_of_mass = ndimage.measurements.center_of_mass(tot_density)
            logger.debug("Tot density data dimensions: {}".format(tot_density.shape))
            logger.debug("Center of mass of total density: {}".format(center_of_mass))

            # take the fourier transform of structure in real_space
            fft_coeff = fftpack.fftn(tot_density, shape=(self.nx_fft, self.ny_fft, self.nz_fft))

            # now shift the quadrants around so that low spatial frequencies are in
            # the center of the 2D fourier transformed image.
            fft_coeff_shifted = fftpack.fftshift(fft_coeff)

            # calculate a 3D power spectrum
            power_spect = np.abs(fft_coeff_shifted) ** 2

            if self.use_mask:
                xc = (self.nx_fft - 1.0) / 2.0
                yc = (self.ny_fft - 1.0) / 2.0
                zc = (self.nz_fft - 1.0) / 2.0

                # spherical mask
                a, b, c = xc, yc, zc
                x, y, z = np.ogrid[-a:self.nx_fft - a, -b:self.ny_fft - b, -c:self.nz_fft - c]

                mask_int = x * x + y * y + z * z <= self.mask_r_min * self.mask_r_min
                mask_out = x * x + y * y + z * z >= self.mask_r_max * self.mask_r_max

                for i in range(self.nx_fft):
                    for j in range(self.ny_fft):
                        for k in range(self.nz_fft):
                            if mask_int[i, j, k]:
                                power_spect[i, j, k] = 0.0
                            if mask_out[i, j, k]:
                                power_spect[i, j, k] = 0.0

            # cut the spectrum and keep only the relevant part for crystal-structure recognition of
            # hexagonal closed packed (spacegroup=194)
            # simple cubic (spacegroup=221)
            # face centered cubic (spacegroup=225)
            # diamond (spacegroup=227)
            # body centered cubic (spacegroup=229)
            # this interval (20:108) might need to be varied if other classes are added
            power_spect_cut = power_spect[20:108, 20:108, 20:108]
            # zoom by two times using spline interpolation
            power_spect = ndimage.zoom(power_spect_cut, (2, 2, 2))

            # power_spect.shape = 176, 176, 176
            if plot_3d:
                plot_3d_volume(power_spect)

            vox = np.copy(power_spect)
            logger.debug("nan in data: {}".format(np.count_nonzero(~np.isnan(vox))))

            # optimized
            # these specifications are valid for a power_spect = power_spect[20:108, 20:108, 20:108]
            # and a magnification of 2
            xyz_indices_r = get_slice_volume_indices(vox, min_r=32.0, dr=1.0, max_r=83., phi_bins=self.phi_bins,
                                    theta_bins=self.theta_bins)

            # slow - only for benchmarking the fast implementation below (shells_to_sph, interp_theta_phi_surfaces)
            # (vox_by_slices, theta_phi_by_slices) = _slice_3d_volume_slow(vox)

            # convert 3d shells
            (vox_by_slices, theta_phi_by_slices) = get_shells_from_indices(xyz_indices_r, vox)
            if plot_slices:
                plot_concentric_shells(vox_by_slices, base_folder=self.configs['io']['main_folder'], idx_slices=None,
                                       create_animation=False)

            image_by_slices = interp_theta_phi_surfaces(theta_phi_by_slices, theta_bins=self.theta_bins_fine,
                                                        phi_bins=self.phi_bins_fine)

            if plot_slices_sph_coords:
                plot_concentric_shells_spherical_coords(image_by_slices, base_folder=self.configs['io']['main_folder'],
                                                        idx_slices=None)

            coeffs_list = []
            nl_list = []
            ls_list = []

            for idx_slice in range(image_by_slices.shape[0]):
                logger.debug("img #{} max: {}".format(idx_slice, image_by_slices[idx_slice].max()))

                # set to zero the spherical harmonics coefficients above self.sph_l_cutoff
                coeffs = SHExpandDH(image_by_slices[idx_slice], sampling=2)
                coeffs_filtered = coeffs.copy()
                coeffs_filtered[:, self.sph_l_cutoff:, :] = 0.
                coeffs = coeffs_filtered.copy()

                nl = coeffs.shape[0]
                ls = np.arange(nl)
                coeffs_list.append(coeffs)
                nl_list.append(nl)
                ls_list.append(ls)

            coeffs = np.asarray(coeffs_list).reshape(image_by_slices.shape[0], coeffs.shape[0], coeffs.shape[1],
                                                     coeffs.shape[2])

            sh_coeffs_list = []

            for idx_slice in range(coeffs.shape[0]):
                sh_coeffs = SHCoeffs.from_array(coeffs[idx_slice])
                sh_coeffs_list.append(sh_coeffs)

            sh_spectrum_list = []
            for sh_coeff in sh_coeffs_list:
                sh_spectrum = sh_coeff.spectrum(convention='l2norm')
                sh_spectrum_list.append(sh_spectrum)

            sh_spectra = np.asarray(sh_spectrum_list).reshape(coeffs.shape[0], -1)

            # cut the spherical harmonics expansion to sph_l_cutoff order
            logger.debug('Spherical harmonics spectra maximum before normalization: {}'.format(sh_spectra.max()))
            sh_spectra = sh_spectra[:, :self.sph_l_cutoff]
            sh_spectra = (sh_spectra - sh_spectra.min()) / (sh_spectra.max() - sh_spectra.min())

            # add results in ASE structure info
            descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self),
                                   diffraction_3d_sh_spectrum=sh_spectra)

        else:
            # return array with zeros for structures with less than min_nb_atoms
            sh_spectra = np.zeros((52, int(self.sph_l_cutoff)))
            descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self),
                                   diffraction_3d_sh_spectrum=sh_spectra)

        structure.info['descriptor'] = descriptor_data

        return structure

    def write(self, structure, tar=None, op_id=0, write_sh_spectra_npy=False, write_sh_spectra_png=True,
              write_geo=True, format_geometry='aims'):
        """

        Parameters:

        structure: class, ASE atoms class
            Instance of the class ASE atoms class

        format_geometry: string, optional (default='aims')
            File output format. All ASE valid output formats are accepted.
            For a list: https://wiki.fysik.dtu.dk/ase/ase/io/io.html

        """

        if not is_descriptor_consistent(structure, self):
            raise Exception('Descriptor not consistent. Aborting.')

        desc_folder = self.configs['io']['desc_folder']
        descriptor_info = structure.info['descriptor']['descriptor_info']

        sh_spectra = structure.info['descriptor']['diffraction_3d_sh_spectrum']

        if write_sh_spectra_npy:
            sh_spectra_filename_npy = os.path.abspath(os.path.normpath(os.path.join(desc_folder, structure.info[
                'label'] + '_op' + str(op_id) + self.desc_metadata.ix['diffraction_3d_sh_spectrum']['file_ending'])))
            np.save(sh_spectra_filename_npy, sh_spectra)
            structure.info['diff_3d_sh_spectrum_filename_npy'] = sh_spectra_filename_npy
            tar.add(structure.info['diff_3d_sh_spectrum_filename_npy'])

        if write_sh_spectra_png:
            sh_spectra_filename_png = os.path.abspath(os.path.normpath(os.path.join(desc_folder, structure.info[
                'label'] + '_op' + str(op_id) + self.desc_metadata.ix['diffraction_3d_sh_spectrum_image'][
                                                                                        'file_ending'])))

            plt.imsave(sh_spectra_filename_png, sh_spectra)
            structure.info['diff_3d_sh_spectrum_filename_png'] = sh_spectra_filename_png
            tar.add(structure.info['diff_3d_sh_spectrum_filename_png'])

        if write_geo:
            # to have the file accessible by the Beaker notebook image we need to put them
            # in a special folder ('/user/tmp')
            if self.configs['runtime']['isBeaker']:
                # only for Beaker Notebook
                coord_filename_in = os.path.abspath(os.path.normpath(os.path.join('/user/tmp/',
                                                                                  structure.info['label'] +
                                                                                  self.desc_metadata.ix[
                                                                                      'diffraction_3d_coordinates'][
                                                                                      'file_ending'])))
            else:
                coord_filename_in = os.path.abspath(os.path.normpath(os.path.join(desc_folder, structure.info['label'] +
                                                                                  self.desc_metadata.ix[
                                                                                      'diffraction_3d_coordinates'][
                                                                                      'file_ending'])))

            structure.write(coord_filename_in, format=format_geometry)
            structure.info['diff_3d_coord_filename_in'] = coord_filename_in
            tar.add(structure.info['diff_3d_coord_filename_in'])


def get_design_matrix(structures, method='flatten_images', nn_model=None, layer_name=None):
    """Starting from atomic structures calculate the design matrix for the three-dimensional diffraction fingerprint.

    The list of structures must contain the calculated :py:class:`ai4materials.descriptors.diffraction3d.Diffraction3D`.


    Parameters:

    structures: ``ase.Atoms`` object or list of ``ase.Atoms`` object
        Atomic structure or list of atomic structure.


    Return:

    np.ndarray, shape [n_samples, n_features]
        Returns the design matrix.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    images = []
    for idx_structure, structure in enumerate(structures):
        diffraction_3d_sh_spectrum = structure.info['descriptor']['diffraction_3d_sh_spectrum']
        images.append(diffraction_3d_sh_spectrum)

    images = np.asarray(images)
    images = np.reshape(images, (images.shape[0], -1, images.shape[1], images.shape[2]))

    if method == 'flatten_images':
        design_matrix = np.reshape(images, (images.shape[0], -1))
    elif method == 'nn_representation':
        if nn_model is not None:
            logger.info("Using the convolutional neural network filters as feature matrix.")
            logger.info("Layer name: {0}".format(layer_name))
            logger.debug(nn_model.summary())
            activations = np.asarray(get_activations(nn_model, images, print_shape_only=True, layer_name=layer_name))
            design_matrix = np.reshape(activations, (activations.shape[1], -1))
        else:
            raise ValueError("Please pass a valid Keras neural network model.")

    logger.info("Feature matrix shape: {0}".format(design_matrix.shape))

    return design_matrix


def plot_3d_volume(power_spect):
    """Generate a 3d plot given a numpy array with Mayavi.

    This function can be used to plot any three-dimensional field, passed as a np.array.
    It uses the `mayavi.tools.pipeline.volume` from Mayavi:
    http://docs.enthought.com/mayavi/mayavi/auto/mlab_pipeline_other_functions.html#volume
    In the plot it is assumed that the elements of the array are equally spaced.

    Parameters:

    power_spect: np.ndarray, shape [n_px, n_py, n_pz]
        Array containing a three-dimensional quantity (i.e. field).

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    try:
        from mayavi import mlab
    except ImportError:
        raise ImportError("Could not import Mayavi. Mayavi is required for 3d plotting.")

    mlab.figure(1, bgcolor=(0.5, 0.5, 0.5), size=(800, 800))
    mlab.options.offscreen = False
    mlab.clf()

    # remove nan and normalize the spectrum for plotting purposes only
    power_spect_plot = np.nan_to_num(power_spect)
    power_spect_plot_norm = (power_spect_plot - power_spect_plot.min()) / (
            power_spect_plot.max() - power_spect_plot.min())

    src = mlab.pipeline.scalar_field(power_spect_plot_norm)
    field_min = power_spect_plot_norm.min()
    field_max = power_spect_plot_norm.max()
    mlab.pipeline.volume(src, vmin=0., vmax=field_min + .5 * (field_max - field_min))
    mlab.colorbar(title='Field intensity', orientation='vertical')

    # insert plane parallel to axis passing through the origin
    mlab.pipeline.image_plane_widget(src, plane_orientation='x_axes', slice_index=power_spect_plot_norm.shape[0] / 2, )
    mlab.pipeline.image_plane_widget(src, plane_orientation='y_axes', slice_index=power_spect_plot_norm.shape[1] / 2, )
    mlab.pipeline.image_plane_widget(src, plane_orientation='z_axes', slice_index=power_spect_plot_norm.shape[2] / 2, )
    mlab.colorbar(title='Field intensity', orientation='vertical')

    mlab.show()

    mlab.close(all=True)


def plot_concentric_shells(vox_by_slices, base_folder, idx_slices=None, create_animation=False):
    """Plot the concentric shells for a given three-dimensional volumetric shape.

    The volumetric shape is the three-dimensional diffraction intensity, as calculated by
    :py:mod:`ai4materials.descriptors.diffraction3d.Diffraction3D`. To plot the concentric shells
    for different voxel np.ndarray shapes simply change ``x, y, z = np.mgrid[0:176:176j, 0:176:176j, 0:176:176j]``
    to your desire meshgrid.

    Parameters:

    vox_by_slices: np.ndarray, shape [n_slices, n_px, n_py, n_pz]
        4-dimensional array containing each concentric shell obtained from
        :py:mod:`ai4materials.descriptors.diffraction3d.Diffraction3D`.
        ``n_px``, ``n_py``, ``n_pz`` are given by the interpolation and the region of the space
        considered. In our case, ``n_slices=52``, ``n_px=n_py=n_pz=176``.

    base_folder: str
        Folder to save the figures generated. The figures are saved in a subfolder folder ``shells_png`` of
        ``base_folder``.

    idx_slices: list of int, optional (default=None)
        List of integers defining which concentric shells to plot.
        If `None`, all concentric shells are plotted.

    create_animation: bool, optional (default=True)
        If `True` create an animation containing all concentric shells.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """
    try:
        from mayavi import mlab
    except ImportError:
        raise ImportError("Could not import Mayavi. Mayavi is required for 3d plotting.")

    if idx_slices is None:
        idx_slices = range(1, vox_by_slices.shape[0], 1)

    # create folder for saving files
    shells_images_folder = os.path.join(base_folder, 'png_shells')
    if not os.path.exists(shells_images_folder):
        os.makedirs(shells_images_folder)

    filename_png_list = []
    x, y, z = np.mgrid[0:176:176j, 0:176:176j, 0:176:176j]

    mlab.clf()
    for idx_slice in idx_slices:
        mlab.options.offscreen = False

        filename_png = os.path.join(shells_images_folder, 'desc_slice_' + str(idx_slice) + '.png')
        filename_png_list.append(filename_png)

        scalars = vox_by_slices[idx_slice]
        c_of_mass = ndimage.measurements.center_of_mass(scalars)
        logger.info("Center of mass: {}".format(c_of_mass))
        logger.info("Max scalar field: {} for shell {}".format(scalars.max(), idx_slice))

        expansion = 0.0
        x_new = x * (1.0 + expansion * idx_slice)
        y_new = y * (1.0 + expansion * idx_slice)
        z_new = z * (1.0 + expansion * idx_slice)

        obj = mlab.contour3d(x_new, y_new, z_new, scalars, contours=50, opacity=.2)
        obj.scene.disable_render = True
        obj.scene.anti_aliasing_frames = 0
        mlab.colorbar(title='Field intensity', orientation='vertical')
        mlab.view()

        mlab.savefig(filename=filename_png)
        mlab.show()
        mlab.close(all=True)

    if create_animation:
        pass  # import imageio  # with imageio.get_writer('/home/ziletti/Documents/calc_xray/rot_inv_3d/png_slices/descriptor.gif', mode='I',  #                         fps=2) as writer:  #     for filename_png in filename_png_list:  #         image = imageio.imread(filename_png)  # writer.append_data(image)  # for filename_png in filename_png_list:  #  # os.remove(filename_png)


def plot_concentric_shells_spherical_coords(image_by_slices, base_folder, idx_slices=None):
    """Plot the concentric shells for a given three-dimensional volumetric shape.

    The volumetric shape is the three-dimensional diffraction intensity, as calculated by
    :py:mod:`ai4materials.descriptors.diffraction3d.Diffraction3D`.


    Parameters:

    image_by_slices: np.ndarray, shape [n_slices, theta_bins_fine, phi_bins_fine]
        Three-dimensional array containing each concentric shell obtained in spherical coordinate, as calculated by
        :py:mod:`ai4materials.descriptors.diffraction3d.Diffraction3D`
        ``n_slices``, ``theta_bins_fine``, ``phi_bins_fine`` are given by the interpolation and the region of the space
        considered. In our case, ``n_slices=52``, ``theta_bins_fine=256``, ``phi_bins_fine=512``, as defined in
        :py:mod:`ai4materials.descriptors.diffraction3d.Diffraction3D` in ``phi_bins_fine`` and ``theta_bins_fine``.

    base_folder: str
        Folder to save the figures generated. The figures are saved in a subfolder folder ``shells_png`` of
        ``base_folder``.

    idx_slices: list of int, optional (default=None)
        List of integers defining which concentric shells to plot.
        If `None`, all concentric shells - in spherical coordinates - are plotted.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    if idx_slices is None:
        idx_slices = range(image_by_slices.shape[0])

    # create folder for saving files
    shells_images_folder = os.path.join(base_folder, 'shells_png')
    if not os.path.exists(shells_images_folder):
        os.makedirs(shells_images_folder)

    filename_png_list = []
    for idx_slice in idx_slices:
        filename_png = os.path.join(shells_images_folder, 'desc_sph_coords_slice' + str(idx_slice) + '.png')
        filename_png_list.append(filename_png)

        logger.debug("Slide idx: {}".format(idx_slice))
        logger.debug("Image max: {}".format(image_by_slices[idx_slice].max()))

        coeffs = SHExpandDH(image_by_slices[idx_slice], sampling=2)

        coeffs_filtered = coeffs.copy()

        imgs = [MakeGridDH(coeffs_filtered[:, :, :], sampling=2), MakeGridDH(coeffs_filtered[:, :16, :], sampling=2),
                MakeGridDH(coeffs_filtered[:, :32, :], sampling=2), MakeGridDH(coeffs_filtered[:, :64, :], sampling=2)]

        fig, axes = plt.subplots(nrows=2, ncols=2)
        for idx_ax, ax in enumerate(axes.flat):
            im = ax.imshow(imgs[idx_ax], interpolation='none')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)

        plt.savefig(filename_png, dpi=100, format="png")
