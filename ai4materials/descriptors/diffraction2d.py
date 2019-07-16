#!/usr/bin/python
# coding=utf-8
# Copyright 2016-2019 Angelo Ziletti
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
from future.utils import iteritems

__author__ = "Angelo Ziletti"
__copyright__ = "Copyright 2016, The NOMAD Project"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "21/06/18"

import condor
import logging
from ai4materials.descriptors.base_descriptor import Descriptor
from ai4materials.descriptors.base_descriptor import is_descriptor_consistent
from ai4materials.utils.utils_crystals import rot_mat_x
from ai4materials.utils.utils_crystals import rot_mat_y
from ai4materials.utils.utils_crystals import rot_mat_z
from ai4materials.utils.utils_crystals import scale_structure
import numpy as np
import os
from PIL import Image

logger = logging.getLogger('ai4materials')


class Diffraction2D(Descriptor):
    """The two dimensional diffraction descriptor.

    This is the descriptor introduced in Ziletti et al., Nature Communications 9, 2775, (2018). The default values
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

    desc_angles: dict, optional (default={"r": [-45., 45.], "g": [-45., 45.], "b": [-45., 45.]})
        As described in Ziletti et al., each structure is rotated clockwise and counterclockwise about a given
        crystal axis, the diffraction pattern for each rotation is calculated, and then the two patterns
        are superimposed. This procedure is then repeated for all three crystal axes.
        The final result is represented as one RGB image for crystal structure, where each color channel shows
        the diffraction patterns obtained by rotating about a given axis
        (red (R) for x-axis, green (G) for y-axis, and blue (B) for z-axis).
        For example: \n
        `"r": [-45., 45.]` \n
        means that the diffraction pattern for the R channel is given by the superposition of the diffraction pattern
        obtained by rotating the structure around the first crystal axis clockwise and counterclockwise by 45 degrees.

    atoms_scaling: string, optional (default='avg_nn')
        Type of scaling used in the atom structure scaling. See :py:mod:`ai4materials.utils.utils_crystals.scale_structure`
        for more details.

    atoms_scaling_cutoffs: list of float, optional (default=[4.0, 5.0, 7.0, 9.0, 11.0, 12.0])
        List of cutoffs to be used in the determination of the lengthscale of the system to be used in
        :py:mod:`ai4materials.utils.utils_crystals.scale_structure`.

    use_mask: bool, optional (default=True)
        If `True`, a mask such that only (`mask_r_min` < points < `mask_r_max`) are kept in the calculated
        diffraction pattern. This is done because the central peak in diffraction is very intense
        but does not carry useful information regarding the atomic structure.

    mask_r_min: float, optional (default=5)
        The intensity of the points in the diffraction pattern with a radius smaller than `mask_r_min` are set to zero.

    mask_r_max: float, optional (default=30)
        The intensity of the points in the diffraction pattern with a radius larger than `mask_r_max` are set to zero.

    .. note::
        This descriptor was introduced in the following article:

        A. Ziletti, D. Kumar, M. Scheffler, and L. M. Ghiringhelli, "Insightful classification of crystal structures
        using deep learning", Nature Communications 9, 2775, (2018) [`Link to article <https://www.nature.com/articles/s41467-018-05169-6>`_]

        Please cite this manuscript if you find this descriptor useful in your work.

    .. seealso::
        To calculate the diffraction pattern we use the open source software Condor.
        See the Condor webpage for more details: http://condor.readthedocs.io/en/.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    def __init__(self, configs=None, param_source=None, param_detector=None, desc_angles=None, atoms_scaling='avg_nn',
                 atoms_scaling_cutoffs=None, use_mask=True, mask_r_min=None, mask_r_max=None):
        super(Diffraction2D, self).__init__(configs=configs)

        if param_source is None:
            param_source = {'wavelength': 5.0E-12, 'pulse_energy': 1E-6, 'focus_diameter': 1E-6}

        if param_detector is None:
            param_detector = {'distance': 0.1, 'pixel_size': 4E-4, 'nx': 64, 'ny': 64}

        if atoms_scaling is None:
            atoms_scaling = 'avg_distance_nn'

        if atoms_scaling_cutoffs is None:
            atoms_scaling_cutoffs = [4.0, 5.0, 7.0, 9.0, 11.0, 12.0]

        if desc_angles is None:
            desc_angles = {"r": [-45., 45.], "g": [-45., 45.], "b": [-45., 45.]}

        params = {}
        params.update({"param_source": param_source, "param_detector": param_detector, "use_mask": use_mask})

        self.atoms_scaling = atoms_scaling
        self.atoms_scaling_cutoffs = atoms_scaling_cutoffs
        self.n_px = param_detector["nx"]
        self.n_py = param_detector["ny"]
        self.param_source = param_source
        self.param_detector = param_detector
        self.use_mask = use_mask

        assert self.n_px == self.n_py, "Images needs to have equal width and height while we have {0} and {1}".format(
            self.n_px, self.n_py)

        # add derived parameters
        if mask_r_min is None:
            mask_r_min = (5. / 32.) * self.n_px

        if mask_r_max is None:
            mask_r_max = self.n_px / 2. - self.n_px / 32.

        rot_matrices = {}
        rot_matrices_x = []
        for desc_angle in desc_angles["r"]:
            rot_matrices_x.append(rot_mat_x(desc_angle))
        rot_matrices["r"] = rot_matrices_x

        rot_matrices_y = []
        for desc_angle in desc_angles["g"]:
            rot_matrices_y.append(rot_mat_y(desc_angle))
        rot_matrices["g"] = rot_matrices_y

        rot_matrices_z = []
        for desc_angle in desc_angles["b"]:
            rot_matrices_z.append(rot_mat_z(desc_angle))
        rot_matrices["b"] = rot_matrices_z

        self.mask_r_min = mask_r_min
        self.mask_r_max = mask_r_max
        self.rot_matrices = rot_matrices

    def calculate(self, structure, **kwargs):
        """Calculate the descriptor for the given ASE structure.

        Parameters:

        structure: `ase.Atoms` object
            Atomic structure.

        """

        atoms = scale_structure(structure, scaling_type=self.atoms_scaling,
                                atoms_scaling_cutoffs=self.atoms_scaling_cutoffs)

        # Source
        src = condor.Source(**self.param_source)

        # Detector
        det = condor.Detector(**self.param_detector)

        # Atoms
        atomic_numbers = map(lambda el: el.number, atoms)
        atomic_numbers = [atomic_number + 2 for atomic_number in atomic_numbers]

        # convert Angstrom to m (CONDOR uses meters)
        atomic_positions = list(map(lambda pos: [pos.x * 1E-10, pos.y * 1E-10, pos.z * 1E-10], atoms))

        intensity_rgb = []
        rs_rgb = []
        ph_rgb = []
        real_space = None
        phases = None

        for [_, rot_matrices] in iteritems(self.rot_matrices):
            # loop over channels
            intensity_channel = []
            rs_channel = []
            ph_channel = []
            for rot_matrix in rot_matrices:
                # loop over the rotation matrices in a given channel
                # and sum the intensities in the same channel
                quaternion = condor.utils.rotation.quat_from_rotmx(rot_matrix)
                rotation_values = np.array([quaternion])
                rotation_formalism = 'quaternion'
                rotation_mode = 'intrinsic'

                par = condor.ParticleAtoms(atomic_numbers=atomic_numbers, atomic_positions=atomic_positions,
                                           rotation_values=rotation_values, rotation_formalism=rotation_formalism,
                                           rotation_mode=rotation_mode)

                s = 'particle_atoms'
                condor_exp = condor.Experiment(src, {s: par}, det)
                res = condor_exp.propagate()

                # retrieve results
                real_space = np.fft.fftshift(np.fft.ifftn(np.fft.fftshift(res["entry_1"]["data_1"]["data_fourier"])))
                intensity = res["entry_1"]["data_1"]["data"]
                fourier_space = res["entry_1"]["data_1"]["data_fourier"]
                phases = np.angle(fourier_space) % (2 * np.pi)

                if self.use_mask:
                    # set to zero values outside a ring-like mask
                    xc = (self.n_px - 1.0) / 2.0
                    yc = (self.n_py - 1.0) / 2.0
                    n = self.n_px
                    a, b = xc, yc
                    x, y = np.ogrid[-a:n - a, -b:n - b]

                    mask_int = x * x + y * y <= self.mask_r_min * self.mask_r_min
                    mask_ext = x * x + y * y >= self.mask_r_max * self.mask_r_max

                    for i in range(self.n_px):
                        for j in range(self.n_py):
                            if mask_int[i, j]:
                                intensity[i, j] = 0.0
                            if mask_ext[i, j]:
                                intensity[i, j] = 0.0

                intensity_channel.append(intensity)
                rs_channel.append(real_space.real)
                ph_channel.append(phases)

            # first sum the angles within the channel, then normalize
            intensity_channel = np.asarray(intensity_channel).sum(axis=0)
            rs_channel = np.asarray(rs_channel).sum(axis=0)
            ph_channel = np.asarray(ph_channel).sum(axis=0)

            # append normalized data from different channels
            # and divide by the angles per channel
            intensity_rgb.append(intensity_channel)
            rs_rgb.append(rs_channel)
            ph_rgb.append(ph_channel)

        intensity_rgb = np.asarray(intensity_rgb)
        intensity_rgb = (intensity_rgb - intensity_rgb.min()) / (intensity_rgb.max() - intensity_rgb.min())

        rs8 = (((real_space.real - real_space.real.min()) / (
                real_space.real.max() - real_space.real.min())) * 255.0).astype(np.uint8)
        ph8 = (((phases - phases.min()) / (phases.max() - phases.min())) * 255.0).astype(np.uint8)

        # reshape to have nb of color channels last
        intensity_rgb = intensity_rgb.reshape(intensity_rgb.shape[1], intensity_rgb.shape[2], intensity_rgb.shape[0])

        # add results in ASE structure info
        descriptor_data = dict(descriptor_name=self.name, descriptor_info=str(self),
                               diffraction_2d_intensity=intensity_rgb, diffraction_2d_real_space=rs8,
                               diffraction_2d_phase=ph8)

        structure.info['descriptor'] = descriptor_data

        return structure

    def write(self, structure, tar, op_id=0, write_intensity_npy=True, write_intensity_png=True, write_rspace_png=False,
              write_phase_png=False, write_geo=True, format_geometry='aims'):
        """Write the descriptor to file.

        Parameters:

        structure: `ase.Atoms` obejct
            Atomic structure.

        tar: TarFile object
            TarFile archive where the descriptor is added. This is created internally with `tarfile.open`.

        op_id: int, optional (default=0)
            Number of the applied operation to the descriptor. At present always set to zero in the code.

        write_intensity_png: bool, optional (default=`True`)
            If `True`, write to file diffraction intensities as numpy arrays.

        write_rspace_png: bool, optional (default=`False`)
            If `True`, write to file a png file with the real space structure for which the diffraction pattern
            is calculated.

        write_phase_png: bool, optional (default=`False`)
            If `True`, write to file a png file with the phases of the diffraction pattern.

        write_geo: bool, optional (default=`True`)
            If `True`, write a coordinate file of the structure for which the diffraction pattern is calculated.

        format_geometry: string, optional (default=`aims`)
            Output format of the geometry file. All ASE valid output formats are accepted.
            For a complete list see: https://wiki.fysik.dtu.dk/ase/ase/io/io.html


        """

        if not is_descriptor_consistent(structure, self):
            raise Exception('Descriptor not consistent. Aborting.')

        desc_folder = self.configs['io']['desc_folder']
        descriptor_info = structure.info['descriptor']['descriptor_info']

        intensity_rgb = structure.info['descriptor']['diffraction_2d_intensity']

        if write_intensity_npy:
            intensity_filename_npy = os.path.abspath(os.path.normpath(os.path.join(desc_folder,
                                                                                   structure.info['label'] +
                                                                                   self.desc_metadata.ix[
                                                                                       'diffraction_2d_intensity'][
                                                                                       'file_ending'])))
            np.save(intensity_filename_npy, intensity_rgb)
            structure.info['diff_2d_intensity_filename_npy'] = intensity_filename_npy
            tar.add(structure.info['diff_2d_intensity_filename_npy'])

        if write_intensity_png:
            intensity_filename_png = os.path.abspath(os.path.normpath(os.path.join(desc_folder,
                                                                                   structure.info['label'] +
                                                                                   self.desc_metadata.ix[
                                                                                       'diffraction_2d_intensity_image'][
                                                                                       'file_ending'])))
            rgb_array = np.zeros((intensity_rgb.shape[0], intensity_rgb.shape[1], intensity_rgb.shape[2]), 'uint8')

            current_img = list(intensity_rgb.reshape(-1, intensity_rgb.shape[0], intensity_rgb.shape[1]))

            for ix_ch in range(len(current_img)):
                rgb_array[..., ix_ch] = current_img[ix_ch] * 255

            img = Image.fromarray(rgb_array)
            img.save(intensity_filename_png)
            structure.info['diff_2d_intensity_filename_png'] = intensity_filename_png
            tar.add(structure.info['diff_2d_intensity_filename_png'])

        if write_rspace_png:
            real_space = structure.info['descriptor']['diffraction_2d_real_space']
            real_space_filename_png = os.path.abspath(os.path.normpath(os.path.join(desc_folder,
                                                                                    structure.info['label'] +
                                                                                    self.desc_metadata.ix['real_space'][
                                                                                        'file_ending'])))
            img = Image.fromarray(real_space)
            img.save(real_space_filename_png)
            structure.info['diff_2d_real_space_filename_png'] = real_space_filename_png
            tar.add(structure.info['diff_2d_real_space_filename_png'])

        if write_phase_png:
            phase = structure.info['descriptor']['diffraction_2d_phase']
            phase_filename_png = os.path.abspath(os.path.normpath(os.path.join(desc_folder, structure.info['label'] +
                                                                               self.desc_metadata.ix[
                                                                                   'diffraction_2d_phase'][
                                                                                   'file_ending'])))
            img = Image.fromarray(phase)
            img.save(phase_filename_png)
            structure.info['diffraction_2d_phase'] = phase_filename_png
            tar.add(structure.info['diffraction_2d_phase'])

        if write_geo:
            # to have the file accessible by the Beaker notebook image we need to put them
            # in a special folder ('/user/tmp')
            if self.configs['runtime']['isBeaker']:
                # only for Beaker Notebook
                coord_filename_in = os.path.abspath(os.path.normpath(os.path.join('/user/tmp/',
                                                                                  structure.info['label'] +
                                                                                  self.desc_metadata.ix['coordinates'][
                                                                                      'file_ending'])))
            else:
                coord_filename_in = os.path.abspath(os.path.normpath(os.path.join(desc_folder, structure.info['label'] +
                                                                                  self.desc_metadata.ix[
                                                                                      'diffraction_2d_coordinates'][
                                                                                      'file_ending'])))

            structure.write(coord_filename_in, format=format_geometry)
            structure.info['diff_2d_coord_filename_in'] = coord_filename_in
            tar.add(structure.info['diff_2d_coord_filename_in'])
