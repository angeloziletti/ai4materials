# coding=utf-8
# Copyright 2016-2018 Emre Ahmetick
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

__author__ = "Emre Ahmetick"
__copyright__ = "Copyright 2018, Emre Ahmetick"
__maintainer__ = "Emre Ahmetick"
__email__ = "ahmetick@fhi-berlin.mpg.de"
__date__ = "23/09/18"

import numpy as np
from ai4materials.utils.utils_config import SSH
import os
import sched
import time
import sys
import logging
from shutil import rmtree
import pandas as pd
from subprocess import Popen
import operator as opop
from copy import deepcopy
from functools import reduce


F_unit = [
    ['IP(A)', 'IP(B)', 'EA(A)', 'EA(B)'],
    ['E_HOMO(A)', 'E_HOMO(B)', 'E_LUMO(A)', 'E_LUMO(B)'],
    ['r_s(A)', 'r_s(B)', 'r_p(A)', 'r_p(B)', 'r_d(A)', 'r_d(B)', 'r_sigma(AB)', 'r_pi(AB)'],
    ['Z(A)', 'Z(B)', 'Z_val(A)', 'Z_val(B)', 'period(A)', 'period(B)'],
    ['d(AB)', 'd(A)', 'd(B)'],
    ['E_b(AB)', 'E_b(A)', 'E_b(B)'],
    ['HL_gap(AB)', 'HL_gap(A)', 'HL_gap(B)'],
]

reals = [
    'IP(A)',
    'IP(B)',
    'EA(A)',
    'EA(B)',
    'E_HOMO(A)',
    'E_HOMO(B)',
    'E_LUMO(A)',
    'E_LUMO(B)',
    'r_s(A)',
    'r_s(B)',
    'r_p(A)',
    'r_p(B)',
    'r_d(A)',
    'r_d(B)',
    'd(AB)',
    'd(A)',
    'd(B)',
    'E_b(AB)',
    'E_b(A)',
    'E_b(B)',
    'HL_gap(AB)',
    'HL_gap(A)',
    'HL_gap(B)',
    'r_sigma(AB)',
    'r_pi(AB)']
ints = ['Z(A)', 'Z(B)', 'Z_val(A)', 'Z_val(B)', 'period(A)', 'period(B)']

standard_format = [
    'IP(A)',
    'IP(B)',
    'EA(A)',
    'EA(B)',
    'E_HOMO(A)',
    'E_HOMO(B)',
    'E_LUMO(A)',
    'E_LUMO(B)',
    'r_s(A)',
    'r_s(B)',
    'r_p(A)',
    'r_p(B)',
    'r_d(A)',
    'r_d(B)',
    'd(AB)',
    'd(A)',
    'd(B)',
    'Z(A)',
    'Z(B)',
    'Z_val(A)',
    'Z_val(B)',
    'E_b(AB)',
    'E_b(A)',
    'E_b(B)',
    'HL_gap(AB)',
    'HL_gap(A)',
    'HL_gap(B)',
    'r_sigma(AB)',
    'r_pi(AB)',
    'period(A)',
    'period(B)']
converted_format = [
    'ipA',
    'ipB',
    'eaA',
    'eaB',
    'homoA',
    'homoB',
    'lumoA',
    'lumoB',
    'rsA',
    'rsB',
    'rpA',
    'rpB',
    'rdA',
    'rdB',
    'disAB',
    'disA',
    'disB',
    'zA',
    'zB',
    'valA',
    'valB',
    'ebAB',
    'ebA',
    'ebB',
    'hlgapAB',
    'hlgapA',
    'hlgapB',
    'rsigmaAB',
    'rpiAB',
    'periodA',
    'periodB']

standard_2_converted = dict(zip(standard_format, converted_format))
converted_2_standard = dict(zip(converted_format, standard_format))


""" Set logger for outputs as errors, warnings, infos. """

#
# try:
#     hdlr = logging.FileHandler(configs["output_file"], mode='a')
# except:
#     hdlr = logging.FileHandler(configs["output_file"], mode='w')
#
# level = logging.getLevelName(configs["log_level_general"])
#
# logger = logging.getLogger(__name__)
# logger.setLevel(level)
# logging.basicConfig(level=level)
# FORMAT = "%(levelname)s: %(message)s"
# formatter = logging.Formatter(fmt=FORMAT)
# handler = logging.StreamHandler()
# handler.setFormatter(formatter)
# hdlr.setFormatter(formatter)
# logger.addHandler(handler)
# logger.addHandler(hdlr)
# logger.setLevel(level)
# logger.propagate = False
#
# __metainfopath__ = configs["meta_info_file"]

# START PARAMETERS REFERENCE


# In the following lists of tuples the order of the items might be important. Thus no dict is used.
# If value is tuple, then only one of items are possible as value when passing the dict control to the SIS class.
Tuple_list = [
    # FCDI
    ('mpiname', str),    # code will be run by: mpiname codename. set mpiname='' for serial run.
    ('desc_dim', int),               # starting iteration (can be n if iteration up to n-1 already calculated before)
    ('ptype', ('quanti', 'quali')),      # property type: 'quanti'(quantitative),'quali'(qualitative)
    ('ntask', int),      # number of tasks (properties)
    ('nsample', list),   # number of samples for each task (and group for classification, e.g. (4,3,5),(7,9) )
    ('width', float),     # for classification, the boundary tolerance
    # FC
    ('nsf', int),  # number of scalar features (i.e.: the atomic parameters)
    ('task_arr', int),  # number of tasks arranged in columns
    ('rung', int),  # rung of feature spaces (rounds of combination)
    ('opset', list),  # oprators(currently: (+)(-)(*)(/)(exp)(log)(^-1)(^2)(^3)(sqrt)(|-|) )
    ('ndimtype', int),  # number of dimension types (for dimension analysis)
    ('dimclass', list),   # specify features in each class denoted by ( )
    ('allele', bool),  # Should all elements appear in each of the selected features?
    ('nele', int),  # number of element (<=6): useful only when symm=.true. and/or allele=.true.
    ('maxfval_lb', float),  # features having the max. abs. data value <maxfval_lb will not be selected
    ('maxfval_ub', float),  # features having the max. abs. data value >maxfval_ub will not be selected
    ('subs_sis', int),  # total number of features selected by sure independent screen
    # DI
    ('method', ('L1L0', 'L0')),  # 'L1L0' or 'L0'
    ('size_fs', int),  # number of total features in each taskxxx.dat (same for all)
    ('nfL0', int),  # number of features for L0(ntotf->nfL0 if nfL0>ntotf)
    ('metric', ('LS_RMSE', 'CV_RMSE', 'CV_MAE')),        # metric for the evaluation: LS_RMSE,CV_RMSE,CV_MAE
    ('n_eval', int),                 # number of top models (based on fitting) to be evaluated by the metric
    ('CV_fold', int),  # k-fold CV (>=2)
    ('CV_repeat', int),  # repeated k-fold CV
    ('n_out', int),         # number of top models to be output, off when =0
]

# Generate lists and dics for easier coding later.
Param_key_list = [i for i, j in Tuple_list]
Param_dic = dict(Tuple_list)

# Important: control reference. Specifies how the structure of input control dict to SIS class should look like.
# If key tuple, then value has to be tuple, too. A tuple stands for the option that on and only one of the keys
# have to set.
control_ref = {
    'local_paths': {'local_path': str, 'SIS_input_folder_name': str},
    ('local_run', 'remote_run'): (
        {'SIS_code_path': str, 'mpi_command': str},
        {'SIS_code_path': str, 'username': str, 'hostname': str, 'port': int, 'remote_path': str,
            'eos': bool, 'mpi_command': str, 'nodes': int, ('key_file', 'password'): (str, str)}
    ),
    'parameters': {'rung': int, 'subs_sis': int, 'desc_dim': int, 'opset': list, 'ptype': ('quanti', 'quali')},
    'advanced_parameters': Param_dic
}

# All keys which do not need to be set in input control dict tree. If they are not set, default values are used.
not_mandotary = ['advanced_parameters', 'eos', 'nodes', 'port', 'FC', 'DI', 'FCDI'] + Param_key_list

# Availabel OPs for the SIS fortran code, at the moment.
available_OPs = ['+', '-', '*', '/', 'exp', 'exp-', '^-1', '^2', '^3', 'sqrt', 'log', '|-|', 'SCD', '^6']

un_OP = ['exp', '^2', 'exp-', '^-1', '^2', '^3', 'sqrt', 'log', 'SCD', '^6']
bin_OP = ['-', '/']
bin_OP_bino = ['+', '|-|', '*']
# END PARAMETERS REFERENCE


class SIS(object):
    """ Python interface with the fortran SIS+(Sure Independent Screening)+L0/L1L0 code.

    The SIS+(Sure Independent Screening)+L0/L1L0 is a greedy algorithm. It enhances the OMP, by considering
    not only the closest feature vector to the residual in each step, but collects the closest 'n_SIS' features vectors.
    The final model is then built after a given number of iterations by determining the (approximately) best linear combination
    of the collected features using the L0 (L1-L0) algorithm.

    To execute the code, besides the SIS code parameters also folder paths are needed as well as account
    information of a remote machine to let the code be executed on it.

    Parameters
    ----------
    P : array, [n_sample]; list; [n_sample]
        P refers to the target (label). If ptype = 'quali' list of ints is required

    D : array, [n_sample, n_features]
        D refers to the feature matrix. The SIS code calculates algebraic combinations
        of the features and then applies the SIS+L0/L1L0 algorithm.

    feature_list : list of strings
        List of feature names. Needs to be in the same order as the feature vectors (columns) in D.
        Features must consist of strings which are in F_unit (See above).

    feature_unit_classes : None or {list integers or the string: 'no_unit'}
        integers correspond to the unit class of the features from feature_list. 'no_unit' is reserved for
        dimensionless unit.

    output_log_file : string
        file path for the logger output.

    rm_existing_files : bool
        If SIS_input_path on local or remote machine (remote_input_path) exists, it is removed.
        Otherwise it is renamed to SIS_input_path_$number.

    control : dict of dicts (of dicts)
        Dict tree: {
            'local_paths': { 'local_path':str, 'SIS_input_folder_name':str},
            ('local_run','remote_run')  : (
                {'SIS_code_path':str, 'mpi_command':str},
                {'SIS_code_path':str, 'username':str, 'hostname':str, 'remote_path':str, 'eos':bool, 'mpi_command':str, 'nodes':int, ('key_file', 'password'):(str,str)}
            ),
            'parameters' : {'n_comb':int, 'n_sis':int, 'max_dim':int, 'OP_list':list},
            'advanced_parameters' : {'FC':FC_dic,'DI':DI_dic, 'FCDI':FCDI_dic}
        }
        Here the tuples (.,.) mean that one and only one of the both keys has to be set.
        To see forms of FC_dic, DI_dic, FCDI_dic check FC_tuplelist, DI_tuplelist and FCDI_tuplelist above in PARAMETERS REFERENCE.



    Attributes
    ----------
    start : -
        starts the code

    get_results :  list [max_dim] of dicts {'D', 'coefficients', 'P_pred'}
        get_results[model_dim-1]['D'] : pandas data frame [n_sample, model_dim+1]
            Descriptor matrix with the columns being algebraic combinations of the input feature matrix.
            Column names are thus strings of the algebraic combinations of strings of inout feature_list.
            Last column is full of ones corresponding to the intercept

        get_results[model_dim-1]['coefficients'] : array [model_dim+1]
            Optimizing coefficients.

        get_results[model_dim-1]['P_pred'] : array [m_sample]
            Fit : np.dot( np.array(D), coefficients)

    Notes
    -----
    For remote_run the library nomad_sim.ssh_code is needed. If remote machine is eos,
    in dict control['remote_run'] the (key:value) 'eos':True has to be set. Then set
    for example in addition 'nodes':1 and 'mpi_run -np 32' can be set.

    Paths (say name: path) are all set in the intialization part with self.path and
    used in other functions with self.path. In general the other variables are directly
    passed as arguements to the functions. There are a few exceptions as self.ssh.

    Examples
    --------
    # >>> import numpy as np
    # >>> from nomad_sim.SIS import SIS
    # >>> ### Specify where on local machine input files for the SIS fortran code shall be created
    # >>> Local_paths = {
    # >>> 'local_path' : '/home/beaker/',
    # >>> 'SIS_input_folder_name' : 'SIS_input',
    # >>> }
    # >>> # Information for ssh connection. Instead of password also 'key_file' for rsa key
    # >>> # file path is possible.
    # >>> Remote_run = {
    # >>>     'mpi_command':'',
    # >>>     'remote_path' : '/home/username/',
    # >>>     'SIS_code_path' : '/home/username/SIS_code/',
    # >>>     'hostname' :'hostname',
    # >>>     'username' : 'username',
    # >>>     'password' : 'XXX'
    # >>> }
    # >>> # Parameters for the SIS fortran code. If at each iteration a different 'OP_list'
    # >>> # shall be used, set a list of max_dim lists, e.g. [ ['+','-','*'], ['/','*'] ], if
    # >>> # n_comb = 2
    # >>> Parameters = {
    # >>>     'n_comb' : 2,
    # >>>     'OP_list' : ['+','|-|','-','*','/','exp','^2'],
    # >>>     'max_dim' : 2,
    # >>>     'n_sis' : 10
    # >>> }
    # >>> # Final control dict for the SIS class. Instead of remote_run also local_run can be set
    # >>> # (with different keys). Also advanced_parameters can be set, but should be done only
    # >>> # if the parameters of the SIS fortran code are understood.
    # >>> SIS_control = {'local_paths':Local_paths, 'remote_run':Remote_run, 'parameters':Parameters}
    # >>> # Target (label) vector P , feature_list, feature matrix D. The values are made up.
    # >>> P = np.array( [1,2,3,-2,-9] )
    # >>> feature_list=['r_p(A)','r_p(B)', 'Z(A)']
    # >>> D = np.array([[7,-11,3],
    # >>>              [-1,-2,4],
    # >>>              [2,20,3],
    # >>>              [8,1,8],
    # >>>              [-3,4,1]])
    # >>> # Use the code
    # >>> sis = SIS(P,D,feature_list, control = SIS_control, output_log_file ='/home/ahmetcik/codes/beaker/output.log')
    # >>> sis.start()
    # >>> results = sis.get_results()
    # >>>
    # >>> coef_1dim = results[0]['coefficients']
    # >>> coef_2dim = results[1]['coefficients']
    # >>> D_1dim = results[0]['D']
    # >>> D_2dim = results[1]['D']
    # >>> print coef_2dim
    # [-3.1514 -5.9171  3.9697]
    # >>>
    # >>> print D_2dim
    #    ((rp(B)/Z(A))/(rp(A)+rp(B)))  ((Z(A)/rp(B))/(rp(B)*Z(A)))  intercept
    # 0                      0.916670                     0.008264        1.0
    # 1                      0.166670                     0.250000        1.0
    # 2                      0.303030                     0.002500        1.0
    # 3                      0.013889                     1.000000        1.0
    # 4                      4.000000                     0.062500        1.0
    #
    # """

# START INIT
    def __init__(self, P, D, feature_list, feature_unit_classes=None, target_unit='eV', control=None,
                 output_log_file='/home/beaker/.beaker/v1/web/tmp/output.log', rm_existing_files=False, if_print=True, check_only_control=False):

        control = deepcopy(control)
        self.rm_existing_files = rm_existing_files
        self.target_unit = target_unit
     #   set_logger(output_log_file)
        self.logger = logger
        self.if_print = if_print

        # Check inputs
        self.check_arrays(P, D, feature_list, feature_unit_classes, control['parameters']['ptype'])
        self.check_control(control, control_ref, "control")
        self.check_quali_dim(control)
        self.check_OP_list(control)
        if check_only_control:
            return

        # Distribute the control keys to the corresponding init functions.
        self.set_main_settings(P, D, feature_list, feature_unit_classes, **control['local_paths'])
        if 'remote_run' in control:
            self.set_ssh_connection(**control['remote_run'])
        else:
            self.set_local_run(**control['local_run'])

        if 'advanced_parameters' in control:
            advanced_parameters = control['advanced_parameters']
        else:
            advanced_parameters = None
        self.set_SIS_parameters(advanced_parameters=advanced_parameters, **control['parameters'])

        self.predicted_feature_space_size = None
        self.l0_steps = None

        self.checking_expense = True
        self.if_print = False
        self.if_close_ssh = False
        self.estimate_calculation_expense(feature_list)
        self.checking_expense = False
        self.if_print = if_print
        if control['parameters']['ptype'] == 'quanti':
            self.if_close_ssh = True

    def set_main_settings(self, P, D, feature_list, feature_unit_classes,
                          local_path='/home/beaker/', SIS_input_folder_name='input_folder'):
        """ Set local environment and P, D and feature_list."""

        self.local_path = local_path
        self.SIS_input_folder_name = SIS_input_folder_name
        self.SIS_input_path = os.path.join(self.local_path, SIS_input_folder_name)

        if feature_unit_classes is None:
            feature_unit_classes = [0 for _ in feature_list]

        # Bring feature_list and D in the feature_order of F_unit becauese self.check_feature_untis needs it.
        ordered_indices = np.argsort(feature_unit_classes)
        self.feature_unit_classes = [feature_unit_classes[i] for i in ordered_indices]
        self.feature_list = [feature_list[i] for i in ordered_indices]
        self.D = D[:, ordered_indices]

        self.P = P
        self.ssh_connection = False
        self.local_run = False

    def set_local_run(self, SIS_code_path='~/codes/SIS_code/', mpi_command=''):
        """ Set and check local enviroment if local_run is used."""
        self.local_run = True

        self.SIS_code_path = SIS_code_path
        self.SIS_code_FCDI = os.path.join(self.SIS_code_path, 'FCDI')
        self.mpi_command = mpi_command

        # Check if SIS_code_path exists and if the SIS codes FC, DI and FCDI exist in it.
        if os.path.isdir(self.SIS_code_path):
            for program in ['FCDI', 'FC', 'DI']:
                program_path = os.path.join(self.SIS_code_path, program)
                if not os.path.exists(program_path):
                    raise OSError("No executable: %s" % program_path)
        else:
            raise OSError("No such directory: %s" % self.SIS_code_path)

    def set_ssh_connection(self, hostname=None, username=None, port=22, key_file=None, password=None,
                           remote_path=None, SIS_code_path=None, eos=False, nodes=1, mpi_command=''):
        """ Set ssh connection. Set and check remote enviroment if remote_run is used."""
        self.ssh_connection = True
        # weather close ssh connection at the end of do_transfer
        self.if_close_ssh = True

        self.remote_path = remote_path
        self.SIS_code_path = SIS_code_path
        self.SIS_code_FCDI = os.path.join(self.SIS_code_path, 'FCDI')
        self.remote_input_path = os.path.join(self.remote_path, self.SIS_input_folder_name)
        self.username = username
        self.mpi_command = mpi_command
        self.eos = eos

        key_file = self.check_(key_file)

        # set ssh connection
        try:
            self.ssh = SSH(hostname=hostname, username=self.username, port=port, key_file=key_file, password=password)
            os.remove(key_file)
        except Exception as e:
            os.remove(key_file)
            self.logger.error('ssh connection failed. The error message:\n%s' % e)
            sys.exit(1)

        # set number of CPUs for job submission script.
        if eos:
            self.CPUs = nodes * 32
        else:
            # Further remote machines... Now only eos
            self.CPUs = None

        # check paths on remote machine
        # Check if SIS_code_path exists and if the SIS codes FC, DI and FCDI exist in it.
        if self.ssh.isdir(self.SIS_code_path):
            for program in ['FCDI', 'FC', 'DI']:
                program_path = os.path.join(self.SIS_code_path, program)
                if not self.ssh.exists(program_path):
                    raise OSError("No such executable on remote machine: %s" % program_path)
        else:
            raise OSError("No such directory on remote machine: %s" % self.SIS_code_path)

        if not self.ssh.isdir(self.remote_path):
            raise OSError("No such directory on remote machine: %s" % self.remote_path)

    def set_SIS_parameters(self, desc_dim=2, subs_sis=100, rung=1, opset=[
                           '+', '-', '/', '^2', 'exp'], ptype='quanti', advanced_parameters=None):
        """ Set the SIS fortran code parameters

        If advanced parameters is passed, they will be used, otherwise default values will be used.
        Also max_dim, n_sis, n_comb, and OP_list can be overwritten by advanced_parameters if specified.

        """

        # Get units. It is a list of strings, e.g. ['(1:4)','(5:8)',...], specifiying which columns/features of D
        # belong to a unit class. Index starts with 1. The columns/features were ordered in self.set_main_settings
        # such that columns/features of same unit are next to each other.
        units_list = self.check_feature_units(self.feature_unit_classes)
        ndimtype = len(units_list)
        nsf = len(self.feature_list)

        # self.set_par will use it
        self.advanced_parameters = advanced_parameters

        # Get shape of P
        if ptype == 'quanti':
            row_lengths = len(self.P)
        else:
            index = np.unique(self.P, return_index=True)[1]
            class_names = [self.P[i] for i in np.sort(index)]
            row_lengths = tuple([len([None for p in self.P if p == current_class]) for current_class in class_names])

        # initilize SIS parameters: self.parameters
        self.parameters = dict.fromkeys(Param_key_list)
        # set parameters
        # FCDI
        # code will be run by: mpiname codename. set mpiname='' for serial run.
        self.parameters['mpiname'] = self.mpi_command
        self.parameters['desc_dim'] = desc_dim           # ending iteration
        self.parameters['ptype'] = ptype              # property type: 'quanti'(quantitative),'quali'(qualitative)
        self.parameters['ntask'] = 1                # number of tasks (properties)
        # number of samples for each task (and group for classification, e.g. (4,3,5),(7,9) )
        self.parameters['nsample'] = row_lengths
        self.parameters['width'] = 0.01                 # for classification, the boundary tolerance
        # FC
        self.parameters['nsf'] = nsf  # number of scalar features (i.e.: the atomic parameters)
        self.parameters['task_arr'] = '1c'  # number of tasks arranged in columns
        self.parameters['rung'] = rung  # rung of feature spaces (rounds of combination)
        self.parameters['opset'] = opset  # oprators(currently: (+)(-)(*)(/)(exp)(log)(^-1)(^2)(^3)(sqrt)(|-|) )
        self.parameters['ndimtype'] = ndimtype  # number of dimension types (for dimension analysis)
        self.parameters['dimclass'] = units_list        # specify features in each class denoted by ( )
        self.parameters['allele'] = False  # Should all elements appear in each of the selected features?
        self.parameters['nele'] = 0  # number of element (<=6): useful only when symm=.true. and/or allele=.true.
        # features having the max. abs. data value <maxfval_lb will not be selected
        self.parameters['maxfval_lb'] = 1e-8
        # features having the max. abs. data value >maxfval_ub will not be selected
        self.parameters['maxfval_ub'] = 1e5
        self.parameters['subs_sis'] = subs_sis  # total number of features selected by sure independent screen
        # DI
        self.parameters['method'] = 'L0'  # 'L1L0' or 'L0'
        self.parameters['size_fs'] = ''  # number of total features in each taskxxx.dat (same for all)
        self.parameters['nfL0'] = ''  # number of features for L0(ntotf->nfL0 if nfL0>ntotf)
        self.parameters['metric'] = 'LS_RMSE'           # metric for the evaluation: LS_RMSE,CV_RMSE,CV_MAE
        # number of top models (based on fitting) to be evaluated by the metric
        self.parameters['n_eval'] = 1000
        self.parameters['CV_fold'] = 10  # k-fold CV (>=2)
        self.parameters['CV_repeat'] = 1  # repeated k-fold CV
        self.parameters['n_out'] = 100                  # number of top models to be output, off when =0

        # overwrite parameter values if specified in advanced_parameters
        if not advanced_parameters is None:
            for key, value in advanced_parameters.iteritems():
                self.parameters[key] = value

# END INIT

    def start(self):
        """ Attribute which starts the calculations after init. """
        # Check if folders exists. If yes delete (if self.rm_existing_files)
        # or rename it to self.SIS_input_path_old_#
        if os.path.isdir(self.SIS_input_path):
            self.logger.warning('Directory %s already exists.' % self.SIS_input_path)
            if self.rm_existing_files:
                rmtree(self.SIS_input_path)
                self.logger.warning('It is removed.')
            else:
                for i in range(1000):
                    old_name = "%s_old_%s" % (self.SIS_input_path, i)
                    if not os.path.isdir(old_name):
                        os.rename(self.SIS_input_path, old_name)
                        break
                self.logger.warning('It is renamed to %s.' % old_name)
        # creat input folder on local machine
        os.mkdir(self.SIS_input_path)

        # write input files in inputfolder
        self.write_P_D(self.P, self.D, self.feature_list)
        self.write_parameters()

        # decide if calculation on local or remote machine
        if self.ssh_connection:
            self.do_transfer(ssh=self.ssh, eos=self.eos, username=self.username, CPUs=self.CPUs)
        else:
            # calculate on local machine. (At the moment not clear if python blocks parallel computing)
            os.chdir(self.SIS_input_path)
            Popen(self.SIS_code_FCDI).wait()

    def set_logger(self, output_log_file):
        """ Set logger for outputs as errors, warnings, infos. """
        self.logger = logging.getLogger(__name__)

        hdlr = logging.FileHandler(output_log_file)

        self.logger.setLevel(logging.INFO)
        logging.basicConfig(level=logging.INFO)
        FORMAT = "%(levelname)s: %(message)s"
        formatter = logging.Formatter(fmt=FORMAT)
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        hdlr.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.addHandler(hdlr)
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False


# START ckecking functions before calculations
    def check_arrays(self, P_in, D, feature_list, feature_unit_classes, ptype):
        """ Check arrays/list P, D and feature_list"""

        P, D, feature_list = np.array(P_in), np.array(D), np.array(feature_list)
        P_shape, D_shape, f_shape = P.shape, D.shape, feature_list.shape

        if not len(D_shape) == 2:
            self.logger.error(
                'Dimension of feature matrix is %s. A two-dimensional list or array is needed.' %
                len(D_shape))
            sys.exit(1)

        if not len(f_shape) == 1:
            self.logger.error(
                'Dimension of feature list is %s. A one-dimensional list or array is needed.' %
                len(f_shape))
            sys.exit(1)

        if not P_shape[0] == D_shape[0]:
            self.logger.error(
                "Length (%s) of target property has to match to number of rows (%s) of feature matrix." %
                (P_shape[0], D_shape[0]))
            sys.exit(1)

        if ptype == 'quanti':
            if not all(isinstance(el, (float, int)) for el in P):
                self.logger.error("For ptype = 'quanti', a 1-dimensional array of floats/ints is required is required.")
                sys.exit(1)

        if ptype == 'quali':
            if not all(isinstance(el, int) for el in P_in):
                self.logger.error("For ptype = 'quali', a 1-dimensional array of ints is required is required.")
                sys.exit(1)

            index = np.unique(P, return_index=True)[1]
            class_names = P[np.sort(index)]
            n_class = len(class_names)
            current_i = 0
            for p in P:
                if not p == class_names[current_i]:
                    current_i += 1
                if n_class == current_i:
                    self.logger.error("For ptype = 'quali', the target property has to be ordered by classes:")
                    self.logger.error("first all members of the first class, next all members of the next class ...")
                    sys.exit(1)

        if not D_shape[1] == f_shape[0]:
            self.logger.error(
                'Length (%s) of feature_list has to match to number of columns (%s) of feature matrix.' %
                (f_shape[0], D_shape[1]))
            sys.exit(1)

        if f_shape[0] < 2:
            self.logger.error('Length of feature_list is %s. Choose at least two features.' % f_shape[0])
            sys.exit(1)

        if not isinstance(feature_unit_classes, (np.ndarray, list, type(None))):
            raise TypeError("'feature_unit_classes' must be numpy array, list or None.")

        if isinstance(feature_unit_classes, (np.ndarray, list)) and f_shape[0] != len(feature_unit_classes):
            self.logger.error('Length of feature_unit_classes does not match length of feature_list.')
            sys.exit(1)

        feature_unit_classes_integers = [f for f in feature_unit_classes if isinstance(f, int)]
        feature_unit_classes_strings = [f for f in feature_unit_classes if isinstance(f, str)]
        if isinstance(feature_unit_classes, (np.ndarray, list)) and (not all(isinstance(f_c, int)
                                                                             for f_c in feature_unit_classes_integers) or not all(f_c == 'no_unit' for f_c in feature_unit_classes_strings)):
            raise TypeError("'feature_unit_classes' must consist of integers or the string 'no_unit', where each integer stands for the unit of a feature, i.e. 1:eV, 2:Angstrom. 'no_unit' is reserved for dimensionless unit.")

    def check_control(self, par_in, par_ref, par_in_path):
        """ Recursive Function to check input control dict tree.

        If for example check_control(control,control_ref,'control')
        function goes through dcit tree control and compares with control_ref
        if correct keys (mandotory, not_mandotory, typos of key string) are set
        and if values are of correct type or of optional list.
        Furthermore it gives Errors with hints what is wrong, and what is needed.

        Parameters
        ----------
        par_in : any key
            if par_in is dict, then recursion.

        par_ref: any key
            Is compared to par_in, if of same time.
            If par_in and par_key are dict, alse keys are compared.

        par_in_path: string
            Gives the dict tree path where, when error occurs, e.g.
            control[key_1][key_2]... For using function from outside
            start with name of input dict, e.g. 'control'

        """

        # check if value_in has correct type = value_ref_type
        self.check_type(par_in, par_ref, par_in_path)

        if isinstance(par_in, dict):
            # check if correct keys are used
            self.check_keys(par_in, par_ref, par_in_path)

            for key_in, value_in in par_in.iteritems():
                # get reference value like: dictionary[key_1][key_2] or here: par_ref[key_in]
                # Needed because control_ref has special form.
                value_ref = self.get_value_from_dic(par_ref, [key_in])

                # recursion
                self.check_control(value_in, value_ref, par_in_path + "['%s']" % key_in)

    def get_type(self, value):
        if isinstance(value, type):
            return value
        else:
            return type(value)

    def check_type(self, par_in, par_ref, par_in_path, if_also_none=False):
        """ Check type of par_in and par_ref.

        If par_ref is tuple, par_in must be item of par_ref:
        else: they must have same type.
        """

        # if par_ref is tuple, then only a few values are allowed. Thus just checked if
        # par_in is in par_ref instead of checking type.
        if isinstance(par_ref, tuple):
            if not par_in in par_ref:
                self.logger.error('%s must be in %s.' % (par_in_path, par_ref))
                sys.exit(1)
        # check if type(par_in) = type(par_ref)
        else:
            # get type of par_ref. type(par_ref) is not enough, since in control_ref
            # strings,integers,dictionaries... AND types as <int>, <dict>, <str> are given.
            ref_type = self.get_type(par_ref)

            if not isinstance(par_in, ref_type):
                if if_also_none and par_in is None:
                    pass
                else:
                    self.logger.error('%s must be %s.' % (par_in_path, ref_type))
                    sys.exit(1)

    def get_value_from_dic(self, dictionary, key_tree_path):
        """ Returns value of the dict tree

        Parameters
        ----------
        dictionary: dict or 'dict tree' as control_ref
            dict_tree is when key is tuple of keys and value is tuple of
            corresponding values.

        key_tree_path: list of keys
            Must be in the correct order beginning from the top of the tree/dict.

        # Examples
        # --------
        # >>> print get_value_from_dic[control_ref, ['local_run','SIS_code_path']]
        # <type 'str'>

        """

        value_ref = dictionary
        for key in key_tree_path:
            value_ref_keys = value_ref.keys()
            if key in value_ref_keys:
                value_ref = value_ref[key]
            else:
                tuples = [tup for tup in value_ref_keys if isinstance(tup, tuple)]
                try:
                    select_tuple = [tup for tup in tuples if key in tup][0]
                except BaseException:
                    raise KeyError
                index = [i for i, key_tuple in enumerate(select_tuple) if key == key_tuple][0]
                value_ref = value_ref[select_tuple][index]
        return value_ref

    def check_keys(self, par_in, par_ref, par_in_path):
        """ Compares the dicts par_in and par_ref.

        Collects which keys are missing (only if keys are not in not_mandotary) amd
                 whcih keys are not expected (if for example there is a typo).
        If there are missing or not expected ones, error message with missing/not expected ones.

        Parameters
        ----------
        par_in : dict

        par_ref : dict

        par_in_path : string
            Dictionary path string for error message, e.g 'control[key_1][key_2]'.

        """

        keys_in, keys_ref = par_in.keys(), par_ref.keys()
        # check if wrong keys are in keys_in
        wrong_keys = [key for key in keys_in if not key in self.flatten(keys_ref)]
        # check missing keys and if exactly one of optional keys is selected
        missing_keys = []
        for key in keys_ref:
            if isinstance(key, tuple):
                optional_in = [k for k in keys_in if k in key]
                leng = len(optional_in)
                if leng > 1:
                    self.logger.error("The following keys are set in %s: %s." % (par_in_path, optional_in))
                    self.logger.error("Please select only one of %s" % list(key))
                    sys.exit(1)
                if leng == 0 and not key in not_mandotary:
                    missing_keys.append("--one of: (%s)" % (", ".join(["'%s'" % k for k in key])))
                    #missing_keys.append(('--one of:',)+key)
            elif not key in keys_in and not key in not_mandotary:
                missing_keys.append(key)

        # error message if needed
        len_wrong, len_missing = len(wrong_keys), len(missing_keys)
        if len_wrong > 0 or len_missing > 0:
            if len_wrong > 0:
                self.logger.error("The following keys are not expected in %s: %s" % (par_in_path, wrong_keys))
            if len_missing > 0:
                self.logger.error("The following keys are missing in %s: %s" % (par_in_path, missing_keys))
            sys.exit(1)

    def check_OP_list(self, control):
        """ Checks form and items of control['parameters']['OP_list'].

        control['parameters']['OP_list'] must be a list of operations strings
        or list of n_comb lists of operation strings. Furthermore if operation
        strings are item of available_OPs (see above) is checked.


        Parameters
        ----------
        control : dict

        Returns
        -------
        control : with manipulated control['parameters']['OP_list']

        """

        OP_list = control['parameters']['opset']
        n_comb = control['parameters']['rung']

        # If just list of strings make list of n_comb lists
        if all(isinstance(OPs, str) for OPs in OP_list):
            # check if correct operations
            self.check_OP_strings(OP_list)

            OP_list = [OP_list for i in range(n_comb)]
            control['parameters']['opset'] = OP_list
            return control

        # If list of lists/tuples check if n_comb lists/tuples
        elif all(isinstance(OPs, (list, tuple)) for OPs in OP_list):
            if not len(OP_list) == n_comb:
                self.return_OP_error()
            try:
                # check if correct operations
                self.check_OP_strings(self.flatten(OP_list))
                control['parameters']['opset'] = OP_list
                return control
            except BaseException:
                self.return_OP_error()
        # False form
        else:
            self.return_OP_error()

    def check_OP_strings(self, OPs):
        """ Check if all items of OPs are items of available_OPs"""
        if not all(op in available_OPs for op in OPs):
            self.logger.error("Available operations: %s" % available_OPs)
            sys.exit(1)

    def return_OP_error(self):
        """ Error message if control['parameters']['OP_list'] has wrong form  """
        self.logger.error("'OP_list' must consist of 'n_comb' tuples/lists of strings of operations.")
        self.logger.error("The other option is that it contains only strings of operations.")
        self.logger.error("Then for each iteration the same operations will be used.")
        sys.exit(1)

    def check_quali_dim(self, control):
        """ Check if quali then also desc_dim=2 """
        if control['parameters']['ptype'] == 'quali' and not control['parameters']['desc_dim'] == 2:
            self.logger.error("At the moment, for ptype = quali only desc_dim = 2 allowed ")
            sys.exit(1)

    def check_(self, k):
        self.key_to_maxcpu_dic = {"/home/keys/Q8E8RS2hj441kaFaLFHSY678g2rgF20f": 1,  # hands-on-CS
                                  "/home/keys/Kucn93hf1F0F38aypq5fD63n7XhDyOP0": 24,  # sis-tutorial metal-nonmetal
                                  "/home/keys/4Sofj9D3I1kc03E39k1fIPO9w9A03N5Z": 5,  # sis-tutorial binaries
                                  "/home/keys/Zn98Li73k39h5Bd0a12eq344ba3maye3": 5}  # sis-tutorial topological insulators
        self.kkey = k
        self.n_cpu = 1
        if k in self.key_to_maxcpu_dic:
            max_cpu = self.key_to_maxcpu_dic[k]
            k = os.path.join(self.local_path, "key.mpi")
            key = base64.b64decode(for_me)
            with open(k, 'w') as f:
                f.write(key)
        else:
            max_cpu = 1

        if not(not self.mpi_command or self.mpi_command.isspace()):
            try:
                idx_n_cpu, self.n_cpu = [(i, int(s)) for i, s in enumerate(self.mpi_command.split()) if s.isdigit()][-1]
                if self.n_cpu > max_cpu:
                    self.n_cpu = max_cpu
                    if self.if_print:
                        self.logger.warning("For your pupose, the maximum allowed CPU number is %s." % max_cpu)
                    self.mpi_command = self.mpi_command.split()
                    self.mpi_command[idx_n_cpu] = str(self.n_cpu)
                    self.mpi_command = " ".join(self.mpi_command)
                if self.if_print:
                    self.logger.info("The calculations are running on %s CPUs." % self.n_cpu)
            except BaseException:
                self.n_cpu = 1
                self.mpi_command = ''
                self.logger.warning("MPI command not known. The calculations are restricted to run on only one CPU.")

        return k

    # feature space estimation

    def ncr(self, n, r):
        """ Binomial coefficient"""
        r = min(r, n - r)
        if r == 0:
            return 1
        numer = reduce(opop.mul, xrange(n, n - r, -1))
        denom = reduce(opop.mul, xrange(1, r + 1))
        return numer // denom

    def check_l0_steps(self, max_dim, n_sis, upper_limit=10000):
        """ Check if number of l0 steps is larger then a upper_limit"""
        l0_steps_list = [self.ncr(n_sis * dim, dim) for dim in range(1, max_dim + 1)]
        l0_steps = sum(l0_steps_list)

        self.l0_steps = l0_steps

        if l0_steps > upper_limit * self.n_cpu:
            logger.error(
                "With the given settings in the l0-regularizaton %s combinations of features have to be considered." %
                l0_steps)
            logger.error(
                "In this version the upper limit for ptype = '%s' is %s*n_CPUs. Choose a smaller" %
                (self.parameters['ptype'], upper_limit))
            logger.error("'Optimal descriptor maximum dimension' or 'Number of collected features per SIS iteration'")
            sys.exit(1)

    def get_next_size(self, n_features, ops):
        new_features = 0
        for op in ops:
            if op in un_OP:
                new_features += n_features
            elif op in bin_OP:
                new_features += n_features**2
            else:
                new_features += self.ncr(n_features, 2)
        return new_features + n_features

    def estimate_feature_space(self, n_comb, n_features, ops, rate=1., n_comb_start=0):
        if isinstance(rate, (float, int)):
            rate = [rate for i in range(n_comb)]
        for i in range(n_comb_start, n_comb):
            n_features = int(self.get_next_size(n_features, ops) * rate[i])
        return int(n_features)

    def check_feature_space_size(self, feature_list, n_target=5, upper_bound=300000000):
        n_comb = deepcopy(self.parameters['rung'])
        max_dim = deepcopy(self.parameters['desc_dim'])
        n_sis = deepcopy(self.parameters['subs_sis'])

        self.parameters['rung'] = 2
        self.parameters['desc_dim'] = 1
        self.parameters['subs_sis'] = 1

        OP_list = self.parameters['opset']

        P = np.random.random((n_target))
        D = np.random.random((n_target, len(feature_list)))

        # make sis calculation to obtain self.featurespace(rung=2) for feature_space estimation
        self.start()
        self.get_results()
        feature_space_size_ncomb2 = self.featurespace

        # set parameters back
        self.parameters['rung'] = n_comb
        self.parameters['desc_dim'] = max_dim
        self.parameters['subs_sis'] = n_sis

        estimate = self.estimate_feature_space(3, feature_space_size_ncomb2, OP_list, rate=0.12, n_comb_start=2)
        self.predicted_feature_space_size = estimate

        if estimate * max_dim > upper_bound * self.n_cpu:
            digit_len = len(str(estimate)) - 1
            logger.error(
                "Estimated order of magnitude of feature space size: 10^%s - 10^%s" %
                (digit_len, digit_len + 1))
            logger.error("In this version the upper bound for n_features is given by:")
            logger.error("%s > n_features*max_dim/n_CPUs" % (upper_bound))
            logger.error("Hint: select less primary features, less operations or a smaller max_dim.")
            logger.error("The registered user will be allowed soon to use larger feature spaces.")
            sys.exit(1)

    def estimate_calculation_expense(self, feature_list):
        """ Check the expense of the SIS+l0 calculations"""
        n_target = 12
        P = np.random.random((n_target))
        D = np.random.random((n_target, len(feature_list)))

        max_dim = self.parameters['desc_dim']
        n_sis = self.parameters['subs_sis']
        n_comb = self.parameters['rung']

        # check l0 steps
        if self.parameters['ptype'] == 'quanti':
            self.check_l0_steps(max_dim, n_sis, upper_limit=1100000)
        else:
            u_l = 180000
            if self.kkey in "/home/keys/Zn98Li73k39h5Bd0a12eq344ba3maye3":  # topological insulator
                u_l /= 5
            elif self.kkey in "/home/keys/Kucn93hf1F0F38aypq5fD63n7XhDyOP0":  # metal-nonmetal
                u_l = 1150000
            self.check_l0_steps(max_dim, n_sis, upper_limit=u_l)

        # check feature spcae
        if n_comb == 3:
            if self.kkey in "/home/keys/Zn98Li73k39h5Bd0a12eq344ba3maye3":  # topological insulator
                logger.error(
                    "A 'number of iterations for the construction for the feature space' > 2 is not allowed for this tutorial.")
                sys.exit()
            u_l = 4460000
            if self.kkey in "/home/keys/Kucn93hf1F0F38aypq5fD63n7XhDyOP0":
                u_l = 4460000 * 2
            self.check_feature_space_size(feature_list, n_target=n_target, upper_bound=u_l)
        elif n_comb > 3:
            logger.error("A 'number of iterations for the construction for the feature space' >3 is not allowed.")
            sys.exit(1)


# END checking functions

    def do_transfer(self, ssh=None, eos=None, username=None, CPUs=None):
        """ Run the calcualtion on remote machine

        First checks if already folder self.remote_input_path exists on remote machine,
        if yes it deletes or renames it.
        Then copies file system self.SIS_input_path with SIS fortran code files into the
        folder self.remote_input_path. Finally lets run the calculations on remote machine
        and copy back the file system with results.
        If eos, writes submission script, submits script and checks qstat if calculation
        finished.

        Parameters
        ----------
        ssh : object
            Must be from code nomad_sim.ssh_code.

        eos : bool
            If remote machine is eos. To write submission script and submit ...

        username: string
            needed to check qstat on eos

        CPUs : int
            To reserve the write number of CPUs in the eos submission script

        """

        # check if remote_input_path exists and if yes rename it to remote_input_path_old_#
        if self.ssh.isdir(self.remote_input_path):
            self.logger.warning('Directory %s on remote machine already exists.' % self.remote_input_path)
            if self.rm_existing_files:
                ssh.rm(self.remote_input_path)
                self.logger.warning('It is removed.')
            else:
                for i in range(1000):
                    old_name = "%s_old_%s" % (self.remote_input_path, i)
                    if not self.ssh.isdir(old_name):
                        self.ssh.rename(self.remote_input_path, old_name)
                        break
                self.logger.warning('It is renamed to %s.' % old_name)

        if eos:
            self.write_submission_script(CPUs)

        # copy self.SIS_input_path INto self.remote_path
        ssh.put_all(self.SIS_input_path, self.remote_path)
        rmtree(self.SIS_input_path)

        if eos:
            seconds = 1
            # submit job called go.sge
            ssh.command("cd %s; qsub go.sge" % self.remote_input_path)
            self.SCHEDule = sched.scheduler(time.time, time.sleep)

            # check each seconds if is job is finished
            self.SCHEDule.enter(seconds, 1, self.ask_periodically, (self.SCHEDule, seconds, 0, username))
            self.SCHEDule.run()
        else:
            # execute SIS_code on remote machine
            # exporting path is needed, since code FCDI calls the codes FC and DI by just 'FC' and 'DI'.
            ssh.command('export PATH=$PATH:%s; cd %s; %s' %
                        (self.SIS_code_path, self.remote_input_path, self.SIS_code_FCDI))

        # copy back file system with results
        ssh.get_all(self.remote_input_path, self.local_path)
        ssh.rm(self.remote_input_path)
        # close ssh connection
        if self.if_close_ssh:
            ssh.close()

    def check_status(self, filename, username):
        """ Check if calculation on eos is finished

        Parameters
        filename: str
            qstat will be written into this file. The file will be then read.

        username: str
            search in filename for this username. If not appears calculation is finished.

        Returns
        -------
        status : bool
            True if calculations is still running.
        """
        # write qstat into filenmae
        self.ssh.command("qstat -u %s > %s" % (username, filename))
        status = False

        # read filename
        lines = self.ssh.open_file(filename).readlines()
        for line in lines:
            split = line.split()
            if len(split) > 3:
                # if job name SIS_tutori (only 10 char) and username appears
                if split[2] == 'SIS_tutori' and split[3] == username:
                    status = True
        return status

    def ask_periodically(self, sc, seconds, counter, username):
        """ Recursive function that runs periodically (each seconds) the
            function self.check_status.
        """

        counter += 1
        filename = os.path.join(self.remote_input_path, 'status.dat')
        if counter > 1000:
            return 1
        if not self.check_status(filename, username):
            return 0
        self.SCHEDule.enter(seconds, 1, self.ask_periodically, (sc, seconds, counter, username))

    def write_submission_script(self, CPUs):
        """ writes eos job submission script. """

        strings = [
            "#$ -S /bin/bash",
            "#$ -j n",
            "#$ -N SIS_tutorial",  # jobname
            "#$ -cwd",
            "#$ -m n",
            "#$ -pe impi_hydra %s" % CPUs,  # CPUs= nodes*32!
            "#$ -l h_rt=00:01:00",  # time reservation for job
            "%s" % SIS_code_FCDI
        ]
        # write submission file "go.sge"
        submission_file = open(os.path.join(self.SIS_input_path, 'go.sge'), 'w')
        for s in strings:
            submission_file.write("%s\n" % s)
        submission_file.close()

    def check_feature_units(self, feature_unit_classes):
        """ Check feature units

        Checks which

        Parameters
        ----------
        feature_unit_classes : list integers
            list must be sorted.

        Returns
        -------
        unit_strings : list of strings
            In the form ['(1:3)','(4:8)',..], where the indices start from 1,
        """

        index = np.unique(feature_unit_classes, return_index=True)[1]
        class_names = [feature_unit_classes[i] for i in np.sort(index)]
        unit_strings = []
        col = 0
        for i, cl in enumerate(class_names):
            length = len([None for p in feature_unit_classes if p == cl])
            if cl != 'no_unit':
                unit_strings.append("(%s:%s)" % (col + 1, col + length))
            col += length
        return unit_strings

    def convert_feature_strings(self, feature_list):
        """  Convert feature strings.

        Puts an 'sr' for reals and an 'si' for integers at the beginning of a string.
        Returns the list with the changed strings.
        """

        converted = []
        for f in feature_list:
            if f in reals:
                which = 'r'
            elif f in ints:
                which = 'i'
            else:
                self.logger.error("Developer error: %s not found in the list reals or ints." % f)
                sys.exit(1)
            f = standard_2_converted[f]
            converted.append('s%s_%s' % (which, f))
        return converted

    def write_parameters(self):
        """ Write parameters into the SIS fortran code input files. Convert the parameters into
            the special format before."""

        filename = 'FCDI.in'
        input_file = open(os.path.join(self.SIS_input_path, filename), 'w')

        # loop in correct order as in Param_key_list could be essential. So better no iteritems()
        for key in Param_key_list:
            value = self.parameters[key]
            value = self.convert_2_fortran(key, value)
            input_file.write("%s=%s\n" % (key, value))
        input_file.close()

    def convert_2_fortran(self, parameter, parameter_value):
        """ Convert parameters to SIS fortran code style.

            Converts e.g. True to string '.true.' or a string 's' to
            "'s'", and other special formats.
            Returns the converted parameter.
        """

        if parameter == 'opset':
            return self.get_OPs(parameter_value)
        elif parameter == 'dimclass':
            return "".join(parameter_value)
        elif isinstance(parameter_value, bool):
            if parameter_value == True:
                return '.true.'
            else:
                return '.false.'
        elif isinstance(parameter_value, str):
            return "'%s'" % parameter_value
        elif isinstance(parameter_value, tuple) and len(parameter_value) == 1:
            return "(%s)" % parameter_value[0]
        else:
            return parameter_value

    def get_OPs(self, OP_list):
        """ Conver OP_list to special format for SIS fortran input."""

        list_of_strings = []
        for OPs in OP_list:
            # convert OP_list: in example ['+', '-', '/', '^2', 'exp'] to '(+)(-)(/)(^2)(exp)'
            OP_string = ""
            for op in OPs:
                OP_string += '(%s)' % op
            list_of_strings.append("'%s'" % OP_string)

        # make string of OP_string listed ncomb times e.g. "'(+)(-)(/)(^2)(exp)','(+)(-)(/)(^2)(exp)',..."
        converted = ",".join(list_of_strings)

        return converted

    def flatten(self, list_in):
        """ Returns the list_in collapsed into a one dimensional list

            Parameters
            ----------
            list_in : list/tuple of lists/tuples of ...
        """
        list_out = []
        for item in list_in:
            if isinstance(item, (list, tuple)):
                list_out.extend(self.flatten(item))
            else:
                list_out.append(item)
        return list_out

    def write_P_D(self, P, D, feature_list):
        """ Writes 'train.dat' as SIS fortran code input with P, D and feature strings"""

        #converted_features = self.convert_feature_strings(feature_list)
        converted_features = feature_list
        P = np.array(P)
        P_shape = P.shape
        if self.parameters['ptype'] == 'quanti':
            if len(P_shape) > 1 and not P_shape[1] == 1:
                first_line = ['#'] + ['target_%s' % (t + 1) for t in range(P_shape[1])]
            else:
                first_line = ['#', 'target']
            P = np.transpose(np.vstack((['xxx' for i in range(len(P))], P)))
        else:
            entries_of_P = len(P)
            P = P.reshape([entries_of_P, 1])
            first_line = ['#']

        first_line.extend(converted_features)

        Out = np.hstack((P, D))
        Out = np.vstack((first_line, Out))
        np.savetxt(os.path.join(self.SIS_input_path, "train.dat"), Out, fmt='%s', delimiter="    ")

    def get_des(self, x):
        """ Change the descriptor strings read from the output DI.out.
        Remove characters as ':' 'si', 'sr'. Then convert feature strings for printing"""
        index = [n_i for n_i, i in enumerate(x) if i == ':'][0]
        x = x[index + 2:-1]
        x = list(x)
        remove_index = []
        for n_i, i in enumerate(x):
            if i == 's':
                if x[n_i + 1] in ['r', 'i']:
                    if x[n_i + 2] == '_':
                        remove_index.extend(range(n_i, n_i + 3))
        x = [s for i, s in enumerate(x) if not i in remove_index]
        if x[0] == '(' and x[-1] == ')':
            x = x[1:-1]
        new_string = "".join(x)

        return new_string

    def check_FC(self, file_path):
        """ Check FC.out, if calculation has finished and feature space_sizes.

        Returns
        -------
        calc_finished : bool
            If calculation finished there shoul be a 'Have a nice day !'.

        featurespace : integer
            Total feature space size generated, before the redundant check.

        n_collected : integer
            The number of features collected in the current iteration.
            Should be n_sis.

        """

        lines = open(file_path, 'r').readlines()
        featurespace = None
        n_collected = None
        calc_finished = False
        feature_space_list = []
        for line in lines:
            if line.rfind('Total Featurespace:') > -1:
                feature_space_list.append(line.split()[2])
            if line.rfind('Have a nice day !') > -1:
                calc_finished = True
            if line.rfind('Final feature space size:') > -1:
                n_collected = int(line.split()[4])
        return calc_finished, feature_space_list, n_collected

    def check_DI(self, file_path):
        """ Check DI.out, if calculation has finished. """
        lines = open(file_path, 'r').readlines()
        calc_finished = False
        for line in lines:
            if line.rfind('Have a nice day !') > -1:
                calc_finished = True
        return calc_finished

    def check_files(self, iter_folder_name, dimension):
        """ Check which file is missing and maybe why.

        This function, if something went wrong to find out where the problem occured.
        Returns an error string.
        """
        iter_path = os.path.join(self.SIS_input_path, iter_folder_name)
        DI_path = os.path.join(iter_path, 'DI.out')
        FC_path = os.path.join(iter_path, 'FC.out')

        if_iter = os.path.isdir(iter_path)
        if_FC = os.path.isfile(FC_path)
        if_DI = os.path.isfile(DI_path)

        n_sis = self.parameters['subs_sis']
        sub_space_size = dimension * n_sis
        if if_iter:
            if if_FC:
                calc_finished, feature_space, n_collected = self.check_FC(FC_path)
                if not calc_finished:
                    return 'FC.out not finished'
                if feature_space is None:
                    return "'Total Featurespace' not found"
            else:
                return 'FC.out not found'

            if n_collected < n_sis:
                return 'No %sD descriptor!\nThe number of collected feateres in iteration %s is %s. Probably the total feature space size is not large enough. Collect less features per iteration.\nTotal feature space size before redundant check: %s\n      Target total number of collected features: %s\nAfter eliminating redundant features the total feature space becomes smaller.' % (
                    dimension, dimension, n_collected, feature_space, sub_space_size)
            if if_DI:
                calc_finished = self.check_DI(DI_path)
                if not calc_finished:
                    return 'DI.out not finished'
            else:
                return 'DI.out not found'

            return 'Unknown error'
        else:
            return '%s not found' % iter_folder_name

    def read_results(self, iter_folder_name, dimension, task, tsizer):
        """ Read results from DI.out.


        parameters
        ----------
        iter_folder : string
            Name of the iter_folder the outputs of the corresponding iteration of SIS+l1/l1l0,
            e.g. 'iter01', 'iter02'.

        dimension : integer
            DI.out provides for example in iteration three 1-3 dimensionl descriptors.
            Here choose which dimension should be returned.

        task : integer < 100
            For multi task, must be worked on.

        tsizer : integer
            Number of samples, e.g. number ofrows of D or P.

        Returns
        -------
        RMSE : float
            Root means squares error of model

        Des : list of strings
            List of the descriptors

        coef : array [model_dim+1]
            Coefficients including the intercept

        D : array [n_sample, model_dim+1]
            Matrix with columns being the selected features (descriptors) for the model.
            The last column is full of ones corresponding to the intercept

        """
        iter_path = os.path.join(self.SIS_input_path, iter_folder_name)
        DI_path = os.path.join(iter_path, 'DI.out')

        if task > 9:
            s_task = '0%s' % task
        else:
            s_task = '00%s' % task
        desc_path = os.path.join(iter_path, 'desc_dat', 'desc%s_%s.dat' % (dimension, s_task))

        count_dim = 0
        lines = open(DI_path, 'r').readlines()

        for line in lines:
            if line.rfind('@@@descriptor') > -1:
                count_dim += 1
                if count_dim == dimension:
                    des = line.split()[1:]
                    Des = [self.get_des(x) for x in des]  # convert strings
            if count_dim == dimension:
                if line.rfind('coefficients_') > -1:
                    coef = np.array([float(i) for i in line.split()[1:]])
                if line.rfind('Intercept_') > -1:
                    inter = float(line.split()[1])
                    coef = np.append(coef, inter)
                if line.rfind('LSrmse') > -1:
                    RMSE = float(line.split()[1])

        D = np.empty([tsizer, dimension])
        lines = open(desc_path, 'r').readlines()
        for i, line in enumerate(lines):
            if i > 0:
                for j, val in enumerate(line.split()[3:]):
                    D[i - 1, j] = val
        D = np.column_stack((D, np.ones(tsizer)))
        return RMSE, Des, coef, D

    def get_indices_of_top_descriptors(self):
        try:
            filename = [f for f in os.listdir(self.iter_path,) if f[-2:] == '2D' and f[:3] == 'top'][0]
        except BaseException:
            self.logger.error("Calculation Aborted.")
            self.logger.error("The Number of collected features in the SIS step might have exceeded")
            self.logger.error("the number of features in the created feature space.")
            self.logger.error("Hint: Try a smaller 'Number of collected features per SIS iteration'")
            self.logger.error("Hint: or increase the feature space size.")
            sys.exit()
            #filename = "top%04d_02D" % n_out
        filename = os.path.join(self.iter_path, filename)
        top_dat = open(filename, 'r').readlines()
        Ind = []
        Overlaps = []
        old_n_overlap, old_overlap_area = None, None
        for l, line in enumerate(top_dat):
            if l > 0:
                n_overlap, overlap_area = int(line.split()[1]), float(line.split()[2])
                if old_n_overlap in [n_overlap, None] and old_overlap_area in [overlap_area, None]:
                    indices = [int(idx) - 1 for idx in line.split()[-2:]]
                    Ind.append(indices)
                    Overlaps.append(n_overlap)
                    old_n_overlap, old_overlap_area = n_overlap, overlap_area
                else:
                    break
        return Overlaps, Ind

    def manipulate_descriptor_string(self, d):
        if d[0] == '(' and d[-1] == ')':
            return d[1:-1]
        else:
            return d

    def get_strings_of_top_descriptors(self, top_indices):
        filename = os.path.join(self.iter_path, "task.fname")
        lines = open(filename, 'r').readlines()
        descriptors = [line.split()[0] for line in lines]
        # importan to return [1:-1] to remove brackets in string
        return [[self.manipulate_descriptor_string(descriptors[i]) for i in indices] for indices in top_indices]

    def get_arrays_of_top_descriptors(self, top_indices):
        n_models = len(top_indices)
        top_indices = np.array(top_indices)
        filename = os.path.join(self.iter_path, 'task001.dat')
        lines = open(filename, 'r').readlines()
        Ds = []
        for line in lines:
            ls = line.split()
            Ds.append([float(ls[i]) for i in top_indices.flatten()])
        Ds = np.array(Ds)
        return [Ds[:, [2 * i, 2 * i + 1]] for i in range(n_models)]

    def read_results_quali(self):
        """ Read results for 2D desriptor from calculations with qualitative run.

        Returns
        -------
        results: list of lists
            Each sublist characterizes separate model (if multiple model have same score/cost
            all of them are returned). Sublist contains [descriptor_strings, D, n_overlap]
            where D (D.shape = (n_smaple,2)) is array with descriptor vectors.
        """
        self.iter_path = os.path.join(self.SIS_input_path, "iter02")
        Overlaps, Top_indices = self.get_indices_of_top_descriptors()
        Top_strings = self.get_strings_of_top_descriptors(Top_indices)
        Top_Ds = self.get_arrays_of_top_descriptors(Top_indices)
        return [[Top_strings[i], Top_Ds[i], Overlaps[i]] for i in range(len(Top_indices))]

    def string_descriptor(self, RMSE, features, coefficients, target_unit):
        """ Make string for output in the terminal with model and its RMSE."""

        dimension = len(features)

        string = '%sD descriptor:\nRoot Mean Squared Error (RMSE): %s %s\nModel: \n' % (dimension, RMSE, target_unit)
        for i in range(dimension + 1):
            if coefficients[i] > 0:
                sign = '+'
                c = coefficients[i]
            else:
                sign = '-'
                c = abs(coefficients[i])
            if i < dimension:
                string += '%s %.5f %s\n' % (sign, c, features[i])
            else:
                string += '%s %.5f\n' % (sign, c)
        return string

    def get_results(self, ith_descriptor=0):
        """ Attribute to get results from the file system.

        Parameters
        -------
        ith_descriptor: int
            Return the ith best descriptor.

        Returns
        -------
        out : list [max_dim] of dicts {'D', 'coefficients', 'P_pred'}

        out[model_dim-1]['D'] : pandas data frame [n_sample, model_dim+1]
            Descriptor matrix with the columns being algebraic combinations of the input feature matrix.
            Column names are thus strings of the algebraic combinations of strings of inout feature_list.
            Last column is full of ones corresponding to the intercept

        out[model_dim-1]['coefficients'] : array [model_dim+1]
            Optimizing coefficients.

        out[model_dim-1]['P_pred'] : array [m_sample]
            Fit : np.dot( np.array(D) , coefficients)

        """

        max_dim = self.parameters['desc_dim']
        Results_list = []
        tsizer = len(self.flatten(self.P))

        if self.parameters['ptype'] == 'quanti':
            for dimension in range(1, max_dim + 1):
                if dimension < 10:
                    iter_folder_name = 'iter0%s' % (dimension)
                else:
                    iter_folder_name = 'iter%s' % (dimension)

                try:
                    results = self.read_results(iter_folder_name, dimension, 1, tsizer)
                    Results_list.append(results)
                    if dimension == 1:
                        iter_path = os.path.join(self.SIS_input_path, iter_folder_name)
                        FC_path = os.path.join(iter_path, 'FC.out')
                        # feature space size
                        feature_space_list = self.check_FC(FC_path)[1]
                        try:
                            self.featurespace = int(feature_space_list[-1])
                            featurespace = int(self.featurespace * 0.5)
                        except BaseException:
                            if self.parameters['rung'] == 3:
                                featurespace = int(feature_space_list[-2])
                                self.featurespace = self.estimate_feature_space(
                                    3, featurespace, self.parameters['opset'], rate=0.12, n_comb_start=2)
                                featurespace = int(self.featurespace)
                            else:
                                self.logger.error("Developper error: feature space estimation and rung conflict!")
                                self.exit(1)
                        if self.if_print:
                            digit_len = len(str(featurespace)) - 1
                            self.logger.info(
                                "Estimated order of magnitude of feature space size: 10^%s - 10^%s" %
                                (digit_len, digit_len + 1))
                except Exception as e:
                    message = self.check_files(iter_folder_name, dimension)
                    if dimension > 2:
                        self.logger.warning(message)
                        break
                    else:
                        self.logger.error(message)
                        self.logger.error("## See below the Error message:")
                        self.logger.error(e)
                        sys.exit(1)
            out = []
            # print results, make pandas DataFrames and calulate predicted/fitted values
            for RMSE, features_selected, coefficients, D_model in Results_list:
                if self.if_print:
                    string = self.string_descriptor(RMSE, features_selected, coefficients, self.target_unit)
                    self.logger.info(string)

                # predicted/fitted values of the model
                fit = np.dot(D_model, coefficients)

                # D_model and selected features as pandas DataFrames
                features_selected.append('intercept')
                D_df = pd.DataFrame(D_model, columns=features_selected)

                out.append({'D': D_df, 'coefficients': coefficients, 'P_pred': fit})
            rmtree(self.SIS_input_path)
            return out

        else:  # 'quali'. Only for specific case of 2D
            dimension = 2
            iter_folder_name = 'iter0%s' % (dimension)
            try:
                iter_path = os.path.join(self.SIS_input_path, iter_folder_name)
                FC_path = os.path.join(self.SIS_input_path, 'iter01', 'FC.out')

                # feature space size
                feature_space_list = self.check_FC(FC_path)[1]
                try:
                    self.featurespace = int(feature_space_list[-1])
                    if self.parameters['rung'] == 3:
                        featurespace = int(self.featurespace * 0.5)
                    else:
                        featurespace = int(self.featurespace)
                except BaseException:
                    if self.parameters['rung'] == 3:
                        featurespace = int(feature_space_list[-2])
                        self.featurespace = self.estimate_feature_space(
                            3, featurespace, self.parameters['opset'], rate=0.12, n_comb_start=2)
                        featurespace = int(self.featurespace)
                    else:
                        self.logger.error("Developper error: feature space estimation and rung conflict!")
                        self.exit(1)
                digit_len = len(str(featurespace)) - 1
                first_digit = str(round(featurespace, -digit_len))[0]
                feature_space_message = "Size of feature space: %s*10^%s" % (first_digit, digit_len)

                # get results
                results_list = None
                if not self.checking_expense:
                    results_list_v1 = self.read_results_quali()
                    rmtree(self.SIS_input_path)

                    n_results = len(results_list_v1)

                    # get real overlap with width=0
                    self.parameters['rung'] = 0
                    self.parameters['subs_sis'] = 1
                    self.parameters['width'] = 0.0

                    self.parameters['ndimtype'] = 2
                    self.parameters['dimclass='] = ['(1:1)', '(2:2)']
                    self.parameters['nsf'] = 2

                    self.parameters['mpiname'] = ''
                    #self.if_print = False
                    try:
                        Des, D_selected, overlap = results_list_v1[ith_descriptor]
                    except BaseException:
                        Des, D_selected, overlap = results_list_v1[-1]
                    self.D = D_selected
                    self.feature_list = Des
                    self.feature_unit_classes = [1, 2]

                    self.if_close_ssh = True
                    self.start()
                    final_result = self.read_results_quali()[0]
                    rmtree(self.SIS_input_path)
                try:
                    rmtree(self.SIS_input_path)
                except BaseException:
                    pass

                if self.if_print:
                    self.logger.info("SISSO CALCULATION FINISHED")
                    self.logger.info(feature_space_message)

                return final_result
            except Exception as e:
                self.logger.error(e)
                sys.exit(1)
