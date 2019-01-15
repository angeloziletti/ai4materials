#!/usr/bin/python
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

__author__ = "Angelo Ziletti"
__copyright__ = "Angelo Ziletti"
__maintainer__ = "Angelo Ziletti"
__email__ = "ziletti@fhi-berlin.mpg.de"
__date__ = "20/08/18"

import json
import logging
from ai4materials.external.local_meta_info import loadJsonFile, InfoKindEl
import os
import paramiko
from stat import S_ISDIR
import yaml
import warnings
import shutil
import pkgutil
import sys
import pkg_resources

logger = logging.getLogger('ai4materials')


def set_configs(main_folder='./', config_file=None):

    if config_file is not None:
        config_file_path = os.path.abspath(os.path.normpath(config_file))

        with open(config_file_path, 'r') as config_data_file:
            # use safe_load for security reasons
            configs = yaml.safe_load(config_data_file)
    else:
        configs = {}

    atomic_metainfo = get_data_filename('descriptors/atomic_data.nomadmetainfo.json')
    ureg_file = get_data_filename('utils/units.txt')
    css_file_viewer = get_data_filename('visualization/nomad_viewer.css')
    jsmol_folder = get_data_filename('data/viewer_files/jsmol')

    configs['metadata'] = dict(nomad_meta_info=atomic_metainfo)
    configs['others'] = dict(ureg_file=ureg_file)
    configs['html'] = dict(css_file_viewer=css_file_viewer, jsmol_folder=jsmol_folder)
    configs['runtime'] = dict(isBeaker=False, log_level_general='INFO')

    if 'io' not in configs.keys():
        configs['io'] = dict()

    if 'main_folder' not in configs['io'].keys():
        configs['io']['main_folder'] = os.path.abspath(os.path.normpath(main_folder))

    # create main folder if it does not exist
    if not os.path.exists(configs['io']['main_folder']):
        os.makedirs(configs['io']['main_folder'])

    desc_folder = os.path.abspath(os.path.normpath(os.path.join(configs['io']['main_folder'], 'desc_folder')))
    results_folder = os.path.abspath(os.path.normpath(os.path.join(configs['io']['main_folder'], 'results_folder')))
    tmp_folder = os.path.abspath(os.path.normpath(os.path.join(configs['io']['main_folder'], 'tmp_folder')))
    control_file = os.path.abspath(os.path.normpath(os.path.join(configs['io']['main_folder'], 'control.json')))
    configs['io']['control_file'] = control_file

    configs['io']['desc_folder'] = desc_folder
    configs['io']['results_folder'] = results_folder
    configs['io']['tmp_folder'] = tmp_folder

    # derived paths
    desc_info_file = os.path.abspath(os.path.normpath(os.path.join(configs['io']['desc_folder'],
                                                                   'desc_info.json.info')))
    log_file = os.path.abspath(os.path.normpath(os.path.join(configs['io']['tmp_folder'], 'output.log')))
    results_file = os.path.abspath(os.path.normpath(os.path.join(configs['io']['results_folder'], 'results.csv')))
    conf_matrix_file = os.path.abspath(os.path.normpath(os.path.join(configs['io']['results_folder'], 'confusion_matrix.png')))

    # output_logfile = tempfile.NamedTemporaryFile(prefix='log_', delete=False)
    configs['io']['desc_info_file'] = desc_info_file
    configs['io']['log_file'] = log_file
    configs['io']['results_file'] = results_file
    configs['io']['conf_matrix_file'] = conf_matrix_file

    # create folder
    path_dirs = [configs['io']['desc_folder'], configs['io']['results_folder'], configs['io']['tmp_folder']]
    for path_dir in path_dirs:
        if not os.path.exists(path_dir):
            os.makedirs(path_dir)

    return configs


def copy_directory(src, dest):
    try:
        shutil.copytree(src, dest)
    # Directories are the same
    except shutil.Error as err:
        logger.warning('Directory not copied. Error: {}'.format(err))
    # Any error saying that the directory doesn't exist
    except OSError as err:
        logger.warning('Directory not copied. Error: {}'.format(err))


def overwrite_configs(configs, dataset_folder=None, desc_folder=None, main_folder=None, tmp_folder=None):

    if 'io' not in configs.keys():
        configs['io'] = dict()

    if main_folder is not None:
        logger.debug("Overwriting the main_folder specified in the config file.")
        logger.debug("Main folder: {}".format(main_folder))
        configs['io']['main_folder'] = main_folder

    if dataset_folder is not None:
        logger.debug("Overwriting the dataset_folder specified in the config file.")
        logger.debug("Dataset folder: {}".format(dataset_folder))
        configs['io']['dataset_folder'] = dataset_folder

    if desc_folder is not None:
        logger.debug("Overwriting the desc_folder specified in the config file.")
        logger.debug("Desc folder: {}".format(desc_folder))
        configs['io']['desc_folder'] = desc_folder

    if tmp_folder is not None:
        logger.debug("Overwriting the tmp_folder specified in the config file.")
        logger.debug("Temp folder: {}".format(tmp_folder))
        configs['io']['tmp_folder'] = tmp_folder

    return configs


def setup_logger(configs=None, level=None, display_configs=False):
    """Given specified configurations, setup a logger."""
    if configs is None:
        # load default configs
        configs = set_configs()

    # if level is passed, overwrite the config file values if config is present
    if level is not None:
        if configs is not None:
            configs['runtime']['log_level_general'] = level
    else:
        level = configs['runtime']['log_level_general']

    hdl_file = logging.FileHandler(configs['io']['log_file'] , mode='w+')

    nomadml_logger = logging.getLogger('ai4materials')
    nomadml_logger.setLevel(level)
    logger_format = "%(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt=logger_format)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    hdl_file.setFormatter(formatter)
    nomadml_logger.addHandler(handler)
    nomadml_logger.addHandler(hdl_file)
    nomadml_logger.setLevel(level)
    nomadml_logger.propagate = False

    if display_configs:
        logger.info("Configs: {}".format(json.dumps(configs, indent=2)))

    # disable DeprecationWarning for scikit-learn
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    # disable warnings from pint
    logging.getLogger('pint').setLevel(logging.ERROR)
    # set condor (package to calculate x-ray diffraction) to INFO level even if ai4materials is at DEBUG level
    logging.getLogger('condor').setLevel(logging.INFO)

    return nomadml_logger


class SSH(object):
    """SSH class to connect to the cluster to perform a calculation.

        .. codeauthor:: Emre Ahmetcik <ahmetcik@fhi-berlin.mpg.de>"""
    def __init__(self, hostname='172.17.0.3', username='tutorial', port=22,
                 key_file="/home/beaker/docker.openmpi/ssh/id_rsa.mpi", password=None):
        if password is None:
            pkey_path = key_file
            key = paramiko.RSAKey.from_private_key_file(pkey_path)
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(hostname, username=username, port=port, pkey=key)
        else:
            self.ssh = paramiko.SSHClient()
            self.ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            self.ssh.connect(hostname, username=username, port=port, password=password)
        self.sftp = self.ssh.open_sftp()

    def close(self):
        self.sftp.close()

    def command(self, cmd):
        stdin, stdout, stderr = self.ssh.exec_command(cmd)
        return stdout.read()

    def mkdir(self, path):
        try:
            self.sftp.mkdir(path)
        except BaseException:
            pass

    def remove(self, path):
        if self.isdir(path):
            files = self.sftp.listdir(path=path)
            for f in files:
                filepath = os.path.join(path, f)
                if self.isdir(filepath):
                    self.rm(filepath)
                else:
                    self.sftp.remove(filepath)
            self.sftp.rmdir(path)
        else:
            self.sftp.remove(path)

    def rm(self, path):
        try:
            self.remove(path)
        except BaseException:
            pass

    def exists(self, path):
        try:
            self.sftp.stat(path)
            return True
        except BaseException:
            return False

    def isdir(self, path):
        try:
            return S_ISDIR(self.sftp.stat(path).st_mode)
        except IOError:
            return False

    def open_file(self, filename):
        return self.sftp.open(filename)

    def rename(self, remotefile_1, remotefile_2):
        self.sftp.rename(remotefile_1, remotefile_2)

    def put(self, localfile, remotefile):
        self.sftp.put(localfile, remotefile)

    def put_all(self, localpath, remotepath):
        #  recursively upload a full directory
        os.chdir(os.path.split(localpath)[0])
        parent = os.path.split(localpath)[1]
        for walker in os.walk(parent):
            try:
                self.sftp.mkdir(os.path.join(remotepath, walker[0]))
            except BaseException:
                pass
            for f in walker[2]:
                self.put(os.path.join(walker[0], f), os.path.join(remotepath, walker[0], f))

    def get(self, remotefile, localfile):
        #  Copy remotefile to localfile, overwriting or creating as needed.
        self.sftp.get(remotefile, localfile)

    def sftp_walk(self, remotepath):
        # Kindof a stripped down  version of os.walk, implemented for
        # sftp.  Tried running it flat without the yields, but it really
        # chokes on big directories.
        path = remotepath
        files = []
        folders = []
        for f in self.sftp.listdir_attr(remotepath):
            if S_ISDIR(f.st_mode):
                folders.append(f.filename)
            else:
                files.append(f.filename)
        yield path, folders, files
        for folder in folders:
            new_path = os.path.join(remotepath, folder)
            for x in self.sftp_walk(new_path):
                yield x

    def get_all(self, remotepath, localpath):
        #  recursively download a full directory
        #  Harder than it sounded at first, since paramiko won't walk
        #
        # For the record, something like this would gennerally be faster:
        # ssh user@host 'tar -cz /source/folder' | tar -xz

        self.sftp.chdir(os.path.split(remotepath)[0])
        parent = os.path.split(remotepath)[1]
        try:
            os.mkdir(localpath)
        except BaseException:
            pass
        for walker in self.sftp_walk(parent):
            try:
                os.mkdir(os.path.join(localpath, walker[0]))
            except BaseException:
                pass
            for file in walker[2]:
                self.get(os.path.join(walker[0], file), os.path.join(localpath, walker[0], file))


def get_data_filename(resource, package='ai4materials'):
    """Rewrite of pkgutil.get_data() that return the file path.

    Taken from: https://stackoverflow.com/questions/5003755/how-to-use-pkgutils-get-data-with-csv-reader-in-python
    """
    loader = pkgutil.get_loader(package)
    if loader is None or not hasattr(loader, 'get_data'):
        return None
    mod = sys.modules.get(package) or loader.load_module(package)
    if mod is None or not hasattr(mod, '__file__'):
        return None

    # Modify the resource name to be compatible with the loader.get_data
    # signature - an os.path format "filename" starting with the dirname of
    # the package's __file__
    parts = resource.split('/')
    parts.insert(0, os.path.dirname(mod.__file__))
    resource_name = os.path.normpath(os.path.join(*parts))

    return resource_name


def get_metadata_info():
    """Get the descriptor metadata info"""
    resource_path = '/'.join(('descriptors', 'descriptors.nomadmetainfo.json'))
    desc_metainfo = json.loads(pkg_resources.resource_string('ai4materials', resource_path).decode('utf-8'))

    return desc_metainfo


def read_nomad_metainfo(origin='ai4materials'):
    """Read the atomic nomad meta info file"""
    if origin == 'ai4materials':
        metadata_path = get_data_filename(resource='descriptors/atomic_data.nomadmetainfo.json', package='ai4materials')
    else:
        raise ValueError("Only origin='ai4materials' is currently supported for reading atomic nomad-meta-info.")

    metadata_info, warns = loadJsonFile(filePath=metadata_path, dependencyLoader=None,
                                        extraArgsHandling=InfoKindEl.ADD_EXTRA_ARGS, uri=None)

    return metadata_info
