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
__date__ = "23/09/18"

import logging
import multiprocessing
import os
from ai4materials.utils.utils_data_retrieval import write_summary_file
import tarfile

logger = logging.getLogger('ai4materials')


def split_list(sequence, nb_splits):
    """ Split l in n_split. It can return unevenly sized chunks.

    Parameters:

    sequence: iterable
        Iterable object to be split in `nb_splits`.

    nb_splits: int
        Number of splits.

    Returns:

    iterable
        `sequence` splits in `nb_splits`.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    return [sequence[i::nb_splits] for i in range(nb_splits)]


def collect_desc_folders(descriptor, desc_file, desc_folder, nb_jobs, tmp_folder, remove=False):
    """ Collects the descriptors calculated by different parallel processes and merge them a single .tar.gz file.

    Parameters:

    descriptor: `ai4materials.descriptors.Descriptor`
        The descriptor used in the calculation.

    desc_file: str
        Name of the file where the descriptor is going to be saved.

    desc_folder: str
        Path to the descriptor folder.

    nb_jobs: int
        Number of parallel jobs sent.

    tmp_folder: str
        Path to the tmp folder.

    remove: bool
        If ``True``, the descriptor files from the parallel computations are erased after they have been merged.

    Returns:

    str
        Path to the file in which all the descriptor files from different processes have been merged.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    desc_file = os.path.normpath(os.path.join(desc_folder, desc_file))

    logger.info("Collecting descriptor folder: {0}".format(desc_folder))

    desc_file_with_ext = os.path.normpath(os.path.join(desc_folder, desc_file))
    tar = tarfile.open(desc_file_with_ext, 'w:gz')

    tmp_files = []
    desc_files = []

    for i in range(nb_jobs):
        desc_file_i = desc_file + '_' + str(i) + '.tar.gz'
        desc_files.append(desc_file_i)

        archive_i = tarfile.open(desc_file_i, 'r')
        archive_i.extractall(tmp_folder)

        for member in archive_i.getmembers():
            # for file list we need member names only
            # because we need to erase the tmp files in desc_folder
            tmp_file_no_path = member.name.rsplit("/")[-1]
            tmp_file = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, tmp_file_no_path)))
            tmp_files.append(tmp_file)

            member.name = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, member.name)))

            # do not add the full path as name in the tar.gz file
            tar.add(member.name, arcname=tmp_file_no_path)

    tar.close()

    desc_file_master = write_summary_file(descriptor, desc_file, tmp_folder,
                                          desc_file_master=desc_file_with_ext + '.tar.gz', clean_tmp=False)

    if remove:
        for i in range(nb_jobs):
            desc_file_i = desc_file + '_' + str(i) + '.tar.gz'
            os.remove(desc_file_i)
        for tmp_f in tmp_files:
            try:
                os.remove(tmp_f)
            except OSError as os_e:
                logger.debug("Could not remove file {0}. {1}".format(tmp_f, os_e))
            except Exception as e:
                logger.error("Could not remove file: {0}. Exception: {1}".format(tmp_f, e.__class__.__name__))

    return desc_file_master


# def worker_apply_operations(arg):
#     ase_atoms, operations_on_structure = arg
#     return _apply_operations(ase_atoms, operations_on_structure)

def dispatch_jobs(function_to_calc, data, nb_jobs, desc_folder, desc_file):
    """ Dispatch the calculation of `function_to_calc` to `nb_jobs` parallel processes.

    `function_to_calc` is applied to `data`. `data` is split as evenly as possible across processes.
    Each process will create a descriptor file named `desc_file_job_nb` where `job_nb` is process
    number in the processing pool. These `nb_jobs` descriptor files are saved in the `desc_folder` folder.

    Parameters:

    function_to_calc: `ai4materials.descriptors.Descriptor`
        Function to apply to the 'data' list.

    data: list of `ase.Atoms`
        List of atomic structures that it is going to be split across processors.

    desc_file: str
        Name of the file where the descriptor is going to be saved.

    desc_folder: str
        Path to the descriptor folder.

    nb_jobs: int
        Number of parallel jobs to dispatch.

    .. codeauthor:: Angelo Ziletti <angelo.ziletti@gmail.com>

    """

    slices = split_list(data, nb_jobs)
    jobs = []

    # https://stackoverflow.com/questions/15536295/python-multiprocessing-process-crashes-silently

    desc_file = os.path.normpath(os.path.join(desc_folder, desc_file))

    # https://pymotw.com/3/multiprocessing/basics.html

    # with concurrent.futures.ProcessPoolExecutor(max_workers=nb_jobs) as executor:
    #     print("using concurrent features")
    #     ase_atoms_list_with_op_nested = executor.map(worker_apply_operations,
    #                                                  ((ase_atoms, operations_on_structure) for ase_atoms in
    #                                                   ase_atoms_list))
    #


    # for idx_slice, slice_ in enumerate(slices):
    #     desc_file_i = desc_file + '_' + str(idx_slice) + '.tar.gz'
    #     multiprocessing.log_to_stderr(logging.DEBUG)
    #     job = multiprocessing.Process(target=function_to_calc, args=(slice_, desc_file_i, idx_slice))
    #     jobs.append(job)
    #
    # for job in jobs:
    #     job.start()
    #
    # for job in jobs:
    #     job.join()

    import time

    # data_pairs = [[3, 5], [4, 3], [7, 3], [1, 6]]
    #
    # import numpy as np
    # # define what to do with each data pair ( p=[3,5] ), example: calculate product
    # def myfunc(p):
    #     product_of_list = np.prod(p)
    #     return product_of_list
    #
    # pool = multiprocessing.Pool(processes=4)
    # result_list = pool.map(myfunc, data_pairs)
    # print(result_list)


    # desc_file_i = desc_file + '_' + str(idx_slice) + '.tar.gz'
    # job = multiprocessing.Process(target=function_to_calc, args=(slice_, desc_file_i, idx_slice))
    # jobs.append(job)
    #

    # import time
    # import concurrent.futures
    #
    # def sleep_print_return(input):
    #     print("{} started".format(input))
    #     time.sleep(input % 3)
    #     return input
    #
    # with concurrent.futures.ProcessPoolExecutor(max_workers=None) as executor:
    #     for result in executor.map(sleep_print_return, range(25)):
    #         # do stuff
    #         print("{} finished".format(result))
    #         pass



        # for result in executor.map(sleep_print_return, range(25)):
            # do stuff
            # print
            # "{} finished".format(result)
            # pass


