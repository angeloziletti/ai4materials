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
__date__ = "20/04/18"

if __name__ == "__main__":
    import sys
    import os.path
    import pandas as pd

    # filename_in = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/results_paper/inclusion/inclusion_fcc_bcc_pristine_baa.xyz'
    # filename_out = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/results_paper/inclusion/inclusion_fcc_bcc_pristine_baa_fcc_only.xyz'

    # filename_in = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/results_paper/inclusion/inclusion_fcc_bcc_vac50_baa.xyz'
    # filename_out = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/results_paper/inclusion/inclusion_fcc_bcc_vac50_baa_fcc_only.xyz'

    # filename_in = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/results_paper/inclusion/inclusion_fcc_bcc_pristine_ptm.xyz'
    # filename_out = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/results_paper/inclusion/inclusion_fcc_bcc_pristine_ptm_fcc_only.xyz'

    filename_in = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/results_paper/inclusion/inclusion_fcc_bcc_vac50_ptm.xyz'
    filename_out = '/home/ziletti/Documents/calc_nomadml/rot_inv_3d/results_paper/inclusion/inclusion_fcc_bcc_vac50_ptm_fcc_only.xyz'

    with open(filename_in) as f:
        content = f.readlines()  # you may also want to remove whitespace characters like `\n` at the end of each line
        content = [x.strip() for x in content]

    header = content[:2]
    content_split = [item.split() for item in content[2:]]
    # convert everything apart from the first two lines in a panda dataframe
    df = pd.DataFrame(content_split, columns=['atom_type', 'x', 'y', 'z', 'particle_id', 'structure_id'])

    df_fcc_only = df[df['structure_id'] == '1']
    df_fcc_only = df_fcc_only.drop('particle_id', axis=1)
    df_fcc_only = df_fcc_only.drop('structure_id', axis=1)

    fcc_only_structures = df_fcc_only.values.tolist()

    new_list = []
    for item in fcc_only_structures:
        new_list.append(" ".join(str(x) for x in item))

    header[0] = str(len(df_fcc_only))
    header[1] = ""

    with open(filename_out, 'w') as f_out:
        for line in header:
            f_out.writelines(line + '\n')
        for line in new_list:
            f_out.writelines(line + '\n')

        f_out.close()

    print("Done")
