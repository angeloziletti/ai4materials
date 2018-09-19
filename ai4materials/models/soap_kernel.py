#! /usr/bin/env python
from __future__ import absolute_import

__author__ = "Carl Poelking"
__copyright__ = "Copyright 2017, The NOMAD Project"
__maintainer__ = "Carl Poelking"
__email__ = "cp605@cam.ac.uk"
__date__ = "06/04/16"

import soap
import sys
import ase.io
import json
import numpy as np
import h5py
import os
import logging

from ai4materials.wrappers import plot, logger


logger.setLevel(logging.INFO)
dimred_method = 'kernelpca'
#dimred_method = 'mds'
logger.info("Dim.-red. method: %s" % dimred_method)

tmp_folder = '/home/beaker/.beaker/v1/web/tmp/'
control_file = '/home/beaker/.beaker/v1/web/tmp/control.json'
lookup_file = '/home/beaker/.beaker/v1/web/tmp/lookup.dat'


# ================
# KERNEL FUNCTIONS
# ================

class BaseKernelDot(object):
    def __init__(self, options):
        self.xi = options['xi']
        self.delta = options['delta']
        return

    def compute(self, X, Y):
        return self.delta**2 * X.dot(Y.T)**self.xi


class TopKernelRematch(object):
    def __init__(self, options, basekernel):
        self.gamma = options['gamma']
        self.basekernel = basekernel
        return

    def compute(self, g1, g2, log=None):
        if log:
            log << "[Kernel] %-15s %-15s" % (g1.graph_info['label'], g2.graph_info['label']) << log.endl
        K_base = self.basekernel.compute(g1.P, g2.P)
        # Only works with float64 (due to C-casts?) ...
        if K_base.dtype != 'float64':
            K_base = K_base.astype('float64')
        k_top = soap.linalg.regmatch(K_base, self.gamma, 1e-6)
        return k_top


class TopKernelAverage(object):
    def __init__(self, options, basekernel):
        self.basekernel = basekernel
        return

    def compute(self, g1, g2, log=None):
        p1_avg = g1.P_avg
        p2_avg = g2.P_avg
        return self.basekernel.compute(p1_avg, p2_avg)


BaseKernelFactory = {
    'dot': BaseKernelDot
}

TopKernelFactory = {
    'rematch': TopKernelRematch,
    'average': TopKernelAverage
}

# ==============
# READ/FILTER IO
# ==============


def read_filter_configs(
        config_file,
        index=':',
        filter_types=None,
        types=[],
        do_remove_duplicates=False,
        key=lambda c: c.info['label'],
        log=None):
    if log:
        log << "Reading" << config_file << log.endl
    configs = ase.io.read(config_file, index=index)
    if log:
        log << log.item << "Have %d initial configurations" % len(configs) << log.endl
    if do_remove_duplicates:
        configs, duplics = remove_duplicates(configs, key=key)
        if log:
            log << log.item << "Removed %d duplicates" % len(duplics) << log.endl
    if filter_types:
        configs_filtered = []
        for config in configs:
            types_config = config.get_chemical_symbols()
            keep = True
            for t in types_config:
                if not t in types:
                    keep = False
                    break
            if keep:
                configs_filtered.append(config)
        configs = configs_filtered
        if log:
            log << log.item << "Have %d configurations after filtering" % len(configs) << log.endl
    return configs


def remove_duplicates(array, key=lambda a: a):
    len_in = len(array)
    label = {}
    array_curated = []
    array_duplicates = []
    for a in array:
        key_a = key(a)
        if key_a in label:
            array_duplicates.append(a)
        else:
            array_curated.append(a)
            label[key_a] = True
    len_out = len(array_curated)
    return array_curated, array_duplicates

# =================
# GRAPH CALCULATION
# =================


class Graph(object):
    def __init__(self,
                 idx=-1,
                 label='',
                 feature_mat=None,
                 feature_mat_avg=None,
                 position_mat=None,
                 connectivity_mat=None,
                 vertex_info=[],
                 graph_info={}):
        # Labels
        self.idx = idx
        self.label = label
        self.graph_info = graph_info
        # Vertex data: descriptors
        self.P = feature_mat
        self.P_avg = feature_mat_avg
        self.P_type_str = str(type(self.P))
        # Vertex data: positions, labels
        self.R = position_mat
        self.vertex_info = vertex_info
        # Edge data
        self.C = connectivity_mat
        return

    def save_to_h5(self, h5f, dtype='float32'):
        group = h5f.create_group('%06d' % self.idx)
        if self.P_type_str == "<class 'soap.soapy.kernel.DescriptorMapMatrix'>":
            # Save list of descriptor maps
            g0 = group.create_group('feature_dmap')
            for dmap_idx, dmap in enumerate(self.P):
                g1 = g0.create_group('%d' % dmap_idx)
                for key in dmap:
                    g1.create_dataset(
                        key,
                        data=dmap[key],
                        compression='gzip',
                        dtype=dtype)
            # Save averaged descriptor map
            g0_avg = group.create_group('feature_dmap_avg')
            for key in self.P_avg:
                g0_avg.create_dataset(
                    key,
                    data=self.P_avg[key],
                    compression='gzip',
                    dtype=dtype)
        elif self.P_type_str == "<type 'numpy.ndarray'>":
            # Save numpy arrays
            group.create_dataset(
                'feature_mat',
                data=self.P,
                compression='gzip',
                dtype=dtype)
            group.create_dataset(
                'feature_mat_avg',
                data=self.P_avg,
                compression='gzip',
                dtype=dtype)
        else:
            raise NotImplementedError(self.P_type_str)
        group.create_dataset('position_mat', data=self.R)
        group.create_dataset('connectivity_mat', data=self.C)
        group.attrs['idx'] = self.idx
        group.attrs['label'] = self.label
        group.attrs['vertex_info'] = json.dumps(self.vertex_info)
        group.attrs['graph_info'] = json.dumps(self.graph_info)
        group.attrs['P_type_str'] = self.P_type_str
        return

    def load_from_h5(self, h5f):
        self.idx = h5f.attrs['idx']
        self.label = h5f.attrs['label']
        self.vertex_info = json.loads(h5f.attrs['vertex_info'])
        self.graph_info = json.loads(h5f.attrs['graph_info'])
        self.P_type_str = h5f.attrs['P_type_str']
        if self.P_type_str == "<class 'soap.soapy.kernel.DescriptorMapMatrix'>":
            # Load list of descriptor maps
            self.P = soap.soapy.kernel.DescriptorMapMatrix()
            g0 = h5f['feature_dmap']
            for i in range(len(g0)):
                Pi = soap.soapy.kernel.DescriptorMap()
                g1 = g0['%d' % i]
                for key in g1:
                    Pi[key] = g1[key].value
                self.P.append(Pi)
            # Load averaged descriptor map
            self.P_avg = soap.soapy.kernel.DescriptorMap()
            g0_avg = h5f['feature_dmap_avg']
            for key in g0_avg:
                self.P_avg[key] = g0_avg[key].value
        elif self.P_type_str == "<type 'numpy.ndarray'>":
            self.P = h5f['feature_mat'].value
            self.P_avg = h5f['feature_mat_avg'].value
        else:
            raise NotImplementedError(self.P_type_str)
        self.R = h5f['position_mat'].value
        self.C = h5f['connectivity_mat'].value
        return self


def compute_graph(
        config,
        descriptor_options,
        log):
    if log:
        soap.soapy.util.MP_LOCK.acquire()
        log << log.back << "[Graph] %-15s" % config.info['label'] << log.endl
        soap.soapy.util.MP_LOCK.release()
    # Config => struct + connectivity matrices
    config, struct, top, frag_bond_mat, atom_bond_mat, frag_labels, atom_labels = \
        soap.tools.structure_from_ase(
            config,
            do_partition=False,
            add_fragment_com=False,
            log=None)
    # Struct => spectrum
    feature_mat, feature_mat_avg, position_mat, type_vec = compute_soap(
        struct,
        descriptor_options)
    # Config, spectrum => graph
    connectivity_mat = atom_bond_mat
    vertex_info = atom_labels
    graph = Graph(
        idx=config.info['idx'],
        label=str(config.info['label']),
        feature_mat=feature_mat,
        feature_mat_avg=feature_mat_avg,
        position_mat=position_mat,
        connectivity_mat=connectivity_mat,
        vertex_info=vertex_info,
        graph_info=config.info)
    return graph


def compute_soap(struct, options):
    # OPTIONS
    options_soap = soap.Options()
    for key, val in options.items():
        if isinstance(val, list):
            continue
        options_soap.set(key, val)
    # Exclusions
    excl_targ_list = options['exclude_targets']
    excl_cent_list = options['exclude_centers']
    options_soap.excludeCenters(excl_cent_list)
    options_soap.excludeTargets(excl_targ_list)
    # Compute spectrum
    spectrum = soap.Spectrum(struct, options_soap)
    spectrum.compute()
    spectrum.computePower()
    if options['spectrum.gradients']:
        spectrum.computePowerGradients()
    if options['spectrum.global']:
        spectrum.computeGlobal()
    # Adapt spectrum
    adaptor = soap.soapy.kernel.KernelAdaptorFactory[options['kernel.adaptor']](
        options_soap,
        types_global=options['type_list'])
    IX, IR, types = adaptor.adapt(spectrum, return_pos_matrix=True)
    # Calculate average
    X_avg = IX.sum(axis=0)
    X_avg = X_avg / X_avg.dot(X_avg)**0.5
    return IX, X_avg, IR, types

# ========
# MAIN EXE
# ========

# added:


def calc_soap_similarity(ase_atoms_list, tmp_folder, h5_filename, options):

    h5_file = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, h5_filename)))

    log = soap.soapy.momo.osio

    # Setup HDF5 storage
    h5 = h5py.File(h5_file, 'w')

    # ================
    # Read ASE configs
    # ================

    # TODO Load structures here as <list(ase.atoms)>
    # configs = read_filter_configs(
    #    'configs.xyz',
    #    index=':',
    #    filter_types=False,
    #    types=[],
    #    do_remove_duplicates=False,
    #    key=lambda c: c.info['label'],
    #    log=log)

    # LOAD DATA, CONVERT TO ASE

    # ==============
    # Compute graphs
    # ==============

    log << log.mg << "Computing graphs ..." << log.endl
    # Options
    descriptor_type = options['descriptor']['type']
    descriptor_options = options['descriptor'][descriptor_type]
    log << "Descriptor" << descriptor_type << json.dumps(
        descriptor_options, indent=2, sort_keys=True) << log.endl
    # Compute
    graphs = [
        compute_graph(
            config=config,
            descriptor_options=descriptor_options,
            log=log
        ) for config in ase_atoms_list
    ]
    # Store
    h5_graphs = h5.create_group('/graphs')
    h5.attrs['descriptor_options'] = json.dumps(descriptor_options)

    for g in graphs:
        g.save_to_h5(h5_graphs)

    # Optional: Save labels
    labels = np.zeros((len(h5_graphs),), dtype=[('idx', 'i8'), ('tag', 'a32')])
    for g in h5_graphs.iteritems():
        idx = int(g[0])
        tag = g[1].attrs['label']
        g_info = json.loads(g[1].attrs['graph_info'])
        labels[idx] = (idx, tag)
    h5_labels = h5.create_group('labels')
    h5_labels.create_dataset('label_mat', data=labels)

    # ==============
    # COMPUTE KERNEL
    # ==============

    soap.silence()

    log << log.mg << "Computing kernel ..." << log.endl
    # Options
    basekernel_type = options["basekernel"]["type"]
    basekernel_options = options["basekernel"][basekernel_type]
    basekernel = BaseKernelFactory[basekernel_type](basekernel_options)
    topkernel_type = options["topkernel"]["type"]
    topkernel_options = options["topkernel"][topkernel_type]
    topkernel = TopKernelFactory[topkernel_type](topkernel_options, basekernel)
    log << "Base-kernel" << basekernel_type << json.dumps(
        basekernel_options, indent=2, sort_keys=True) << log.endl
    log << "Top-kernel" << topkernel_type << json.dumps(
        topkernel_options, indent=2, sort_keys=True) << log.endl
    # (Re-)load graphs
    graphs = [
        Graph().load_from_h5(h5_graphs['%06d' % i])
        for i in range(len(h5_graphs))
    ]
    # Compute pair-wise kernel entries
    kmat = np.zeros((len(graphs), len(graphs)), dtype='float32')
    for i in range(len(graphs)):
        for j in range(i, len(graphs)):
            kmat[i, j] = topkernel.compute(graphs[i], graphs[j], log)

    kmat = kmat + kmat.T
    np.fill_diagonal(kmat, kmat.diagonal() * 0.5)
    # Store
    h5_kernel = h5.create_group('kernel')
    h5_kernel.create_dataset('kernel_mat', data=kmat)
    h5.close()

    ###
    kmat_new = kmat.copy()
    np.fill_diagonal(kmat_new, 0.5 * kmat_new.diagonal())
    # logger.debug(kmat[0:7,0:7])
    dmat = (1. - kmat**2 + 1e-10)**0.5

    return kmat, dmat

    # Run
    # log.cd("/home/beaker/tutorials/nomad-sim-map/configs")
    # soap.silence()
    #run(log=log, cmdline_options=cmdline_options, json_options=json_options)
    #run(log=log, json_options=json_options)
    # log.root()


def calc_soap_cross_similarity(ase_atoms_list_rows, ase_atoms_list_cols, tmp_folder, h5_filename, options):

    h5_file = os.path.abspath(os.path.normpath(os.path.join(tmp_folder, h5_filename)))

    log = soap.soapy.momo.osio

    # Setup HDF5 storage
    h5 = h5py.File(h5_file, 'w')

    # ==============
    # Compute graphs
    # ==============

    log << log.mg << "Computing graphs ..." << log.endl
    # Options
    descriptor_type = options['descriptor']['type']
    descriptor_options = options['descriptor'][descriptor_type]
    log << "Descriptor" << descriptor_type << json.dumps(
        descriptor_options, indent=2, sort_keys=True) << log.endl
    # Compute
    graphs_rows = [
        compute_graph(
            config=config,
            descriptor_options=descriptor_options,
            log=log
        ) for config in ase_atoms_list_rows
    ]
    graphs_cols = [
        compute_graph(
            config=config,
            descriptor_options=descriptor_options,
            log=log
        ) for config in ase_atoms_list_cols
    ]
    # Store
    h5_graphs_rows = h5.create_group('/graphs-rows')
    h5_graphs_cols = h5.create_group('/graphs-cols')
    h5.attrs['descriptor_options'] = json.dumps(descriptor_options)

    for g in graphs_rows:
        g.save_to_h5(h5_graphs_rows)
    for g in graphs_cols:
        g.save_to_h5(h5_graphs_cols)

    # Optional: Save labels
    labels_rows = np.zeros((len(h5_graphs_rows),), dtype=[('idx', 'i8'), ('tag', 'a32')])
    labels_cols = np.zeros((len(h5_graphs_cols),), dtype=[('idx', 'i8'), ('tag', 'a32')])
    for g in h5_graphs_rows.iteritems():
        idx = int(g[0])
        tag = g[1].attrs['label']
        g_info = json.loads(g[1].attrs['graph_info'])
        labels_rows[idx] = (idx, tag)
    for g in h5_graphs_cols.iteritems():
        idx = int(g[0])
        tag = g[1].attrs['label']
        g_info = json.loads(g[1].attrs['graph_info'])
        labels_cols[idx] = (idx, tag)
    h5_labels_rows = h5.create_group('labels-rows')
    h5_labels_cols = h5.create_group('labels-cols')
    h5_labels_rows.create_dataset('label_mat', data=labels_rows)
    h5_labels_cols.create_dataset('label_mat', data=labels_cols)

    # ==============
    # COMPUTE KERNEL
    # ==============

    soap.silence()

    log << log.mg << "Computing kernel ..." << log.endl
    # Options
    basekernel_type = options["basekernel"]["type"]
    basekernel_options = options["basekernel"][basekernel_type]
    basekernel = BaseKernelFactory[basekernel_type](basekernel_options)
    topkernel_type = options["topkernel"]["type"]
    topkernel_options = options["topkernel"][topkernel_type]
    topkernel = TopKernelFactory[topkernel_type](topkernel_options, basekernel)
    log << "Base-kernel" << basekernel_type << json.dumps(
        basekernel_options, indent=2, sort_keys=True) << log.endl
    log << "Top-kernel" << topkernel_type << json.dumps(
        topkernel_options, indent=2, sort_keys=True) << log.endl
    # (Re-)load graphs
    graphs_rows = [
        Graph().load_from_h5(h5_graphs_rows['%06d' % i])
        for i in range(len(h5_graphs_rows))
    ]
    graphs_cols = [
        Graph().load_from_h5(h5_graphs_cols['%06d' % i])
        for i in range(len(h5_graphs_cols))
    ]
    # Compute pair-wise kernel entries
    kmat = np.zeros((len(graphs_rows), len(graphs_cols)), dtype='float32')
    for i in range(len(graphs_rows)):
        for j in range(len(graphs_cols)):
            kmat[i, j] = topkernel.compute(graphs_rows[i], graphs_cols[j], log)

    # Store
    h5_kernel = h5.create_group('kernel')
    h5_kernel.create_dataset('kernel_mat', data=kmat)
    h5.close()

    kmat_new = kmat.copy()
    dmat = (1. - kmat**2 + 1e-10)**0.5
    return kmat, dmat
