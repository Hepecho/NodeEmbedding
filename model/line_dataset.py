import os
import pickle
import random
import time

import dgl

import numpy as np
import scipy.sparse as sp
import torch
from dgl.data.utils import (
    _get_dgl_url,
    download,
    extract_archive,
    get_download_dir,
)
from torch.utils.data import DataLoader
from ogb.linkproppred import DglLinkPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset


def find_connected_nodes(G):
    nodes = torch.nonzero(G.out_degrees(), as_tuple=False).squeeze(-1)
    # 输出数组的非零值的索引
    return nodes


class LineDataset:
    def __init__(
        self,
        batch_size,
        num_walks,
        negative=5,
        fast_neg=True,
        graph=None
    ):
        """This class has the following functions:
        1. Transform the txt network file into DGL graph;
        2. Generate random walk sequences for the trainer;
        3. Provide the negative table if the user hopes to sample negative
        nodes according to nodes' degrees;

        Parameter
        ---------
        walk_length int : number of nodes in a sequence
        window_size int : context window size
        num_walks int : number of walks for each node
        batch_size int : number of node sequences in each batch
        negative int : negative samples for each positve node pair
        fast_neg bool : whether do negative sampling inside a batch
        """
        self.batch_size = batch_size
        self.negative = negative
        self.num_walks = num_walks
        self.fast_neg = fast_neg
        self.G = graph

        print("Finish reading graph")

        self.num_nodes = self.G.num_nodes()

        start = time.time()
        seeds = np.random.choice(
            np.arange(self.G.num_edges()), self.num_walks, replace=True
        )  # edge index
        self.seeds = [torch.LongTensor(seeds)]
        # print(self.seeds.shape)
        # self.seeds = torch.split(torch.LongTensor(seeds), self.num_walks, 0)
        # print(self.seeds.shape)
        end = time.time()
        t = end - start
        print("generate %d samples in %.2fs" % (len(seeds), t))

        # negative table for true negative sampling
        self.valid_nodes = find_connected_nodes(self.G)
        if not fast_neg:
            node_degree = self.G.out_degrees(self.valid_nodes).numpy()
            node_degree = np.power(node_degree, 0.75)
            node_degree /= np.sum(node_degree)
            node_degree = np.array(node_degree * 1e8, dtype=int)
            self.neg_table = []

            for idx, node in enumerate(self.valid_nodes):
                self.neg_table += [node] * node_degree[idx]
            self.neg_table_size = len(self.neg_table)
            self.neg_table = np.array(self.neg_table, dtype=np.int64)
            del node_degree

    def create_sampler(self, i):
        """create random walk sampler"""
        return EdgeSampler(self.G, self.seeds[i])

    def save_mapping(self, map_file):
        with open(map_file, "wb") as f:
            pickle.dump(self.node2id, f)


class EdgeSampler(object):
    def __init__(self, G, seeds):
        self.G = G
        self.seeds = seeds
        self.edges = torch.cat(
            (self.G.edges()[0].unsqueeze(0), self.G.edges()[1].unsqueeze(0)), 0
        ).t()

    def sample(self, seeds):
        """seeds torch.LongTensor : a batch of indices of edges"""
        return self.edges[torch.LongTensor(seeds)]
