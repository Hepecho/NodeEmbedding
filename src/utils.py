import numpy as np
import torch
import torch.nn as nn
import dgl.function as fn


class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h是从5.1节的GNN模型中计算出的节点表示
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']


def make_undirected(G):
    G.add_edges(G.edges()[1], G.edges()[0])
    return G
