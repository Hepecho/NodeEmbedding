import numpy as np
import argparse
import time
import torch

import dgl
import matplotlib.pyplot as plt
import networkx as nx
import os
from os.path import join as ospj
from sklearn.manifold import TSNE
from load_dataset import *
from importlib import import_module
from runx.logx import logx


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train embedding')
    # parser.add_argument('--cuda',  action='store_true',
    #                     help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='ogbl-ddi',
                        help='dataset, choose from ["ogbl-ddi", "ogbn-arxiv"]')
    parser.add_argument('--emb_dir', type=str, default='saved_model',
                        help='saved model dir')
    parser.add_argument('--model', type=str, default='deepwalk',
                        help='model in ["deepwalk", "line", "node2vec"]')
    parser.add_argument('--logdir', type=str, default='log',
                        help='target log directory')
    args = parser.parse_args()
    # print(args)

    dataset_choices = ["ogbl-ddi", "ogbn-arxiv"]
    # 分别对应两个基础任务 link_prediction node_prediction_data
    assert args.dataset in dataset_choices, "dataset name must be selected from " + str(dataset_choices)

    os.makedirs(ospj(args.emb_dir, args.dataset), exist_ok=True)
    emb_path = ospj(args.emb_dir, args.dataset, args.model + '.pt')
    logx.initialize(logdir=ospj(args.logdir, args.model), coolname=False, tensorboard=False)
    logx.msg(str(args))

    import sys
    sys.path.insert(0, sys.path[0] + "/../")
    x = import_module('model.' + args.model)
    config = x.Config(args.dataset)
    logx.msg(str(config.__dict__))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)   
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    if args.dataset == "ogbl-ddi":
        config.graph = link_prediction_data()
        # print(config.graph)
        # 在训练embedding阶段，我们采用word2vec的无监督任务
    elif args.dataset == "ogbn-arxiv":
        G, labels = node_prediction_data()
        if args.model == 'deepwalk':
            # 变成无向图 为每条边都创建两个方向的边
            G = dgl.to_bidirected(G)
        config.graph = G
    else:
        pass

    model = x.Model(config)
    model.train(config)
    model.save_embeddings(config, emb_path)

