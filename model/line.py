import argparse
import os
import random
import time

import dgl

import numpy as np
import torch
import torch.multiprocessing as mp
from .line_model import SkipGramModel
from .line_dataset import LineDataset
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
import torch.optim as optim
import time
from runx.logx import logx
from tqdm import tqdm
from os.path import join as ospj


class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'LINE'
        self.graph = None                                               # 用于训练的图
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.seed = 1                                                   # seed

        # self.dropout = 0.5                                              # 随机失活
        self.walk_length = 10                                           # 游走路径长度
        self.num_walks = 400                                             # 平均每个节点生成的游走路径数量
        # self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                          # mini-batch大小

        # line参数
        self.need_1st = True
        self.need_2nd = True
        self.negative = 3                                               # negative samples for each positve node pair
        self.neg_weight = 1.0                                           # 负样本的梯度权重
        self.fast_neg = True                                           # do negative sampling inside a batch

        self.embed_size = 128                                           # 向量的维度
        self.learning_rate = 0.025                                      # 学习率 alpha
        self.workers = 3                                                # 线程数


class Model:
    """LINE"""
    def __init__(self, config):
        """Initializing the trainer with the input arguments"""
        self.config = config
        self.dataset = LineDataset(
            batch_size=config.batch_size,
            negative=config.negative,
            fast_neg=config.fast_neg,
            graph=config.graph,
            num_walks=config.num_walks * len(config.graph.nodes()),
        )
        self.emb_size = self.dataset.G.num_nodes()
        self.emb_model = None

    def init_device_emb(self):
        """set the device before training
        will be called once in fast_train_mp / fast_train
        """

        # initializing embedding on CPU
        self.emb_model = SkipGramModel(
            emb_size=self.emb_size,
            emb_dimension=self.config.embed_size,
            batch_size=self.config.batch_size,
            need_1st=self.config.need_1st,
            need_2nd=self.config.need_2nd,
            neg_weight=self.config.neg_weight,
            negative=self.config.negative,
            lr=self.config.learning_rate,
            fast_neg=self.config.fast_neg,
        )

        print("Run in 1 GPU")
        self.emb_model.all_to_device(0)

    def train(self, config):
        """fast train with dataloader with only gpu / only cpu"""
        self.init_device_emb()

        sampler = self.dataset.create_sampler(0)

        dataloader = DataLoader(
            dataset=sampler.seeds,
            batch_size=self.config.batch_size,
            collate_fn=sampler.sample,
            shuffle=False,
            drop_last=False,
        )

        num_batches = len(dataloader)
        print("num batchs: %d\n" % num_batches)

        start_all = time.time()
        localtime = time.asctime(time.localtime(time.time()))
        logx.msg('======================Start Train Model [{}]======================'.format(localtime))
        with torch.no_grad():
            for i, edges in enumerate(tqdm(dataloader)):
                if self.config.fast_neg:
                    self.emb_model.fast_learn(edges)
                else:
                    # do negative sampling
                    bs = edges.size()[0]
                    neg_nodes = torch.LongTensor(
                        np.random.choice(
                            self.dataset.neg_table,
                            bs * self.config.negative,
                            replace=True,
                        )
                    )
                    self.emb_model.fast_learn(edges, neg_nodes=neg_nodes)

        print("Training used time: %.2fs" % (time.time() - start_all))
        localtime = time.asctime(time.localtime(time.time()))
        logx.msg('======================Finish Train Model [{}]======================'.format(localtime))

    def save_embeddings(self, config, emb_path):
        self.emb_model.save_embedding_pt(self.dataset, emb_path)
