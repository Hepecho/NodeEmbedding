import argparse
import torch
import networkx as nx
import matplotlib.pyplot as plt
from os.path import join as ospj
from sklearn.manifold import TSNE
from load_dataset import *


def plot_graph(src, dst, node_pos):
    g = nx.Graph()
    src = src.numpy()
    dst = dst.numpy()
    edgelist = zip(src, dst)
    for i, j in edgelist:
        g.add_edge(i, j)
    # pos_3d = nx.spring_layout(g, dim=3, k=0.5)
    nx.draw(g, with_labels=g.nodes, pos=node_pos)
    # plt.savefig('test.png')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='graph visualization')
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

    emb_path = ospj(args.emb_dir, args.dataset, args.model + '.pt')
    emb_tensor = torch.load(emb_path, map_location='cpu')
    # emb_list = emb_tensor.tolist()

    model = TSNE(n_components=2)
    node_pos = model.fit_transform(emb_tensor)  # numpy list [[123,432], [156, 66],...]
    # print(node_pos[:10])
    # exit()

    if args.dataset == "ogbl-ddi":
        g = link_prediction_data()
        # print(g.edges())
        src, dst = g.edges()
        # 无向图删除一半边
        src = src[::2]
        dst = dst[::2]
        node_list = g.nodes().tolist()
        node_pos = dict(zip(node_list, node_pos))

        # print(node_pos[4000])
        plot_graph(src, dst, node_pos)
        # print(config.graph)
        # 在训练embedding阶段，我们采用word2vec的无监督任务
    elif args.dataset == "ogbn-arxiv":
        g, labels = node_prediction_data()

    else:
        pass
