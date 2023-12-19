import random
import torch
from tqdm import tqdm
import dgl
from dgl import NodeShuffle
from dgl.sampling import random_walk, sample_neighbors
from .alias import alias_sample, create_alias_table
import torch.utils.data as data


class DeepWalker:
    def __init__(self, G):
        self.G = G

    def simulate_walks(self, num_walks, walk_length):
        g = self.G
        transform = NodeShuffle()
        all_walks = None
        for _ in range(num_walks):
            g = transform(g)
            walks, _ = random_walk(g, list(g.nodes()), length=walk_length-1)
            for walk in walks:
                if torch.tensor(-1) in walk:
                    print(walk)
            if all_walks is None:
                all_walks = walks
            else:
                all_walks = torch.cat((all_walks, walks), 0)

        return all_walks


class Node2VecWalker:
    def __init__(self, G, p=1, q=1):
        self.G = G
        self.p = p
        self.q = q
        self.fir_nbrs = None
        self.alias_edges = None

    def get_fir_nbrs(self):
        G = self.G
        # 计算每个节点的一阶邻居
        inverted_G = dgl.reverse(G)  # 由于sample_neighbors 只计算入边邻居，所以先做反转
        fir_nbrs = []
        node_list = sorted(list(inverted_G.nodes()))
        for node in node_list:
            # 虽然sample_neighbors支持传入node列表，但是这样得到的邻居节点会汇总成一个列表，需要分割
            subgraph = sample_neighbors(inverted_G, node, -1)  # -1表示取所有邻居
            fir_nbrs.append(subgraph.edges()[0].tolist())  # 一阶邻居
        return fir_nbrs

    def get_alias_edges(self):
        """
        对图G的所有边，充当(t, v)，生成对应的accept和alias数组
        :return: alias_edges dict类型，key为元组(t, v)，value为accept、alias数组以及idx2node构成的三维列表
                idx2node: list类型，是accept和alias数组idx到node的id的映射
        """
        G = self.G
        alias_edges = {}
        t_list, v_list = G.edges()
        print(len(t_list))
        for t, v in tqdm(zip(t_list, v_list)):
            t = t.item()
            v = v.item()
            t_nbrs_1st = self.fir_nbrs[t]
            # d_tx == 0
            unnormalized_area_ratio = [1/self.p]
            idx2node = [t]
            if v in t_nbrs_1st:
                t_nbrs_1st.remove(v)
            alpha_pq = 1
            pi_vx = alpha_pq  # 本次实验的图都是无权图，所以这里权重取1
            # d_tx == 1
            idx2node.extend(t_nbrs_1st)
            unnormalized_area_ratio.extend([pi_vx] * len(t_nbrs_1st))

            t_nbrs_2nd = []
            for nbr_1st in t_nbrs_1st:
                t_nbrs_2nd.extend(self.fir_nbrs[nbr_1st])

            if v in t_nbrs_2nd:
                t_nbrs_2nd.remove(v)
            t_nbrs_2nd = [nbr_2nd for nbr_2nd in t_nbrs_2nd if nbr_2nd not in idx2node]
            alpha_pq = 1 / self.q
            pi_vx = alpha_pq  # 本次实验的图都是无权图，所以这里权重取1
            # d_tx == 2
            idx2node.extend(t_nbrs_2nd)
            unnormalized_area_ratio.extend([pi_vx] * len(t_nbrs_2nd))

            total_pro = sum(unnormalized_area_ratio)
            area_ratio = [pro/total_pro for pro in unnormalized_area_ratio]
            accept, alias = create_alias_table(area_ratio)
            alias_edges[(t, v)] = [accept, alias, idx2node]

        # t == v:
        for node in G.nodes():
            t = node.item()
            t_nbrs_1st = self.fir_nbrs[t]

            # d_tx == 0 不存在，因为相当于原地不动
            unnormalized_area_ratio = []
            idx2node = []

            alpha_pq = 1
            pi_vx = alpha_pq  # 本次实验的图都是无权图，所以这里权重取1
            # d_tx == 1
            idx2node.extend(t_nbrs_1st)
            unnormalized_area_ratio.extend([pi_vx] * len(t_nbrs_1st))

            t_nbrs_2nd = []
            for nbr_1st in t_nbrs_1st:
                t_nbrs_2nd.extend(self.fir_nbrs[nbr_1st])

            if t in t_nbrs_2nd:
                t_nbrs_2nd.remove(t)
            t_nbrs_2nd = [nbr_2nd for nbr_2nd in t_nbrs_2nd if nbr_2nd not in idx2node]
            alpha_pq = 1 / self.q
            pi_vx = alpha_pq  # 本次实验的图都是无权图，所以这里权重取1
            # d_tx == 2
            idx2node.extend(t_nbrs_2nd)
            unnormalized_area_ratio.extend([pi_vx] * len(t_nbrs_2nd))

            total_pro = sum(unnormalized_area_ratio)
            area_ratio = [pro / total_pro for pro in unnormalized_area_ratio]
            accept, alias = create_alias_table(area_ratio)
            alias_edges[(t, t)] = [accept, alias, idx2node]

        return alias_edges

    def node2vec_walk(self, start_node, length):
        walk = [start_node]
        while len(walk) < length + 1:
            cur = walk[-1]
            # 对于起点，没有上一个节点t,默认t==v
            if len(walk) == 1:
                t = cur
            else:
                t = walk[-2]
            edge = (t.item(), cur.item())
            accept = self.alias_edges[edge][0]
            alias = self.alias_edges[edge][1]
            idx2node = self.alias_edges[edge][2]

            next_idx = alias_sample(accept, alias)
            next_node = idx2node[next_idx]
            walk.append(torch.tensor(next_node, dtype=start_node.dtyoe))

        walk = torch.tensor(walk)
        return walk

    def graph_walk(self, start_nodes, length):
        walks = None
        for start_node in start_nodes:
            # 对应node2vecWalk
            walk = self.node2vec_walk(start_node, length)
            if walks is None:
                walks = walk
            else:
                walks = torch.cat((walks, walk), 0)
        return walks

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        # sample_neighbors 只计算入边邻居
        # u, v = torch.tensor([0, 1, 1, 2, 1, 3], dtype=torch.int32), torch.tensor([1, 0, 2, 1, 3, 1], dtype=torch.int32)
        # g = dgl.graph((u, v))
        # v = torch.tensor(1, dtype=torch.int64)
        # subgraph = sample_neighbors(g, v, -1)  # -1表示取所有邻居
        # us = subgraph.edges()[0].numpy()  # 一阶邻居
        # vs = subgraph.edges()[1].numpy()  # v
        # print(us)  # [0]
        # print(vs)  # [1]
        # exit()
        # self.G = g
        self.fir_nbrs = self.get_fir_nbrs()
        self.alias_edges = self.get_alias_edges()
        # print(self.fir_nbrs)
        # print(self.alias_edges)
        # exit()

        # 生成所有walk的最外层循环 1 to r
        transform = NodeShuffle()
        all_walks = None
        for _ in tqdm(range(num_walks)):
            G = transform(G)
            # 第二场循环 u \in V
            walks = self.graph_walk(list(G.nodes()), length=walk_length - 1)
            for walk in walks:
                if torch.tensor(-1) in walk:
                    print(walk)
            if all_walks is None:
                all_walks = walks
            else:
                all_walks = torch.cat((all_walks, walks), 0)

        return all_walks
