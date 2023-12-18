import random
import torch
from dgl import NodeShuffle
from dgl.sampling import random_walk
from .alias import alias_sample
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
    def __init__(self, G, p=1, q=1, use_rejection_sampling=False):
        self.G = G
        self.p = p
        self.q = q
        self.use_rejection_sampling = use_rejection_sampling

    def walk(self, walk_length, start_node):

        G = self.G
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        while len(walk) < walk_length:
            cur = walk[-1]
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                if len(walk) == 1:
                    walk.append(
                        cur_nbrs[alias_sample(alias_nodes[cur][0], alias_nodes[cur][1])])
                else:
                    prev = walk[-2]
                    edge = (prev, cur)
                    next_node = cur_nbrs[alias_sample(alias_edges[edge][0],
                                                      alias_edges[edge][1])]
                    walk.append(next_node)
            else:
                break

        return walk

    def simulate_walks(self, num_walks, walk_length):
        G = self.G
        nodes = list(G.nodes())

        walks = []
        for _ in range(num_walks):
            random.shuffle(nodes)
            for v in nodes:
                walks.append(self.walk(walk_length=walk_length, start_node=v))
        return walks