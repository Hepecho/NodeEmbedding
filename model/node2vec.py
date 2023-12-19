from gensim.models import Word2Vec
from .walker import Node2VecWalker
import torch
import time
from runx.logx import logx
from tqdm import tqdm
from os.path import join as ospj


class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'Node2Vec'
        self.graph = None                                               # 用于训练的图
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.seed = 1                                                   # seed

        # self.dropout = 0.5                                              # 随机失活
        self.walk_length = 10                                           # 游走路径长度
        self.num_walks = 40                                             # 每个节点生成的游走路径数量
        # self.num_epochs = 20                                            # epoch数
        # self.batch_size = 2048                                          # mini-batch大小

        # node2vec控制参数
        self.p = 4
        self.q = 0.5

        # skip-gram模型参数
        self.embed_size = 128                                           # 向量的维度
        self.window_size = 5                                            # 上下文窗口大小
        self.learning_rate = 0.025                                      # 学习率 alpha
        self.iter = 5                                                   # 迭代次数 epochs
        self.workers = 3                                                # 线程数
        self.sg = 1                                                     # 设定为word2vec的skip-gram模型
        self.hs = 1                                                     # 使用Hierarchical Softmax
        self.min_count = 0                                              # 忽略词频小于此值的单词


class Model:
    """Node2Vec"""
    def __init__(self, config):
        self.walker = Node2VecWalker(config.graph, config.p, config.q)
        self.all_walks = None
        self.skip_gram_model = Word2Vec(vector_size=config.embed_size, window=config.window_size, sg=config.sg,
                                        hs=config.hs, min_count=config.min_count, workers=config.workers,
                                        seed=config.seed, alpha=config.learning_rate)

    def train(self, config):
        localtime = time.asctime(time.localtime(time.time()))
        logx.msg('======================Start simulate walks [{}]======================'.format(localtime))
        self.all_walks = self.walker.simulate_walks(num_walks=config.num_walks, walk_length=config.walk_length)
        # all_walks的形状为torch.Size([num_nodes * num_walks, walk_length])
        # proteins torch.Size([10602720, 10]) num_walks = 80
        # arxiv torch.Size([13547440, 10]) num_walks = 80
        logx.msg(str(self.all_walks.shape))
        torch.save(self.all_walks, 'all_walks.pt')
        localtime = time.asctime(time.localtime(time.time()))
        logx.msg('======================Finish simulate walks [{}]======================'.format(localtime))

        localtime = time.asctime(time.localtime(time.time()))
        logx.msg('======================Start Train Model [{}]======================'.format(localtime))
        sentences = self.all_walks.to(config.device)
        sentences = sentences.tolist()
        sentences_str = [[str(node_id) for node_id in node_sequence] for node_sequence in sentences]
        # print(sentences)
        # print(self.skip_gram_model.corpus_count)  # 0
        self.skip_gram_model.build_vocab(sentences_str)
        self.skip_gram_model.train(sentences_str, total_examples=self.skip_gram_model.corpus_count,
                                   epochs=config.iter)
        localtime = time.asctime(time.localtime(time.time()))
        logx.msg('======================Finish Train Model [{}]======================'.format(localtime))
        torch.save(self.skip_gram_model, 'skip_gram.pt')
        return self.skip_gram_model

    def save_embedding(self, config, emb_path):
        nodes_tensor = config.graph.nodes()
        nodes_list = nodes_tensor.tolist()
        vocab = self.skip_gram_model.wv.index_to_key
        keys = sorted(vocab)
        max_node_id = max(nodes_list)
        if max_node_id + 1 != len(self.skip_gram_model.wv.key_to_index):
            print("WARNING: The node ids are not serial.")
            print(max_node_id)
            print(len(self.skip_gram_model.wv.key_to_index))

        embedding = torch.zeros(max_node_id + 1, config.embed_size)

        # idx to key 将skip_gram_model中str(key == node id) --> vector转换成embedding中的 int(idx == node id) --> vector
        idx2key_str = self.skip_gram_model.wv.index_to_key  # list类型
        idx2key_int = [int(key_str) for key_str in idx2key_str]
        # idx2key_int = [key_int for key_int in range(max_node_id + 1)]
        idx2key_tensor = torch.tensor(idx2key_int, dtype=torch.long)

        # 将 Word2Vec 模型的权重复制到 PyTorch 的 embedding 中
        embedding.index_add_(0, idx2key_tensor, torch.from_numpy(self.skip_gram_model.wv.vectors))

        torch.save(embedding, emb_path)
