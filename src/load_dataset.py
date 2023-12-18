from ogb.linkproppred import DglLinkPropPredDataset
from ogb.nodeproppred import DglNodePropPredDataset


def link_prediction_data():
    # 载入OGB的Link Property Prediction数据集
    dataset = DglLinkPropPredDataset(name='ogbl-ddi')
    # split_edge = dataset.get_edge_split()

    g = dataset[0]
    # print(split_edge['train'].keys())
    # print(split_edge['valid'].keys())
    # print(split_edge['test'].keys())
    # dict_keys(['edge'])
    # dict_keys(['edge', 'edge_neg'])
    # dict_keys(['edge', 'edge_neg'])
    return g


def node_prediction_data():
    # 载入OGB的Node Property Prediction数据集
    dataset = DglNodePropPredDataset(name='ogbn-arxiv')
    split_idx = dataset.get_idx_split()

    # there is only one graph in Node Property Prediction datasets
    # 在Node Property Prediction数据集里只有一个图
    g, labels = dataset[0]
    # 获取划分的标签
    train_label = dataset.labels[split_idx['train']]
    valid_label = dataset.labels[split_idx['valid']]
    test_label = dataset.labels[split_idx['test']]
    return g, labels


if __name__ == '__main__':
    link_prediction_data()
