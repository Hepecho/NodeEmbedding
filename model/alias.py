# Alias是一种O(1) 时间复杂度的离散事件抽样算法，它的思路是用空间换时间，给定N个事件（节点），给定对应的采样概率（权重）
# 将概率转化为矩形面积：给定一个N*1的大矩形，事件i在所有事件中的概率比例（归一化），就是它在大矩形中所占面积的比例
# 对于大小不一的事件矩形，以单位1为标准，裁长补短：将面积大于1的事件多出的面积补充到面积小于1对应的事件中，
# 以确保每一个小方格的面积为1，同时，保证每一方格至多存储两个事件
# 维护两个数组accept和alias,accept数组中的accept[i]表示事件i占第i列矩形的面积的比例。 alias[i]表示第i列中不是事件i的另一个事件的编号。
import numpy as np


def create_alias_table(area_ratio):
    """
    创建accept,alias两个数组
    :param area_ratio: sum(area_ratio)=1 已经归一化的概率数组，长度为N
    :return: accept,alias # accept[i]表示事件i占第i列矩形的面积的比例，alias[i]表示第i列中不是事件i的另一个事件的编号
    """
    l = len(area_ratio)
    accept, alias = [0] * l, [0] * l
    small, large = [], []
    area_ratio_ = np.array(area_ratio) * l
    for i, prob in enumerate(area_ratio_):
        if prob < 1.0:
            small.append(i)
        else:
            large.append(i)

    # 裁长补短，直到某一类事件用尽
    while small and large:
        small_idx, large_idx = small.pop(), large.pop()
        accept[small_idx] = area_ratio_[small_idx]
        alias[small_idx] = large_idx
        # 计算多出的部分
        area_ratio_[large_idx] = area_ratio_[large_idx] - (1 - area_ratio_[small_idx])
        if area_ratio_[large_idx] < 1.0:
            small.append(large_idx)
        else:
            large.append(large_idx)

    # 对无法凑1的剩下的一类事件，每个单独占用一个accept
    # 理论上，上一个while循环只可能剩下large数组非空，且对应的area_ratio_[large_idx]==1
    # 这是因为存在前提"所有矩形面积和为N" 经过取长补短后，最后一个更新的area_ratio_[large_idx]==1，所以large.append(large_idx)
    # 但实际上并不是，由于python浮点数误差的存在，可能略小于1
    # 尝试打印如下信息：
    # if len(small) > 0:
    #     print(area_ratio_[small[-1]])
    # 0.9999999999998304
    # 所以它被错误地归类到small中
    # 为了纠正这一点，我们在赋值accept的时候统一为1
    while large:
        large_idx = large.pop()
        accept[large_idx] = 1
    while small:
        small_idx = small.pop()
        accept[small_idx] = 1

    return accept, alias


def alias_sample(accept, alias):
    """
    :param accept: accept[i]表示事件i占第i列矩形的面积的比例
    :param alias: alias[i]表示第i列中不是事件i的另一个事件的编号
    :return: sample index
    """
    N = len(accept)
    i = int(np.random.rand() * N)
    r = np.random.rand()
    if r < accept[i]:
        return i
    else:
        return alias[i]
