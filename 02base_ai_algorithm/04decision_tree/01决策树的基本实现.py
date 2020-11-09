import numpy as np
from math import log

# 决策树
data = [
    ['青年', '否', '否', '一般', '否'],
    ['青年', '否', '否', '好', '否'],
    ['青年', '是', '否', '好', '是'],
    ['青年', '是', '是', '一般', '是'],
    ['青年', '否', '否', '一般', '否'],
    ['中年', '否', '否', '一般', '否'],
    ['中年', '否', '否', '好', '否'],
    ['中年', '是', '是', '好', '是'],
    ['中年', '否', '是', '非常好', '是'],
    ['中年', '否', '是', '非常好', '是'],
    ['老年', '否', '是', '非常好', '是'],
    ['老年', '否', '是', '好', '是'],
    ['老年', '是', '否', '好', '是'],
    ['老年', '是', '否', '非常好', '是'],
    ['老年', '否', '是', '一般', '否']
]


def np_count(nparray, x):
    i = 0
    for n in nparray:
        if n == x:
            i += 1


def cal_entropy(index, data):
    D = data[:, index]
    button = len(D)
    r_D = list(set(D))
    # H(D)的大小
    H_D = 0
    for i in r_D:
        num = np_count(D, i)
        prob = num / button
        H_D -= prob * log(prob, 2)
    return H_D


# 计算应该用哪个维度
def cal_dimension(data):
    H_D = cal_entropy(-1, data)
    result_lis = []
    for i in range(data.shape[1] - 1):
        H_A = cal_entropy(i, data)
        A = data[:, i]
        r_A = list(set(A))
        H_A_D = 0
        for j in r_A:
            sub_data = filter(lambda x: x[i] == j, data)
            H_A_D += cal_entropy(i, sub_data[i])
        # 计算熵增益比
        result_lis.append((H_D - H_A_D) / H_A)
    return result_lis.index(max(result_lis))


class Node(object):
    def __init__(self, parent, index=None, condition=None, is_leaf=False, values=None):
        self.condition = condition
        self.subNode = []
        self.parent = parent
        self.index = index
        self.is_leaf = is_leaf
        self.values = values


def buildNode(data, parent):
    if len(data) <= 1:
        return Node(parent, values=data, is_leaf=True)

    x_index = cal_dimension(data)
    X = data[:, x_index]
    x_feature = list(set(X))
    result = Node(parent)
    result.index

    pass


data = np.array(data)
x_features = []

for i in range(data.shape[1] - 1):
    r = []
    for j in range(data.shape[0]):
        r.append(data[j][i])
    r = list(set(r))
    x_features.append(r)

print(x_features)

root = buildNode(data, None)
