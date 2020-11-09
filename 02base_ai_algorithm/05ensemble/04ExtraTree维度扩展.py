import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.metrics import classification_report

# 设置中文支持
mpl.rcParams["font.family"] = 'Arial Unicode MS'

names = ['A', 'B', 'C', 'D', 'cla', ]
df = pd.read_csv('../../data_set/iris.data', names=names)
df.info()

X = df[names[0:-1]]

"""
n_estimators: Any = 100,最终训练的子模型数量
max_depth: Any = 5,最大树深
min_samples_split: Any = 2,树分裂的最小样本数目
min_samples_leaf: Any = 1,叶子节点最小样本数目
min_weight_fraction_leaf: Any = 0.,样本权重的最小加成参数（暂无用）
max_leaf_nodes: Any = None,最多允许的叶子节点数目 None 不限制
min_impurity_decrease: Any = 0.,分裂导致不纯度减少大于等于该值则分裂
min_impurity_split: Any = None,分裂提前停止阈值，一个节点不纯度大于此阈值才能分裂
sparse_output: Any = True,是否返回稀疏矩阵
warm_start: Any = False, 是否预热（重用之前的模型进行训练） 默认否
n_jobs: Any = None, 线程数
random_state: Any = None, 随机数种子
verbose: Any = 0  是否打印训练过程 0不打印 1打印
"""
algo = RandomTreesEmbedding(n_estimators=10, max_depth=3, sparse_output=False)
algo.fit(X)
x_ex = algo.transform(X)

# 追加特征列表
for x in x_ex[0:10]:
    print(x)
