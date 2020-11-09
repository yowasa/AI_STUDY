import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

"""
产生服从给定方差和均值的高斯分布矩阵 
n_samples 产生多少样本
n_features 产生多少特征
centers 中心点数目或坐标
cluster_std 每个簇对应样本的标准差
center_box 中心坐标的取值范围
shuffle 是否打乱顺序 默认打乱
random_state 随机数种子
"""
n_centers=4
x, y = make_blobs(n_samples=1000,
                  n_features=2,
                  centers=n_centers,
                  center_box=(0, 10),
                  cluster_std=2.0,
                  random_state=22)
# plt.scatter(x[:, 0], x[:, 1], c=y, s=30)
# plt.show()
algo = KMeans(n_clusters=n_centers)
algo.fit(x)

x_test = [[-4, 8], [-3, 7], [0, 5], [9, 9], [8, -8]]
y_pre=algo.predict(x_test)
print(f'预测值{y_pre}')
print(f'簇中心点：{algo.cluster_centers_}')
print(f'损失值：{algo.inertia_}')
#损失函数值
print(f'评分：{algo.score(x)}')

