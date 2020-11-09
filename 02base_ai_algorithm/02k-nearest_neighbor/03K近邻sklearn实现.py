import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# 设置中文支持
mpl.rcParams["font.family"] = 'Arial Unicode MS'

names = ['A', 'B', 'C', 'D', 'cla', ]
df = pd.read_csv('../../data_set/iris.data', names=names)
df.info()

X = df[names[0:-1]]
Y = df[names[-1]]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)
"""
n_neighbors=5, 使用多少个邻居
weights='uniform', k个样本等权重 distance 表示距离反比权重
algorithm='auto',  模型求解方式 可选为brute kdtree balltree
leaf_size=30, 当求解方式为kdtree或balltree时，树最多允许的叶子数目
p=2, 在minkowski距离中 变化为欧几里得距离公式
metric='minkowski',  距离公式计算方式
metric_params=None, 距离公式计算中参数列表
n_jobs=None,使用多少线程计算
"""
algo = KNeighborsClassifier(n_neighbors=3)

algo.fit(X_train, Y_train)

print(f'模型在训练集上的效果r:{algo.score(X_train, Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test, Y_test)}')

test1=X_test.iloc[:10,:]
print(test1)
graph1=algo.kneighbors_graph(test1,n_neighbors=3,mode='distance')
print(graph1)
