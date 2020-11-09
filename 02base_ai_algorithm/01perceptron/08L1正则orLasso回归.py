import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Lasso
import time
"""
L1正则 Lasso回归 由于使用绝对值的形式使得约束与损失函数的切线（最优解）更容易切到坐标轴上 
造成稀疏矩阵 排除噪音数据 经常用于参数筛选
"""


# 设置中文支持
mpl.rcParams["font.family"] = 'Arial Unicode MS'

# 数据来源 UCI:https://archive.ics.uci.edu/ml/index.php
# Individual household electric power consumption 用电数据
# data_set/household_power_consumption.txt

# 1.加载数据
df = pd.read_csv('../../data_set/household_power_consumption.txt', sep=';', nrows=1000)
# 2.数据清洗
# #inplace=True是指在当前对象上直接修改
df.replace('?', np.nan, inplace=True)
# #按照行删除数据为nan的数据 any是任意一个属性为nan则删除 all是只有当所有特征为nan才删除
df = df.dropna(axis=0, how='any')
# 3.根据需求获取目标属性X和标签Y
X = df.iloc[:, 0:2]
Y = df.iloc[:, 4].astype(np.float)
def date_format(t):
    date_str = time.strptime(' '.join(t), "%d/%m/%Y %H:%M:%S")
    return (date_str.tm_year, date_str.tm_mon, date_str.tm_mday, date_str.tm_hour, date_str.tm_min, date_str.tm_sec)
X = X.apply(lambda row: pd.Series(date_format(row)), axis=1)
# 4.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)
# 5.特征工程
poly = PolynomialFeatures(degree=3)
# 6.模型的构建
"""
alpha=1.0, L1正则项比重 越大最终模型参数越小
fit_intercept=True,  是否有截距项
normalize=False, 是否标准化
precompute=False, 是否预处理
copy_X=True, 复制X
max_iter=None, 最大迭代次数
tol=1e-3, 训练的收敛值（变化小于多少收敛）
warm_start=False, 是否在前一阶段的训练结果上继续训练
positive=False, 是否强制要求权重向量的分量都为正数
random_state=None 随机数种子
selection='cyclic' 训练过程方式（循环训练）还有一个random 随机训练
"""
lr = Lasso()
algo = Pipeline(steps=[
    ("poly", poly),
    ("lr", lr)
])
# 7.模型的训练
algo.fit(X_train,Y_train)
# 8.模型效果评估
print(f'各个特征属性的权重:{algo.get_params()["lr"].coef_}')
print(f'截距项值:{algo.get_params()["lr"].intercept_}')
print(f'模型在训练集上的效果r:{algo.score(X_train,Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test,Y_test)}')
