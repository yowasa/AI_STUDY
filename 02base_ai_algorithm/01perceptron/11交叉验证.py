import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import RidgeCV
import time

"""
超参的选取方法 不同取值的超参互相组合进行验证
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
poly = PolynomialFeatures(degree=2)
# 6.模型的构建
"""
alphas=[1e-3, 1e-2, 1e-1, 1, 10] L2正则项目超参选值范围
fit_intercept=True,  是否有截距项
normalize=False, 是否标准化
scoring=None 交叉验证时选择最优模型的方式 默认使用score
cv=None 做几折交叉验证
"""
lr = RidgeCV(alphas=np.logspace(-3, 2, 50), cv=5)
algo = Pipeline(steps=[
    ("poly", poly),
    ("lr", lr)
])
# 7.模型的训练
algo.fit(X_train, Y_train)
# 8.模型效果评估
print(f'各个特征属性的权重:{algo.get_params()["lr"].coef_}')
print(f'截距项值:{algo.get_params()["lr"].intercept_}')
print(f'最优模型参数:{algo.get_params()["lr"].alpha_}')
print(f'模型在训练集上的效果r:{algo.score(X_train, Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test, Y_test)}')
