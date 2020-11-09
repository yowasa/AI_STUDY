import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LogisticRegression
LogisticRegression(multi_class='multionmial')
from sklearn.model_selection import GridSearchCV
import time
import warnings
warnings.filterwarnings('ignore')

"""
网格交叉验证
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
poly = PolynomialFeatures(degree=4)
# 6.模型的构建
lr = ElasticNet(alpha=1.0, l1_ratio=0.1)
pipeline = Pipeline(steps=[
    ("poly", poly),
    ("lr", lr)
])
"""
网格交叉验证
estimator, 算法
param_grid, 给定可选参数列表 key为可选对象的属性名称 value为该属性可选参数列表
scoring=None, 验证时选择最优模型的方式 默认使用score
n_jobs=None, 线程数
iid='deprecated', 
refit=True, 
cv=None, 几折交叉验证 
verbose=0, 
pre_dispatch='2*n_jobs',
error_score=np.nan, 
return_train_score=False 返回训练分数
"""
param = {
    'poly__degree': [1, 2, 3, 4, 5],
    'lr__alpha': [0.01, 0.1, 1.0, 10.0],
    'lr__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
    'lr__fit_intercept': [True, False],
}
algo = GridSearchCV(estimator=pipeline, cv=3, param_grid=param)

# 7.模型的训练
algo.fit(X_train, Y_train)
# 8.模型效果评估
print(f'模型在训练集上的效果r:{algo.score(X_train, Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test, Y_test)}')
print(f'模型参数：{algo.best_params_}')
best_pipeline=algo.best_estimator_

print(f'各个特征属性的权重:{best_pipeline.get_params()["lr"].coef_}')
print(f'截距项值:{best_pipeline.get_params()["lr"].intercept_}')
print(f'最优模型参数:{best_pipeline.get_params()["lr"].alpha_}')

