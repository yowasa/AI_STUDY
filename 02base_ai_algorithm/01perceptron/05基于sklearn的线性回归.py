import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

from sklearn.externals import joblib

# 设置中文支持
mpl.rcParams["font.family"] = 'Arial Unicode MS'

# 数据来源 UCI:https://archive.ics.uci.edu/ml/index.php
# Individual household electric power consumption 用电数据
# data_set/household_power_consumption.txt

# 1.加载数据
df = pd.read_csv('../../data_set/household_power_consumption.txt', sep=';', nrows=10000)
# 2.数据清洗
# #inplace=True是指在当前对象上直接修改
df.replace('?', np.nan, inplace=True)
# #按照行删除数据为nan的数据 any是任意一个属性为nan则删除 all是只有当所有特征为nan才删除
df=df.dropna(axis=0, how='any')
# 3.根据需求获取目标属性X和标签Y
X = df.iloc[:, 2:4].astype(np.float)
Y = df.iloc[:, 5].astype(np.float)
print(X.shape)
print(Y.shape)

# 4.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# 5.特征工程
# 数据标准化 StandardScaler 用这个进行训练 使得所有数据标准化 （将数据变为均值为0方差为1）
stand_scaler = StandardScaler()
# #先训练模型
stand_scaler.fit(X_train, Y_train)
# #用训练好的模型进行转化操作
# 可以用fit_transform变为一步操作
X_train = stand_scaler.transform(X_train)
X_test=stand_scaler.transform(X_test)
# 6.模型的构建
# fit_intercept=True,  是否训练截距项
# normalize=False,    是否归一化处理
# copy_X=True,      是否copy一份训练
# n_jobs=None       指定多少线程训练模型
algo = LinearRegression()
# 7.模型的训练
algo.fit(X_train,Y_train)
# 8.模型效果评估
print(f'各个特征属性的权重:{algo.coef_}')
print(f'截距项值:{algo.intercept_}')
print(f'模型在训练集上的效果r:{algo.score(X_train,Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test,Y_test)}')
# 9.持久化
joblib.dump(stand_scaler,'./model/scaler.m')
joblib.dump(algo,'./model/algo.m')

# 持久化测试
algo_t=joblib.load('./model/algo.m')
print(f'模型持久化后读取的效果r:{algo_t.score(X_test,Y_test)}')