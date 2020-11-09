import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
"""
增加额外的二次或更高次系数，从而防止欠拟合
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
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# 5.特征工程
poly = PolynomialFeatures()
# 6.模型的构建
lr = LinearRegression()
algo = Pipeline(steps=[
    ("poly", poly),
    ("lr", lr)
])
# 7.模型的训练
# 从1到5画5张图
N = 5
t = np.arange(len(X_test))
d_pool = np.arange(1, N, 1)
clrs = []
for c in np.linspace(16711680, 255, d_pool.size):
    clrs.append('#%06x' % int(c))
plt.figure(figsize=(12, 6), facecolor='w')
for i, d in enumerate(d_pool):
    plt.subplot(N - 1, 1, i + 1)
    plt.plot(t, Y_test,'r-',label="真实值", ms=10,zorder=N)
    algo.set_params(poly__degree=d)
    algo.fit(X_train,Y_train)
    lin=algo.get_params()['lr']
    y_pre=algo.predict(X_test)
    plt.plot(t,y_pre,'b-',lw=3,label=f'扩展深度{d}的预测值，模型得分:{algo.score(X_test,Y_test)}',zorder=N-1)
    plt.legend(loc='upper left')
plt.suptitle("线性回归与多项式深度关系",fontsize=20)
plt.show()

