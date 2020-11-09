import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# 划分训练数据集
from sklearn.model_selection import train_test_split

# 数据来源 UCI:https://archive.ics.uci.edu/ml/index.php
# Individual household electric power consumption 用电数据
# data_set/household_power_consumption.txt

df = pd.read_csv('../../data_set/household_power_consumption.txt', sep=';', nrows=1000)

# df.info()
# print(df.head(10))
# 功率与电流关系
X = df.iloc[:, 2:4]
# 增加一行b
X['b'] = pd.Series(data=np.ones(shape=X.shape[0]))
Y = df.iloc[:, 5]
# random_state 指定划分时使用的种子 test_size是训练比重 也可以用train_size
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# 转化格式

x = np.mat(X_train)
# reshape变化格式 -1是行不做限制 1是列为一列
y = np.mat(Y_train).reshape(-1, 1)

print(x.shape)
print(y.shape)

# 解析式求解
theta = (x.T * x).I * x.T * y
print(theta)

# 验证数据
p_y = np.mat(X_test) * theta

# 画图看效果 横坐标为序列号 纵坐标为y
t = np.arange(len(X_test))
# 设置中文字体
plt.rcParams["font.family"] = 'Arial Unicode MS'
# 设置背景色为白色
plt.figure(facecolor='w')
# 红色为预测值
plt.plot(t, p_y, 'r-', label='预测值')
# 蓝色为真实值
plt.plot(t, Y_test, 'b-', label='真实值')
# 标签放在右上
plt.legend(loc='upper right')
plt.show()
