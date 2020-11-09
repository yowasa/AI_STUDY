import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import ElasticNet
import time
"""
L1与L2正则按照一定比重相加
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
lr = ElasticNet()
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
# 更多评估标准
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error,explained_variance_score
pred_train=algo.predict(X_train)
pred_test=algo.predict(X_test)
# 使用模型score算出的结果就是R2
# 1-[(y预测-y实际)**2求和]/[(y预测-y实际值的平均值)**2求和]
print(f'训练集上的R2:{r2_score(Y_train,pred_train)}')
print(f'测试集上的R2:{r2_score(Y_test,pred_test)}')

# 误差平均和 即损失函数的平均值（损失函数算出后除以样本数量）
print(f'训练集上的MSE:{mean_squared_error(Y_train,pred_train)}')
print(f'测试集上的MSE:{mean_squared_error(Y_test,pred_test)}')
# 平均绝对误差 MSE用的平方 MAE用的绝对值
print(f'训练集上的MAE:{mean_absolute_error(Y_train,pred_train)}')
print(f'测试集上的MAE:{mean_absolute_error(Y_test,pred_test)}')
# MSE开根号
print(f'训练集上的RMSE:{np.sqrt(mean_squared_error(Y_train,pred_train))}')
print(f'测试集上的RMSE:{np.sqrt(mean_squared_error(Y_test,pred_test))}')
# 解释性方差较为接近R2 一般不考虑
print(f'训练集上的解释性方差:{explained_variance_score(Y_train,pred_train)}')
print(f'测试集上的解释性方差:{explained_variance_score(Y_test,pred_test)}')

