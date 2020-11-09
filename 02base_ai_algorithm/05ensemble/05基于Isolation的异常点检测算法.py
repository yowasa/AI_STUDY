# 特点:深度小，数据量小，对特征进行阈值划分
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# 设置中文支持
mpl.rcParams["font.family"] = 'Arial Unicode MS'

np.random.seed(28)

x = 0.3 * np.random.rand(100, 2)
x_train = np.vstack((x + 2, x - 2))
x = 0.3 * np.random.rand(20, 2)
x_test = np.vstack((x + 2, x - 2))
x_outliers = np.random.uniform(low=-2.5, high=2.5, size=(20, 2))

algo = IsolationForest()
algo.fit(x_train)

y_pred_train = algo.predict(x_train)
print(f'训练数据预测结果{y_pred_train}')
y_pred_test = algo.predict(x_test)
print(f'测试数据预测结果{y_pred_test}')
y_pred_outliers = algo.predict(x_outliers)
print(f'异常数据预测结果{y_pred_outliers}')

# 等高线图
x1_max = 3
x1_min = -3
x2_max = 3
x2_min = -3

t1 = np.linspace(x1_min, x1_max, 50)
t2 = np.linspace(x2_min, x2_max, 50)
x1, x2 = np.meshgrid(t1, t2)
x_show = np.dstack((x1.flat, x2.flat))[0]
z = algo.decision_function(x_show)
z = z.reshape(x1.shape)
plt.contourf(x1, x2, z, cmap=plt.cm.Blues_r)
plt.scatter(x_train[:, 0], x_train[:, 1], c='b')
plt.scatter(x_test[:, 0], x_test[:, 1], c='g')
plt.scatter(x_outliers[:, 0], x_outliers[:, 1], c='r')
plt.show()
