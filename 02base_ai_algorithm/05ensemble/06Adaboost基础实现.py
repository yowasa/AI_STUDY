import math
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 数据集
X = np.array([[0, 1],
              [1, 1],
              [2, 1],
              [3, -1],
              [4, -1],
              [5, -1],
              [6, 1],
              [7, 1],
              [8, 1],
              [9, -1]])


# 计算错误率
def error_rate(y, pre_y, w):
    count = 0
    for i, value in enumerate(y):
        if value != pre_y[i]:
            count += w[i]
    return count


# 计算alpha
def cal_alpha(error_rate):
    x = (1 - error_rate) / error_rate
    return 0.5 * math.log(x, 2)


# 转化为占比
def transform(w):
    sum = 0
    for i in w:
        sum += i
    w_after = []
    for i in w:
        w_after.append(i / sum)
    return w_after


# 样本权重 初始权重为1/n
w_im = [[0.1] * 10]

# 模型可信度
alphas = []
# 模型
estimators = []


def final_cal(x):
    sum = 0
    for i, d in enumerate(estimators):
        sum += d.predict(x.reshape(-1, 1)) * alphas[i]
    return sum


for i in range(5):
    d = DecisionTreeClassifier(max_depth=1)
    d.fit(X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1), sample_weight=w_im[i])
    estimators.append(d)
    y_pre = d.predict(X[:, 0].reshape(-1, 1))
    e = error_rate(X[:, 1], y_pre, w_im[i])
    alpha = cal_alpha(e)
    alphas.append(alpha)
    w_next = []
    for j in X:
        w_each = math.exp(-j[1] * final_cal(np.array([j[0]])))
        w_next.append(w_each)

    w_im.append(transform(w_next))

print(final_cal(X[:, 0]))
