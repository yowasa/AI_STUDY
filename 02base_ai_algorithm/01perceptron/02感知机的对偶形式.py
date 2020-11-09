import random as rd
import matplotlib.pyplot as plt
import numpy as np

# 对偶形式的使用及验证

# 设置中文字体
plt.rcParams["font.family"] = 'Arial Unicode MS'

# 数据
data = [[1, 4, 1], [0.5, 2, 1], [2, 2.3, 1], [1, 0.5, -1], [2, 1, -1],
        [4, 1, -1], [3.5, 4, 1], [3, 2.2, -1]]
# 学习率
lr = 0.1

# 初始b
b = 0.0

# alpha每个数据每次增长lr
alpha = np.array([0.0] * len(data))
y_i = np.array(data)[:, 2]
x_i = np.array(data)[:, 0:2]

# 数据集内积
graph = x_i.dot(x_i.T)


# 计算y值 i为第n个测试数据
def cal_y(i):
    return (alpha * y_i).dot(graph[:, i]) + b


# 训练100次
for n in range(100):
    errors = []
    # 下标为i的数据 判断为误判时
    for i in range(len(data)):
        y_cal = cal_y(i)
        # 是否算对
        if y_cal * y_i[i] <= 0:
            errors.append(i)
    if len(errors) == 0:
        # 没有算错的数据则终止
        break
    target = rd.choice(errors)
    alpha[target] += lr
    b += lr * y_i[target]

w = (alpha * y_i).dot(x_i)

# 绘图验证
# 正例的x轴
x1 = []
# 正例的y轴
y1 = []
# 负例的x轴
x2 = []
# 负例的y轴
y2 = []

for each in data:
    # 正例
    if each[2] > 0:
        x1.append(each[0])
        y1.append(each[1])
    else:
        x2.append(each[0])
        y2.append(each[1])
# 设置点
plt.scatter(x1, y1, label='正例')
plt.scatter(x2, y2, label='负例')
# # 给图像增加图例
plt.legend()
# 轴名称
plt.xlabel('X1')
plt.ylabel('X2')
# 0到4 生成等差数列 其中有20个点
label_x = np.linspace(0, 4, 20)
# 根据20个点的x和y 带入公式画线
plt.plot(label_x, -(b + w[0] * label_x) / w[1])
# 显示图片
plt.show()
