import random as rd
import matplotlib.pyplot as plt
import numpy as np
# 最基础实现及验证

# 设置中文字体
plt.rcParams["font.family"] = 'Arial Unicode MS'

# 数据
data = [[1, 4, 1], [0.5, 2, 1], [2, 2.3, 1], [1, 0.5, -1], [2, 1, -1],
        [4, 1, -1], [3.5, 4, 1], [3, 2.2, -1]]
# 学习率
lr = 0.01
# 初始权重
w = [0.0, 0.0]
# 初始b
b = 0.0


# 计算y值
def cal_y(x):
    return np.sum(np.array(w) * np.array(x)) + b


# 训练100次
for i in range(100):
    # 每次随机取一个
    index = rd.randint(0, len(data) - 1)
    each_data = data[index]
    # 训练数据的x值
    x = each_data[0:2]
    # 训练数据的y值
    y = each_data[2]
    # 根据模型计算得到的y值
    y_c = cal_y(each_data[0:2])
    # 如果值不同号
    if y * y_c <= 0:
        w = np.array(w) + lr * np.array(x) * y
        b = b + lr * y

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
