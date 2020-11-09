import random as rd
import numpy as np
#通用感知机类构件


# 给予初始的计算进行优化

# 感知机类
class Perceptron:
    # 初始化w向量和截距b，注意初始化为浮点类型
    # x_num 为特征数
    def __init__(self, x_num, lr=0.01):
        self.w = []
        for i in range(x_num):
            self.w.append(0.0)
        self.b = 0.0
        self.lr = lr
        self.x_num = x_num

    # 计算y值
    def cal_y(self, x):
        return np.sum(np.array(self.w) * np.array(x)) + self.b

    def train(self, data, step=100):
        # 训练100次
        for i in range(step):
            # 每次随机取一个
            index = rd.randint(0, len(data) - 1)
            each_data = data[index]
            # 训练数据的x值
            x = each_data[0:self.x_num]
            # 训练数据的y值
            y = each_data[self.x_num]
            # 根据模型计算得到的y值
            y_c = self.cal_y(each_data[0:self.x_num])
            # 如果值不同号
            if y * y_c <= 0:
                self.w = np.array(self.w) + self.lr * np.array(x) * y
                self.b = self.b + self.lr * y


if __name__ == "__main__":
    data = [[1, 4, 1], [0.5, 2, 1], [2, 2.3, 1], [1, 0.5, -1], [2, 1, -1],
            [4, 1, -1], [3.5, 4, 1], [3, 2.2, -1]]
    per = Perceptron(2)
    per.train(data)
    print(per.w, per.b)
