import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.python import keras

"""
基于tensorflow对服装图片进行分类
"""

# 设置中文支持
mpl.rcParams["font.family"] = 'Arial Unicode MS'

# 读取keras中的进阶版mnist数据集
fashion_mnist = keras.datasets.fashion_mnist

# 加载数据集，切分为训练集和测试集
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()

# 从训练集中将后五千张作为验证集，前五千张作为训练集
# [:5000]默认从头开始，从头开始取5000个
# [5000:]从第5001开始，结束位置默认为最后
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
# 打印这些数据集的大小
print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# def show_single_image(img_arr):
#     plt.imshow(img_arr, cmap="binary")
#     plt.show()
# # 显示训练集第一张图片
# show_single_image(x_train[0])


class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
               'Ankle boot']
# tf.keras.models.Sequential() 构建模型的容器

# 创建一个Sequential的对象,顺序模型，多个网络层的线性堆叠
# 可使用add方法将各层添加到模块中
model = keras.models.Sequential()

# 添加层次
# 输入层：Flatten将28*28的图像矩阵展平成为一个一维向量
model.add(keras.layers.Flatten(input_shape=[28, 28]))

# 全连接层（上层所有单元与下层所有单元都连接）：
# 第一层300个单元，第二层100个单元，激活函数为 relu:
# relu: y = max(0, x)
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))

# 输出为长度为10的向量，激活函数为 softmax:
# softmax: 将向量变成概率分布，x = [x1, x2, x3],
# y = [e^x1/sum, e^x2/sum, e^x3/sum],sum = e^x1+e^x2+e^x3
model.add(keras.layers.Dense(10, activation="softmax"))

# 目标函数的构建与求解方法
# 为什么使用sparse_categorical_crossentropy :
# y->是一个数，要用sparse_categorical_crossentropy
# y->是一个向量，直接用categorical_crossentropy

# loss 损失函数
# optimizer 优化方式 gradient_descent为梯度下降  adam自适应矩估计 梯度下降的变形 不会因为梯度很大导致步长过大
# metrics 评估标准 accuracy准确率
tf.keras.optimizers
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam")


# 定义回调函数类
class MyCallback(tf.keras.callbacks.Callback):
    # 在每次训练结束时调用
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('loss') < 0.4:
            print("\n 损失函数足够低 终止训练")
            self.model.stop_training = True


"""
构建模型也可以这样：
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(300,activation="relu"),
    keras.layers.Dense(300,activation="relu"),
    keras.layers.Dense(10,activation="softmax")
])
"""

my_callback = MyCallback()

# 看模型的概况
# model.summary()

# 开启训练
# epochs:训练集遍历次数
# validation_data:每个epoch就会用验证集验证
# 会发现loss和accuracy到后面一直不变，因为用sgd梯度下降法会导致陷入局部最小值点
# 因此将loss函数的下降方法改为 adam
# callback 在执行完一次遍历后回调的函数
history = model.fit(x_train, y_train, epochs=10,callbacks=[my_callback],
                    validation_data=(x_valid, y_valid))


# 画出训练过程图形
def plot_learning_curves(history):
    # 将history.history转换为dataframe格式 表格画画图
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    # 设置显示网格
    plt.grid(True)
    # gca：get current axes 获取当前子图,gcf: get current figure 获取当前图标
    # 设置y值范围
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

# 转换为dataframe格式进行查看
pd.DataFrame(history.history)
