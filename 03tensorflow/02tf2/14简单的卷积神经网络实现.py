import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras

# 读取keras中的进阶版mnist数据集
fashion_mnist = keras.datasets.fashion_mnist
# 加载数据集，切分为训练集和测试集
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
# 从训练集中将后五千张作为验证集，前五千张作为训练集
# [:5000]默认从头开始，从头开始取5000个
# [5000:]从第5000开始(不包含5000)，结束位置默认为最后
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]
# 打印这些数据集的大小
print(x_valid.shape, y_valid.shape)
print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# 归一化处理：x = (x - u)/std :减去均值除以方差

from sklearn.preprocessing import StandardScaler

# 初始化一个StandarScaler对象
scaler = StandardScaler()
# fit_transform要求为二维矩阵，因此需要先转换
# 要进行除法，因此先转化为浮点型
# x_train是三维矩阵[None,28,28]，先将其转换为二维矩阵[None,784],再将其转回三维矩阵
# reshape（-1， 1）转化为一列(-1代表不确定几行)
# fit: 求得训练集的均值、方差、最大值、最小值等训练集固有的属性
# transform: 在fit的基础上，进行标准化，降维，归一化等操作
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28, 1)

# tf.keras.models.Sequential() 构建模型
# 构建深度神经网络
model = keras.models.Sequential()
# 添加卷积层
# filters：卷积核的个数， kernel_size:卷积核的尺寸， padding: 是否填充原图像
# avtivation: 激活函数， input_shape:输入的图像的大小，为1通道
model.add(keras.layers.Conv2D(filters=32, kernel_size=3,
                              padding='same',
                              activation="selu",
                              input_shape=(28, 28, 1)))
model.add(keras.layers.Conv2D(filters=32, kernel_size=3,
                              padding='same',
                              activation="selu"
                              ))
# 添加池化层
# 经过池化层后，图像长宽各减少1/2，面积减少1/4，因此会造成图像的损失
# 所以在之后的卷积层中，卷积核的个数翻倍以缓解这种损失
model.add(keras.layers.MaxPool2D(pool_size=2))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3,
                              padding='same',
                              activation="selu"
                              ))
model.add(keras.layers.Conv2D(filters=64, kernel_size=3,
                              padding='same',
                              activation="selu"
                              ))
model.add(keras.layers.MaxPool2D(pool_size=2))

model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                              padding='same',
                              activation="selu"
                              ))
model.add(keras.layers.Conv2D(filters=128, kernel_size=3,
                              padding='same',
                              activation="selu"
                              ))
model.add(keras.layers.MaxPool2D(pool_size=2))
# 将输出展平
model.add(keras.layers.Flatten())
# 连接全连接层
model.add(keras.layers.Dense(128, activation="selu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])

model.summary()

logdir = os.path.join("cnn-selu-callbacks")
if not os.path.exists(logdir):
    os.mkdir(logdir)
# 在callbacks文件夹下创建文件。c=os.path.join(a,b),c=a/b
output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")

callbacks = [
    keras.callbacks.TensorBoard(log_dir=logdir),
    keras.callbacks.ModelCheckpoint(output_model_file,
                                    save_best_only=True),
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
]
history = model.fit(x_train_scaled, y_train, epochs=10,
                    validation_data=(x_valid_scaled, y_valid),
                    callbacks=callbacks)


def plot_learning_curves(history):
    # 将history.history转换为dataframe格式
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    # gca：get current axes,gcf: get current figure
    plt.gca().set_ylim(0, 3)
    plt.show()


plot_learning_curves(history)

model.evaluate(x_test_scaled, y_test, verbose=2)
