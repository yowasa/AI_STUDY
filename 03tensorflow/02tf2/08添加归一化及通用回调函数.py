import os

import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import StandardScaler
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

# 初始化一个StandarScaler对象
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28, 28)
x_valid_scaled = scaler.transform(x_valid.astype(np.float32).reshape(
    -1, 1)).reshape(-1, 28, 28)
x_test_scaled = scaler.transform(x_test.astype(np.float32).reshape(
    -1, 1)).reshape(-1, 28, 28)

class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress',
               'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag',
               'Ankle boot']
# tf.keras.models.Sequential() 构建模型

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
# 为什么使用sparse? :
# y->是一个数，要用sparse_categorical_crossentropy
# y->是一个向量，直接用categorical_crossentropy
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])
# 开启训练
# epochs:训练集遍历10次
# validation_data:每隔一段时间就会验证集验证
# 会发现loss和accuracy到后面一直不变，因为用sgd梯度下降法会导致陷入局部最小值点
# 因此将loss函数的下降方法改为 adam

# callbcaks:回调函数，在每次迭代之后自动调用一些进程，如判断loss值是否达到要求
# 因此callbacks需要加在训练的过程中，即加在fit中
# 此处使用 Tensorboard, earlystopping, ModelCheckpoint 回调函数

# Tensorboard需要一个文件夹，ModelCheckpoint需要一个文件名
# 因此先创建一个文件夹和文件名

logdir = os.path.join("././callbacks")
if not os.path.exists(logdir):
    os.mkdir(logdir)
# 在callbacks文件夹下创建文件。c=os.path.join(a,b),c=a/b
output_model_file = os.path.join(logdir, "fashion_mnist_model.h5")

callbacks = [
    # 增加TensorBoard回调用于可视化
    keras.callbacks.TensorBoard(log_dir=logdir),
    # 每次遍历后储存模型
    keras.callbacks.ModelCheckpoint(output_model_file, save_best_only=True),
    # 提前终止回调函数 patience 当模型5次没有提升则终止 min_delta 改变量少于多少视为无改善
    keras.callbacks.EarlyStopping(patience=5, min_delta=1e-3),
]
history = model.fit(x_train_scaled,
                    y_train,
                    epochs=10,
                    validation_data=(x_valid_scaled, y_valid),
                    callbacks=callbacks)

# 查看tensorboard:
# 1.在所在的环境下，进入callbacks文件夹所在的目录
# 2.输入：tensorboard --logdir="callbacks"
# 3.打开浏览器：输入localhost:(端口号)


# evaluate 中的 verbose:
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# 只能取 0 和 1；默认为 1

print(model.evaluate(x_test_scaled, y_test, verbose=2))
