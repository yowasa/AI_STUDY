import os

import matplotlib as mpl
import numpy as np
from sklearn.preprocessing import StandardScaler
from tensorflow.python import keras

mpl.rcParams["font.family"] = 'Arial Unicode MS'
fashion_mnist = keras.datasets.fashion_mnist
(x_train_all, y_train_all), (x_test, y_test) = fashion_mnist.load_data()
x_valid, x_train = x_train_all[:5000], x_train_all[5000:]
y_valid, y_train = y_train_all[:5000], y_train_all[5000:]

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
# 构建深度神经网络

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(3):
    model.add(keras.layers.Dense(100, activation="relu"))
    # 加入批归一化（将其放在激活函数的后面）
    model.add(keras.layers.BatchNormalization())
    # 层归一化 适用于动态网络或者批量较少的场景
    # model.add(keras.layers.LayerNormalization())

    """
    将批归一化放在激活函数的前面： 对于relu来说没太大区别 对于其他类似于sigmoid函数来说 放在激活函数前面可以防止梯度消失
    model.add(keras.layers.Dense(100))
    model.add(keras.layers.BatchNormalization())
    model.add(keras.layers.Avtivation("relu")
    """
# tf.keras.models.Sequential() 构建模型
# 构建深度神经网络


# dropout
# for _ in range(20):
#     # selu:自带归一化功能的激活函数
#     model.add(keras.layers.Dense(100,activation="selu"))
# # 在最后几层添加dropout，防止过拟合,rate表示丢掉神经元的比例
# # AlphaDropout比Dropout的优势：
# # 1.激活值均值与方差不变 2. 归一化性质也不变
# model.add(keras.layers.AlphaDropout(rate=0.5))
# # model.add(keras.layers.Dropout(rate=0.5))
model.add(keras.layers.Dense(10, activation="softmax"))
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
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

logdir = os.path.join("../../callbacks")
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


# 用测试集通过模型做评估
# evaluate 中的 verbose:
# verbose：日志显示
# verbose = 0 为不在标准输出流输出日志信息
# verbose = 1 为输出进度条记录
# 只能取 0 和 1；默认为 1
print(model.evaluate(x_test_scaled, y_test, verbose=2))
