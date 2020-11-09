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
    x_train.astype(np.float32).reshape(-1, 1)).reshape(-1, 28 * 28)
x_valid_scaled = scaler.transform(
    x_valid.astype(np.float32).reshape(-1, 1)).reshape(-1, 28 * 28)
x_test_scaled = scaler.transform(
    x_test.astype(np.float32).reshape(-1, 1)).reshape(-1, 28 * 28)

# 定义全连接层有两层，每次有100个神经元
hidden_units = [100, 100]
# 类别数
class_num = 10

# 建立两个placeholder用于存放数据和标签
# placeholder是占位符，数据通过占位符输入到网络
# placeholder()函数是在神经网络构建graph的时候在模型中的占位，此时并没有把要输入的数据传入模型，它只会分配必要的内存。等建立session，在会话中，运行模型的时候通过feed_dict()函数向占位符喂入数据。
x = tf.placeholder(tf.float32, [None, 28 * 28])
y = tf.placeholder(tf.int64, [None])

# 定义层次
# 输入
input_for_next_layer = x
# 隐藏层
for hidden_unit in hidden_units:
    input_for_next_layer = tf.layers.dense(input_for_next_layer,
                                           hidden_unit,
                                           activation=tf.nn.relu)
# 输出层
logits = tf.layers.dense(input_for_next_layer, class_num)

# 定义损失函数：tf.losses.sparse_softmax_cross_entropy
# 1.最后一个隐层的输出*最后一组权重=输出神经节点的输出值->softmax->变成了概率
# 2.对labels做one-hot编码
# 3.计算交叉熵
loss = tf.losses.sparse_softmax_cross_entropy(labels=y,
                                              logits=logits)

# 获得精确度
# 预测值，就是logits中最大的那个值对应的索引
prediction = tf.argmax(logits, 1)
correct_prediction = tf.equal(prediction, y)
# tf.reduce_mean用来计算张量tensor沿着指定轴的平均值
# tf.cast执行tensorflow中张量数据类型的转换
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64))

# 运行一遍train_op,网络就被训练一次
train_op = tf.train.AdamOptimizer(1e-3).minimize(loss)

print(x)
print(logits)

# 构建完图之后，运行图
# session

init = tf.global_variables_initializer()
batch_size = 20
epochs = 10
train_steps_per_epoch = x_train.shape[0] // batch_size
valid_steps = x_valid.shape[0] // batch_size


# 每一个epoch计算一下精度的均值
def eval_with_sess(sess, x, y, accuracy, images, labels, batch_size):
    eval_steps = images.shape[0] // batch_size
    eavl_accuracies = []
    for step in range(eval_steps):
        batch_data = images[step * batch_size:(step + 1) * batch_size]
        batch_label = labels[step * batch_size:(step + 1) * batch_size]
        accuracy_val = sess.run(accuracy,
                                feed_dict={
                                    x: batch_data,
                                    y: batch_label
                                })
        eavl_accuracies.append(accuracy_val)
    return np.mean(eavl_accuracies)


# 打开一个session
with tf.Session() as sess:
    # 初始化
    sess.run(init)
    for epoch in range(epochs):
        for step in range(train_steps_per_epoch):
            batch_data = x_train_scaled[
                         step * batch_size:(step + 1) * batch_size]
            batch_label = y_train[
                          step * batch_size:(step + 1) * batch_size]
            loss_val, accuracy_val, _ = sess.run([loss, accuracy, train_op],
                                                 feed_dict={
                                                     x: batch_data,
                                                     y: batch_label
                                                 })
            print('\r[Train] epoch: %d, step:%d, loss: %3.5f, accuracy: %2.2f'
                  % (epoch, step, loss_val, accuracy_val), end="")
        valid_accuracy = eval_with_sess(sess, x, y, accuracy,
                                        x_valid_scaled, y_valid, batch_size)
        print("\t[Valid] acc: %2.2f" % (valid_accuracy))
