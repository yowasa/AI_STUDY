import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# 设置中文支持
mpl.rcParams["font.family"] = 'Arial Unicode MS'

# 造数据
N = 100
np.random.seed(22)
x = np.linspace(0, 6, N) + np.random.normal(0, 2.0, N)
y = 14 * x + 7 + np.random.normal(0, 5.0, N)

x.shape = -1, 1
y.shape = -1, 1


# 实现线性模型的求解
def train():
    with tf.Graph().as_default():
        # 一、执行图构建
        # a.定义占位符
        input_x = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
        input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1], name='x')
        # b.定义模型参数
        # 使用初始化器 初始化一个shape大小的均值为0 标准差为1的数据
        w = tf.get_variable(dtype=tf.float32, name='w', shape=[1, 1],
                            initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
        # 使用初始化器全部填充0
        b = tf.get_variable(dtype=tf.float32, name='b', shape=[1],
                            initializer=tf.zeros_initializer())

        # c.模型预测的构建（获取预测值）
        y_ = tf.matmul(input_x, w) + b

        # d.损失函数的构建
        loss = tf.reduce_mean(tf.square(input_y - y_))

        # e.构建优化器
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
        train_op = optimizer.minimize(loss=loss)

        with tf.Session() as sess:
            # 在有变量存在时需要进行变量初始化操作
            sess.run(tf.global_variables_initializer())
            # 执行训练的过程 训练100次
            loop = 100
            for step in range(loop):
                _, loss_ = sess.run(fetches=[train_op, loss], feed_dict={input_x: x, input_y: y})
                print(f'第{step}次训练，损失函数为{loss_}')

            y_predict = sess.run(fetches=y_, feed_dict={input_x: x})
            plt.plot(x, y, 'ro')
            plt.plot(x, y_predict, 'g-')
            plt.show()


train()
