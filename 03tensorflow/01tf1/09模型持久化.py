'''
1.持久化模型为二进制文件 需要的时候加载
2.持久化模型参数到数据库 需要的时候加载恢复模型进行预测
3.直接将模型预测结果村吃到数据库中
'''

import tensorflow as tf
import numpy as np


# 模型存储
def f1():
    with tf.Graph().as_default():
        v1 = tf.Variable(initial_value=5.0, name='v2')
        v2 = tf.Variable(initial_value=tf.random_normal(shape=[], mean=0.0, stddev=1.0), name='v2')
        result = v1 + v2
        # 创建一个持久化对象
        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            result_ = sess.run(result)
            print(result_)
            saver.save(sess, './09/model.cpk')


# 模型加载
def f2():
    with tf.Graph().as_default():
        v1 = tf.Variable(initial_value=5.0, name='v2')
        v2 = tf.Variable(initial_value=tf.random_normal(shape=[], mean=0.0, stddev=1.0), name='v2')
        result = v1 + v2
        # 创建一个持久化对象
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # 直接恢复变量 不需要初始化
            saver.restore(sess, './09/model.cpk')
            result_ = sess.run(result)
            print(result_)


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

        # 创建一个持久化对象
        '''
        var_list=None,持久化哪些变量 默认是所有
        max_to_keep=5,指定最多同时保留最近多少份模型
        '''
        saver = tf.train.Saver()

        with tf.Session() as sess:
            # 使用check_point 可以在已有的模型上继续训练
            ckpt = tf.train.get_checkpoint_state('./09T/model.cpk')
            if ckpt:
                saver.restore(sess, './09T/model.cpk')
            else:
                # 在有变量存在时需要进行变量初始化操作
                sess.run(tf.global_variables_initializer())
            # 执行训练的过程 训练100次
            loop = 100
            for step in range(loop):
                _, loss_ = sess.run(fetches=[train_op, loss], feed_dict={input_x: x, input_y: y})
                print(f'第{step}次训练，损失函数为{loss_}')
                saver.save(sess, './09T/model.cpk', global_step=step)
            # y_predict = sess.run(fetches=y_, feed_dict={input_x: x})


# 实现线性模型的求解
def test():
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
        # 创建一个持久化对象
        saver = tf.train.Saver()

        with tf.Session() as sess:
            saver.restore(sess, './09T/model.cpk')
            y_predict = sess.run(fetches=y_, feed_dict={input_x: x})
            print(y_predict)


# f1()
# f2()
train()
# test()
