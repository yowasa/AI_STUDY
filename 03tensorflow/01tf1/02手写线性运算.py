import tensorflow as tf
import numpy as np


# x = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
# w = [-5, 3, 1.5]
# b = 3
# x 4*3 四条数据 每个三个特征
# w 3*1 三个变量
# b 一个偏置项
def f1():
    with tf.Graph().as_default():
        # 定义三个源节点的信息
        x = tf.constant(value=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tf.float32, shape=[4, 3], name='x')
        w = tf.constant(value=[[-5], [3], [1.5]], dtype=tf.float32, shape=[3, 1], name='w')
        b = tf.constant(value=[3], dtype=tf.float32, shape=[1], name='b')

        # 基于规则调用操作
        y_ = tf.matmul(x, w) + b
        print((x, w, b, y_))

        with tf.Session() as sess:
            y = sess.run(y_)
            print(y)

# 使用变量的形式
def f2():
    with tf.Graph().as_default():
        # 定义三个源节点的信息
        x = tf.constant(value=[[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]], dtype=tf.float32, shape=[4, 3], name='x')
        '''
        initial_value=None, 初始化参数
        trainable=None, 是否可以训练 默认为True
        collections=None, 内部存储保存在哪个集合中间
        validate_shape=True, 指定在更新该变量的前后是否要求shap形状一直 True是更新前后shap必须一样
        name=None, tensor名称
        dtype=None, 数据类型
        '''
        w=tf.Variable(initial_value=[[-5], [3], [1.5]], dtype=tf.float32,name='w')
        b = tf.Variable(initial_value=[3], dtype=tf.float32,  name='b')

        # 基于规则调用操作
        y_ = tf.matmul(x, w) + b
        print((x, w, b, y_))

        #全局所有的变量初始化
        init_op=tf.global_variables_initializer()

        with tf.Session() as sess:
            # 在有变量存在时需要进行变量初始化操作
            sess.run(init_op)


            y = sess.run(y_)
            print(y)

# 使用placeholder接收训练数据 run时使用feed传入
def f3():
    with tf.Graph().as_default():
        # 定义三个源节点的信息
        # 给定一个占位符 用于训练时feed数据
        '''
        def placeholder(dtype, shape=None, name=None):
        dtype 数据类型
        shape 数据形状
        name tensor名称
        '''
        x = tf.placeholder(dtype=tf.float32, shape=[4, 3], name='x')
        c = tf.placeholder_with_default(input=1.0, shape=[], name='c')
        w = tf.Variable(initial_value=[[-5], [3], [1.5]], dtype=tf.float32, name='w')
        b = tf.Variable(initial_value=[3], dtype=tf.float32, name='b')

        # 基于规则调用操作
        y_ = tf.matmul(x, w) + b
        y_2 = y_ + c
        print((x, w, b, y_))

        # 全局所有的变量初始化
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            # 在有变量存在时需要进行变量初始化操作
            sess.run(init_op)
            y1,y2 = sess.run(fetches=[y_, y_2], feed_dict={x: [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]})
            print(y1,y2)


# placeholder对应维度改为None 变为可变长度的传入
def f4():
    with tf.Graph().as_default():
        # 定义三个源节点的信息
        # 给定一个占位符 用于训练时feed数据
        '''
        def placeholder(dtype, shape=None, name=None):
        dtype 数据类型
        shape 数据形状
        name tensor名称
        '''
        x = tf.placeholder(dtype=tf.float32, shape=[None, 3], name='x')
        c = tf.placeholder_with_default(input=1.0, shape=[], name='c')
        w = tf.Variable(initial_value=[[-5], [3], [1.5]], dtype=tf.float32, name='w')
        b = tf.Variable(initial_value=[3], dtype=tf.float32, name='b')

        # 基于规则调用操作
        y_ = tf.matmul(x, w) + b
        y_2 = y_ + c
        print((x, w, b, y_))

        # 全局所有的变量初始化
        init_op = tf.global_variables_initializer()

        with tf.Session() as sess:
            # 在有变量存在时需要进行变量初始化操作
            sess.run(init_op)
            y1,y2 = sess.run(fetches=[y_, y_2], feed_dict={x: [[1, 2, 3], [4, 5, 6]]})
            print(y1,y2)


# f1()
# f2()
# f3()
# f4()