import tensorflow as tf


# tensorflow中有两个作用域 一个是namespace 一个是variable_scope 即变量作用域
# variable_scope是给变量增加前缀 namespace是给op_name 也就是操作增加前缀
# 使用tf.Variable 会受到namespace和variable_scope的影响 而使用get_variable只会受到variable_scope的影响

def f1():
    # 创建一个新的tensor变量
    w = tf.Variable(initial_value=tf.random_normal(shape=[2], mean=0.0, stddev=1.0),
                    dtype=tf.float32, name='w')
    return w


def f2():
    # 获取一个名称为name的tensor变量
    '''
    name, 给定名称 必须要有
    shape=None, 形状
    dtype=None, 数据类型
    initializer=None, 初始值的产生方式 生成器
    regularizer=None, 正则化器
    trainable=None, 是否参与模型训练
    基于给定的name 从tensorflow内部获取对应的tensor变量 如果name存在则直接获取 如果不存在则使用初始化产生器创建一个新的tensor变量 
    '''
    w = tf.get_variable(name='w', shape=[2], initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0))
    return w


def h1():
    # 调用创建的时候 遇到重复名称会自动加上下标_1 _2
    w11 = f1()
    w21 = f1()
    # 而使用获取时 重复会出现问题 需要设置变量作用域为可重用
    # 使用get获取variable时只能获取曾经以variable
    w12 = f2()
    # 设置可以重用变量
    tf.get_variable_scope().reuse_variables()
    w22 = f2()
    print(w11, w21, w12, w22)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([w11, w21, w12, w22]))


def h2():
    # 使用with设置变量作用域
    with tf.variable_scope('t1'):
        w11 = f1()
        w21 = f1()
    # 设置变量作用域可重用
    # 设置初始化器为常数初始化器 如果在此variable_scope中get_variable时设置的initializer为None时 会使用此设置
    with tf.variable_scope('t2', reuse=tf.AUTO_REUSE,
                           initializer=tf.constant_initializer(22.0)):
        w12 = f2()
        w22 = f2()
    print(w11, w21, w12, w22)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run([w11, w21, w12, w22]))


# h1()
h2()
