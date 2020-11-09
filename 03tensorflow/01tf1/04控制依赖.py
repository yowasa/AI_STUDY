import tensorflow as tf


# 进行数据累加操作(手动变更变量的值)
def f1():
    with tf.Graph().as_default():
        # 构建图
        input_x = tf.placeholder(dtype=tf.float32, shape=None, name='x')
        # 这是一个变量
        sum_x = tf.Variable(dtype=float, initial_value=0.0, name='sum_x')
        # 而进行add后 sum_x变成了一个tensor
        tmp = tf.add(sum_x, input_x)
        # 将tmp的值赋值给变量sum_x
        assign_op = tf.assign(ref=sum_x, value=tmp)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            datas = [1, 3, 5, 7, 6, 8]
            for data in datas:
                r = sess.run(sum_x, feed_dict={input_x: data})
                print(r)


# 进行动态更新维度操作
def f2():
    with tf.Graph().as_default():
        # 构建图
        # 构建一个可变shape
        x = tf.Variable(initial_value=[[0.0, 0.0, 0.0, 0.0, 0.0]],
                        dtype=tf.float32, shape=None, name='x', validate_shape=False)
        # 按行进行合并
        concat = tf.concat(values=[x, [[0.0, 0.0, 0.0, 0.0, 0.0]]], axis=0)
        assign_op = tf.assign(ref=x, value=concat, validate_shape=False)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(5):
                r = sess.run(assign_op)
                print(r)


# 求n的阶乘*3的操作（需要控制依赖先执行完阶乘 在乘3）
def f3():
    with tf.Graph().as_default():
        # 构建图
        x = tf.placeholder(dtype=tf.float32, shape=None, name='x')
        # 这是一个变量
        mul_x = tf.Variable(dtype=float, initial_value=1.0, name='sum_x')
        # 进行阶乘
        tmp = tf.multiply(mul_x, x)
        # 将tmp的值赋值给变量sum_x
        assign_op = tf.assign(ref=mul_x, value=tmp)

        # 控制依赖 在执行其中的语句之前一定会先执行那些操作
        with tf.control_dependencies([assign_op]):
            y = mul_x * 3

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            n = 5
            for data in range(1, n + 1):
                r = sess.run(y, feed_dict={x: data})
                print(r)
            # print(sess.run(y))


# f1()
# f2()
f3()
