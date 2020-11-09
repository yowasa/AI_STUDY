import tensorflow as tf

import os


# os.environ['CUDA_VISIBLE_DEVICE']="0,1" #允许使用第一第二块gpu
# os.environ['CUDA_VISIBLE_DEVICE']="0" #只允许使用第一块
# os.environ['CUDA_VISIBLE_DEVICE']="-1" #不允许使用gpu
# 进行设备控制
def device():
    with tf.Graph().as_default():
        # 构建图
        # 让前两行代码在cpu0上运行
        with tf.device('/cpu:0'):
            x = tf.placeholder(dtype=tf.float32, shape=None, name='x')
            # 这是一个变量
            mul_x = tf.Variable(dtype=float, initial_value=1.0, name='sum_x')
        # 让这两行代码在cpu1上运行
        with tf.device('/gpu:0'):
            # 进行阶乘
            tmp = tf.multiply(mul_x, x)
            # 将tmp的值赋值给变量sum_x
            assign_op = tf.assign(ref=mul_x, value=tmp)

        # 控制依赖 在执行其中的语句之前一定会先执行那些操作
        with tf.control_dependencies([assign_op]):
            y = mul_x * 3
        # 显示设备的log信息 默认情况下用第一个cpu运行
        # 用gpu版本的tensorflow会默认到第一个gpu 如果有一个以上 默认除了第一个gpu都不参与运算 但是所有gpu都会分配内存
        # allow_soft_placement 当找不到设备是默认到cpu0运行
        with tf.Session(config=tf.ConfigProto(log_device_placement=True, allow_soft_placement=True)) as sess:
            sess.run(tf.global_variables_initializer())
            n = 5
            for data in range(1, n + 1):
                r = sess.run(y, feed_dict={x: data})
                print(r)


device()
