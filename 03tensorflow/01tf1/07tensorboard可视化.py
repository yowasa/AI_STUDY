import tensorflow as tf

with tf.Graph().as_default():
    # 构建图
    with tf.name_scope('n1'):
        x = tf.placeholder(dtype=tf.float32, shape=None, name='x')
        # 这是一个变量
        mul_x = tf.Variable(dtype=float, initial_value=1.0, name='sum_x')

        # 标注可视化内容 一个名称 和想看的对象
        tf.summary.scalar(name='tmp mul x', tensor=mul_x)

        # 进行阶乘
        tmp = tf.multiply(mul_x, x)
        # 将tmp的值赋值给变量sum_x
        assign_op = tf.assign(ref=mul_x, value=tmp)

    with tf.name_scope('n2'):

        # 控制依赖 在执行其中的语句之前一定会先执行那些操作
        with tf.control_dependencies([assign_op]):
            y = mul_x * 3
            # 标注可视化内容 一个名称 和想看的对象
            tf.summary.scalar(name='each step y', tensor=y)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # 合并所有的可视化输出操作
        summary = tf.summary.merge_all()
        # 构建一个日志输出对象
        write = tf.summary.FileWriter(logdir='./07', graph=sess.graph)

        n = 5
        for data in range(1, n + 1):
            r, summary_ = sess.run([y, summary], feed_dict={x: data})
            print(r)
            # 将可视化相关信息输出到磁盘
            write.add_summary(summary=summary_, global_step=data)
        write.close()


# tensorboard --logdir="07" 查看图形界面