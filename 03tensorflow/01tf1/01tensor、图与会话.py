import tensorflow as tf
import numpy as np


def create_graph1():
    print(f"当前图为{tf.get_default_graph}")
    '''
    def constant_v1(value, dtype=None, shape=None, name="Const", verify_shape=False)
    定义一个不许修改的常量tensor对象
    value可以使任意python基本数据类型(数值 布尔 字符串 数组 元组)及np的数组类型
    dtype 明确给定常量对象的数据类型 如果不给定从value中获取 
    shape 给定的tensor数据形状 不给定从value获取 给定会进行强行转换
    '''
    # 定义tf常量tensor对象 dtype必须要为tf.float32 因为b为自动转化的32 类型不同计算时会报错
    a = tf.constant(value=5.0, dtype=tf.float32, shape=None, name='a')
    # 不给定dtype和shape内部可以自动转换
    b = tf.constant(value=8.0)

    # 给定name为v1的变量 值为a与随机数相加
    v1 = tf.add(a, y=np.random.random_sample(), name='v1')
    # 使用tf的random_normal进行随机数生成 shape为None代表数字 dtype指定为float name不给定
    v2 = tf.add(b, y=tf.random_normal(shape=[], dtype=tf.float32))

    result = tf.multiply(x=v1, y=v2)
    print(a, b, v1, v2, result)
    return a, b, v1, v2, result


def create_graph2():
    print(f"当前图为{tf.get_default_graph}")
    a = tf.constant(value=5.0)
    b = tf.constant(value=8.0)

    # 可以不适用tf函数 直接使用+-*/
    v1 = a + np.random.random_sample()
    v2 = b + tf.random_normal(shape=[], dtype=tf.float32)
    result = v1 * v2
    print(a, b, v1, v2, result)
    return result


def create_graph3():
    print(f"当前图为{tf.get_default_graph}")
    a = tf.constant(value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=tf.float32, shape=[3, 5])
    b = tf.constant(value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=tf.float32, shape=[5, 3])

    # 可以不适用tf函数 直接使用+-*/
    v1 = a + np.random.random_sample()
    v2 = b + tf.random_normal(shape=[], dtype=tf.float32)
    # 矩阵相乘时不能直接使用*和multiply 需要使用matmul
    # result = v1 * v2
    result = tf.matmul(x=v1, y=v2)
    print(a, b, v1, v2, result)
    return result


def create_graph4():
    print(f"当前图为{tf.get_default_graph}")
    a = tf.constant(value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=tf.float32, shape=[3, 5])
    # 创建一个新图并使用（设为默认）
    with tf.Graph().as_default():
        print(f"with语句中的图为{tf.get_default_graph}")
        b = tf.constant(value=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15], dtype=tf.float32, shape=[5, 3])

    # 不同的图输出的name可能一致 且不同图中的内容不允许不同图中的tensor对象互相操作
    result = tf.matmul(a, b)
    print(a, b, result)
    return result


# create_graph1()
# create_graph2()
# # 观察到图的地址是一致的 说明实际上是一个默认图
# create_graph3()
# create_graph4()


# 会话 session 指的是进入了执行流程 一个session对象是和一个图关联起来的 如果未指定则使用默认图
a, b, v1, v2, result = create_graph1()

'''
def __init__(self, target='', graph=None, config=None):
target 一个字符串 暂不考虑
graph 运行哪个图
config session相关配置信息
'''
sess = tf.Session()
'''
def run(self, fetches, feed_dict=None, options=None, run_metadata=None):
fetches 给定具体获取那些tensor值 可以使一个参数 也可以是多个 给定多个时一次性获取多个但是图只执行一次
feed_dict 如果执行图需要输入数据 从feed给定  
'''
print(sess.run(a))
print(sess.run(b))
print(sess.run(v1))
print(sess.run(v2))
print(sess.run(result))
print(sess.run([a, b, v1, v2, result]))
# 会话关闭
sess.close()

# 关闭会话后再执行会报错
# print(sess.run(a))

# 第二种方式
with tf.Session() as sess:
    print(f"当前默认图为{tf.get_default_graph}")
    print(f"当前默认会话为{tf.get_default_session}")
    print(f"会话使用的图为{sess.graph}")
    print(sess.run([a, b, v1, v2, result]))
    # 第二种方式执行时可以使用tensor对象的eval方法 或者操作对象的run方法进行执行
    # 执行时需要有默认会话
    print(result.eval())
