import tensorflow as tf
import numpy as np

# 基础操作

# 常量
t = tf.constant([[1, 2, 3], [4, 5, 6]])
print(t)
print(t[:, 1:])
print(t[:, 1])

t_1 = tf.constant([[1., 2., 3.], [4., 5., 6.]])
print(t_1)
print(t_1[:, 1:])
print(t_1[:, 1])

# 常量的操作
print('常量的操作' + '*' * 100)
# 加
print(t + 10)
# 平方
print(tf.square(t))
# 矩阵乘以自己的转置(得用@)
# python中矩阵的叉乘：用@或者tf.matmul(A,C)或者np.dot(A,C)
# python中矩阵的点乘：用*或者tf.multiply(A,C)
print(t @ tf.transpose(t))

# tensorflow对象和numpy的转化
print('tensorflow对象和numpy的转化' + '*' * 100)
print(t.numpy())
print(np.square(t))
np_t = np.array([[1, 2, 3], [4, 5, 6]])
print(tf.constant(np_t))

# tensor为0维，也就是一个数，称为 Scalars
print('tensor为0维，也就是一个数，称为 Scalars' + '*' * 100)
t = tf.constant(2.63554)
print(t.numpy())
print(t.shape)

# 字符串 string
print('字符串' + '*' * 100)
t = tf.constant("cafe")
print(t)
print(tf.strings.length(t))
# 获得utf-8编码长度
print(tf.strings.length(t, unit="UTF8_CHAR"))
# 从unicode转化为UTF-8(UTF-8是unicode编码的一种实现)
print(tf.strings.unicode_decode(t, "utf8"))

# 字符串数组
print('字符串数组' + '*' * 100)
t = tf.constant(["cafe", "coffee", "咖啡"])
print(tf.strings.length(t, unit="UTF8_CHAR"))
r = tf.strings.unicode_decode(t, "utf8")
print(r)

# 不规则张量
print('不规则张量' + '*' * 100)
r = tf.ragged.constant([[11, 22], [21, 22, 33], [1], [4, 5, 6, 9, 3]])
# 索引
print(r)
print(r[1])
print(r[1:3])

# 拼接操作
print('拼接操作' + '*' * 100)
r2 = tf.ragged.constant([[51, 52], [], [77]])
# 行拼接
print(tf.concat([r, r2], axis=0))
# # 列拼接会出错，因为行数不同
# print(tf.concat([r, r2], axis = 1))
# 当行数相同时，将列重新拼接
r3 = tf.ragged.constant([[1, 2], [2, 22, 33], [], [4, 9, 3]])
print(tf.concat([r, r3], axis=1))

# 将不规则张量转变为普通的张量
# 所有0值都在正常值后面
print('将不规则张量转变为普通的张量' + '*' * 100)
print(r.to_tensor())

# 稀疏张量
print('稀疏张量' + '*' * 100)
# indices:values的位置
# values:所填值的大小
# dense_shape:矩阵的真实大小
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                    values=[1., 2., 3.],
                    dense_shape=[3, 4])
print(s)
# 转化成普通的tensor
print(tf.sparse.to_dense(s))

# 稀疏张量支持的操作
print('稀疏张量支持的操作' + '*' * 100)
s2 = s * 2.0
print(s2)
try:
    s3 = s + 1  # 加法不行
except TypeError as ex:
    print(ex)
s4 = tf.constant([[10., 20.],
                  [30., 40.],
                  [50., 60.],
                  [70., 80.]])
# 稀疏张量与密集张量相乘
print(tf.sparse.sparse_dense_matmul(s, s4))

# 稀疏张量 常见的错误，顺序不对，转化成普通tensor会报错
print('稀疏张量 常见的错误' + '*' * 100)
s5 = tf.SparseTensor(indices=[[0, 2], [0, 1], [2, 3]],
                     values=[1., 2., 3.],
                     dense_shape=[3, 4])
print(s5)
# 将顺序reorder即可
s6 = tf.sparse.reorder(s5)
print(tf.sparse.to_dense(s6))
print('*' * 100)

# 变量相关操作
print('变量相关操作' + '*' * 100)
v = tf.Variable([[1., 2., 3.], [4., 5., 6., ]])
print(v)
print(v.value())
print(v.numpy())

# 变量重新赋值(常量不可)
v.assign(2 * v)
print(v.numpy())
v[0, 1].assign(42)
print(v.numpy())
v[1].assign([7., 8., 9.])
print(v.numpy())

# 赋值是不能用等号
print('赋值是不能用等号' + '*' * 100)
try:
    v[1] = [7., 8., 9.]
except TypeError as ex:
    print(ex)
