import tensorflow as tf
from tensorflow import keras
import numpy as np


# 导数的近似简单求法
# 普通函数
def f(x):
    return 3. * x ** 2 + 2. * x - 1


# 求导方法
def approximae_derivative(f, x, eps=1e-3):
    return (f(x + eps) - f(x - eps)) / (2. * eps)


# 求导结果
print(f'普通函数近似求导结果：{approximae_derivative(f, 1.)}')


# 偏导的简单近似求法
def g(x1, x2):
    return (x1 + 5) * (x2 ** 2)


def approximate_gradient(g, x1, x2, eps=1e-3):
    dg_x1 = approximae_derivative(lambda x: g(x, x2), x1, eps)
    dg_x2 = approximae_derivative(lambda x: g(x1, x), x2, eps)
    return dg_x1, dg_x2


print(f'偏导近似求解结果：{approximate_gradient(g, 2., 3.)}')

# Tensorflow中导数的求解
print('Tensorflow中导数的求解' + '*' * 100)
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
z = g(x1, x2)
with tf.GradientTape(persistent = True) as tape:
    z = g(x1, x2)
dz_x1 = tape.gradient(z, x1)
dz_x2 = tape.gradient(z, x2)
print(dz_x1,dz_x2)
# 此时系统不自动删除Tape,需要手动删除
del tape
# 在求x2偏导时还需要在写一次 with tf.GradientTape() as tape
# 因为tape只能用一次
# 解决的办法是在tf.GradientTape（）中加入persistent = True以保存tap


# 常量不可以求导
print('常量不可以求导' + '*' * 100)
x1 = tf.constant(2.0)
x2 = tf.constant(3.0)
with tf.GradientTape() as tape:
    z = g(x1, x2)
dz_x1x2 = tape.gradient(z, [x1,x2])
print(dz_x1x2)

# 将常量变得可导
print('将常量变得可导' + '*' * 100)
# 在tape中关注变量
x1 = tf.constant(2.0)
x2 = tf.constant(3.0)
with tf.GradientTape() as tape:
    tape.watch(x1)
    tape.watch(x2)
    z = g(x1, x2)
dz_x1x2 = tape.gradient(z, [x1,x2])
print(dz_x1x2)


# 一个变量对两个函数求导 （会相加）
print('一个变量对两个函数求导' + '*' * 100)
x = tf.Variable(5.0)
with tf.GradientTape() as tape:
    z1 = 3 * x
    z2 = x ** 2
print(tape.gradient([z1, z2], x))

# 求解二阶导数，利用嵌套
print('求解二阶导数' + '*' * 100)
x1 = tf.Variable(2.0)
x2 = tf.Variable(3.0)
with tf.GradientTape(persistent=True) as outer_tape:
    with tf.GradientTape(persistent=True) as inner_tape:
        z = g(x1, x2)
    inner_grads = inner_tape.gradient(z, [x1, x2])
outer_grads = [outer_tape.gradient(inner_grad, [x1, x2])
              for inner_grad in inner_grads]
print(outer_grads)
del inner_tape
del outer_tape

# 梯度下降
print('梯度下降' + '*' * 100)
learning_rate = 0.1
x = tf.Variable(0.0)

for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    x.assign_sub(learning_rate * dz_dx)
print(x)

# 使用optimizer
print('使用optimizer优化' + '*' * 100)
learning_rate = 0.1
x = tf.Variable(0.0)

optimizer = keras.optimizers.SGD(lr = learning_rate)
for _ in range(100):
    with tf.GradientTape() as tape:
        z = f(x)
    dz_dx = tape.gradient(z, x)
    optimizer.apply_gradients([(dz_dx, x)])
print(x)
