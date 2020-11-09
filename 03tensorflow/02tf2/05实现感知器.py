import tensorflow as tf
import numpy as np
from tensorflow.python import keras

# 定义顺序的神经元 第一层 输入层 只有一个单元 输入特征为1个
model = keras.Sequential([
    keras.layers.Dense(units=1, input_shape=[1])
])
# 损失函数使用均方差 优化器使用随机梯度下降进行求解
model.compile(optimizer='sgd', loss='mean_squared_error')
X = np.array([-1., 0., 1., 2., 3., 4.], dtype=float)
Y = np.array([-3., -1., 1., 3., 5., 7], dtype=float)
# 进行训练
model.fit(X, Y, epochs=500)
y_pre = model.predict([10.])
print(y_pre)
