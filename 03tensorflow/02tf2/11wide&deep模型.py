import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras

mpl.rcParams["font.family"] = 'Arial Unicode MS'

# 经典的推荐系统模型


# 获取数据集
housing = fetch_california_housing()
# 获取数据
x_train_all, x_test, y_train_all, y_test = train_test_split(
    housing.data, housing.target, random_state=22)
x_train, x_valid, y_train, y_valid = train_test_split(
    x_train_all, y_train_all, random_state=22)

# 归一化处理
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_valid_scaled = scaler.transform(x_valid)
x_test_scaled = scaler.transform(x_test)

# wide层：输入后直接输出
# x_train.shape[1:]:表示x_train的第一行有多少列，也就是有每条数据有多少特征
input = keras.layers.Input(shape=x_train.shape[1:])
# deep层
hidden1 = keras.layers.Dense(30, activation='relu')(input)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
# 用concatenate实现wide层和deep层的拼接
concat = keras.layers.concatenate([input, hidden2])
# 输出
output = keras.layers.Dense(1)(concat)

# 将model固化下来
model = keras.models.Model(inputs=[input],
                           outputs=[output])
model.summary()

optimizer = tf.keras.optimizers.RMSprop(0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)
callbacks = [keras.callbacks.EarlyStopping(
    patience=5, min_delta=1e-2)]

history = model.fit(x_train_scaled, y_train,
                    validation_data=(x_valid_scaled, y_valid),
                    epochs=100,
                    callbacks=callbacks)


def plot_learning_curves(history):
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()


plot_learning_curves(history)

print(model.evaluate(x_test_scaled, y_test, verbose=0))
