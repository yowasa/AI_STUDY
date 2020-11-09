import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow.python import keras

# 在keras上下载数据集
imdb = keras.datasets.imdb
# 词语个数
vocab_size = 10000
# id的偏移量
index_from = 3
# 会按照词频进行统计，然后取排名为前一万个的
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(
    num_words=vocab_size, index_from=index_from)

# 打印第一个样本和第一个样本标签
print(train_data[0], train_labels[0])
# 打印样本的大小
print(train_data.shape, train_labels.shape)
# 打印一个样本和第二个样本向量的大小
print(len(train_data[0]), len(train_data[1]))

# 载入词表
word_index = imdb.get_word_index()
print(len(word_index))
print(list(word_index.items())[:5])
# print(word_index)

# 因为之前设置的index_from = 3，所以要将id + 3
word_index = {k: (v + 3) for k, v in word_index.items()}

# id偏移了3之后，就有了特殊的槽位增添特殊字符
word_index['<PAD>'] = 0
word_index['<START>'] = 1
word_index['<UNK>'] = 2
word_index['<END>'] = 3

reverse_word_index = dict(
    [(value, key) for key, value in word_index.items()])


# 将id解析成文本
def decode_review(text_ids):
    return " ".join(
        [reverse_word_index.get(word_id, "<UNK>") for word_id in text_ids])


decode_review(train_data[0])

# 电影评论的情感分析

# 设置句子的长度，长度高于500的会被截断，长度低于500的会被补全
max_length = 500

train_data = keras.preprocessing.sequence.pad_sequences(
    train_data,
    # 填充值
    value=word_index['<PAD>'],
    # padding可取“post”和“pre”,post将padding放在句子后面，pre将放前面
    padding='post',
    maxlen=max_length)

test_data = keras.preprocessing.sequence.pad_sequences(
    test_data,
    value=word_index['<PAD>'],
    padding='post',
    maxlen=max_length)

print(train_data[0])

# 将每个词变成长度为16的embedding向量
embedding_dim = 16
batch_size = 128

model = keras.models.Sequential([
    # keras.layers.Embedding要做的几件事：
    # 1.定义矩阵：[vocab_size, embedding_dim]
    # 2.对于每一个句子/样本，如：[1,2,3,4...],都会去矩阵中查找对应的向量，最后变成成
    # max_length * embeddding_dim的矩阵
    # 3.最后输出的大小为一个三维矩阵：batch_size * max_length * embedding_dim
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    # 对输出做合并
    # 将三维矩阵变为:batch_size * embedding_dim
    # return_sequences是指取得的是最后一步输出还是前面的输出，False表示最后一步输出
    keras.layers.SimpleRNN(units=64, return_sequences=False),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])

model.summary()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_single_rnn = model.fit(train_data,
                               train_labels,
                               epochs=30,
                               batch_size=batch_size,
                               validation_split=0.2)


def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


def plot_learning_curves(history, label, epochs, min_value, max_value):
    data = {}
    data[label] = history.history[label]
    data['val_' + label] = history.history['val_' + label]
    pd.DataFrame(data).plot(figsize=(8, 5))
    plt.grid(True)
    plt.axis([0, epochs, min_value, max_value])
    plt.show()


plot_learning_curves(history_single_rnn, 'accuracy', 30, 0, 1)
plot_learning_curves(history_single_rnn, 'loss', 30, 0, 1)
