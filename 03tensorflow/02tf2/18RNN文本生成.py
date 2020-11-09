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

from urllib import request

file_path = './RNN'
if not os.path.exists(file_path):
    os.mkdir(file_path)
txt_path = './RNN/shakespeare.txt'
if not os.path.exists(txt_path):
    txt_path = os.path.join(file_path, 'shakespeare.txt')
    url = 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
    request.urlretrieve(url, txt_path)

text = open(txt_path, 'r').read()

print(len(text))
print(text[0:100])

# 1.生成词表
# set() 函数创建一个无序不重复元素集
# sorted() 函数对所有可迭代的对象进行排序操作
vocab = sorted(set(text))
print(len(vocab))
print(vocab)

# 2.建立字符与id的对应
# enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
# 同时列出数据和数据下标，一般用在 for 循环当中
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = np.array(vocab)

# 3.将词表数据都转成id
# 将文本转化为id值
text_as_int = np.array([char2idx[c] for c in text])
print(text[0:10])
print(text_as_int[0:10])


# 4.对文本输入做出输出：abcd->bcd*
def split_input_target(id_text):
    """
    abcde -> abcd, bcde
    """
    return id_text[0:-1], id_text[1:]


# 转换为dataset，是词的dataset
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
for _ in char_dataset.take(3):
    print(_, idx2char[_.numpy()])

# 转变为句子的dataset
seq_length = 100
seq_dataset = char_dataset.batch(seq_length + 1,
                                 # 最后一组batch不够的话，就去掉
                                 drop_remainder=True)
for _ in seq_dataset.take(2):
    print(_)
    # repr:使特殊字符也显示出来
    print(repr(''.join(idx2char[_.numpy()])))

# 使用map方法对dataset进行处理

seq_dataset = seq_dataset.map(split_input_target)

for item_input, item_output in seq_dataset.take(2):
    print(item_input.numpy())
    print(item_output.numpy())


batch_size = 64
buffer_size = 10000

seq_dataset = seq_dataset.shuffle(buffer_size).batch(
    batch_size, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024


def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = keras.models.Sequential([
        keras.layers.Embedding(vocab_size, embedding_dim,
                               batch_input_shape=[batch_size, None]),
        keras.layers.SimpleRNN(units=rnn_units,
                               return_sequences=True),
        keras.layers.Dense(vocab_size)
    ])
    return model


model = build_model(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units,
    batch_size=batch_size)
model.summary()

for input_example_batch, target_example_batch in seq_dataset.take(1):
    example_batch_predictions = model(input_example_batch)
    print(example_batch_predictions.shape)
    # 64是batch_size，100是每个句子的长度，65是一个概率分布

# 基于输出的65，进行随机采样
# 当选取概率最大的值时，被称为贪心策略，当随机采样时，为随机策略
# logits：在分类任务中，softmax之前的值就为logits
sample_indices = tf.random.categorical(
    logits=example_batch_predictions[0], num_samples=1)
# (100, 65) -> (100, 1)
print(sample_indices)
# 变成向量
sample_indices = tf.squeeze(sample_indices, axis=-1)
print(sample_indices)

print("Input:", repr("".join(idx2char[input_example_batch[0]])))
print()
print("Output:", repr("".join(idx2char[target_example_batch[0]])))
print()
print("Predictions:", repr("".join(idx2char[sample_indices])))


# 自定义损失函数
def loss(labels, logits):
    return keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)

model.compile(optimizer= 'adam', loss = loss)
example_loss = loss(target_example_batch, example_batch_predictions)
print(example_loss.shape)
print(example_loss.numpy().mean())

output_dir = "./text_generation_checkpoints"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
checkpoint_prefix = os.path.join(output_dir, 'ckpt_{epoch}')
checkpoint_callback = keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

epochs =50
history = model.fit(seq_dataset, epochs=epochs, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(output_dir)

# 载入训练好的模型
model2 = build_model(vocab_size,
                    embedding_dim,
                    rnn_units,
                    batch_size = 1)
# 载入权重
model2.load_weights(tf.train.latest_checkpoint(output_dir))
# 设置输入的size
model2.build(tf.TensorShape([1, None]))

# 文本生成的流程
# 初始是一个字符串char -> A,
# A -> model -> b
# A.append(b) -> Ab
# Ab -> model -> c
# Ab.append(c) -> Abc
# Abc -> model -> ...
model2.summary()