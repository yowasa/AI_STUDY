import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import pandas as pd
import os
import sys
import time
import tensorflow as tf
from tensorflow import keras

# tfrecord是一个文件格式
# 1.tf.train.Example
#  1.1 tf.train.Features:{"key": tf.train.Feature}
#   1.1.1 tf.train.Feature:tf.train.ByteList/FloatList/Int64List

# 将字符串列表转化为utf-8编码
favorite_books = [name.encode('utf-8')
                  for name in ['machine learning', 'cc150']]
# 生成bytes_list
favorite_books_bytelist = tf.train.BytesList(value=favorite_books)
print(favorite_books_bytelist)
# 生成float_list
hours_floatlist = tf.train.FloatList(value=[15.5, 9.5, 7.0, 8.0])
print(hours_floatlist)
# 生成int64_list
age_int64list = tf.train.Int64List(value=[42])
print(age_int64list)

# 生成tf.train.feature属性(key和value的键值对)，在将这些单独feature整合成features
features = tf.train.Features(
    feature={
        "favorite_books": tf.train.Feature(
            bytes_list=favorite_books_bytelist),
        "hours": tf.train.Feature(
            float_list=hours_floatlist),
        "age": tf.train.Feature(int64_list=age_int64list),
    }
)
print(features)

# tf.train.Example在tf.train.Features外面又多了一层封装
example = tf.train.Example(features=features)
print(example)
# 将example序列化，压缩以减少size
serialized_example = example.SerializeToString()
print(serialized_example)

# 将example存入一个文件下,生成一个tfrecords文件
output_dir = 'tfrecord_basic'
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
filename = 'test.tfrecords'
filename_fullpath = os.path.join(output_dir, filename)
# 打开tfrecords文件并写入序列化后的数据
with tf.io.TFRecordWriter(filename_fullpath) as writer:
    for i in range(3):
        writer.write(serialized_example)
print('存储完毕' + '*' * 100)

# 将tfrecord读取为dataset形式
dataset = tf.data.TFRecordDataset([filename_fullpath])
for serialized_example_tensor in dataset:
    print(serialized_example_tensor)

# 定义解析后的形式
expected_feature = {
    "favorite_books": tf.io.VarLenFeature(dtype=tf.string),
    "hours": tf.io.VarLenFeature(dtype=tf.float32),
    "age": tf.io.FixedLenFeature([], dtype=tf.int64),
}

for serialized_example_tensor in dataset:
    # 将example解析
    example = tf.io.parse_single_example(
        serialized_example_tensor,
        expected_feature)
    # 将稀疏矩阵解析出来
    books = tf.sparse.to_dense(example["favorite_books"])
    for book in books:
        print(book.numpy().decode("UTF-8"))
print('解析完毕' + '*' * 100)

# 将tfrecord存成压缩文件
filename_fullpath_zip = filename_fullpath + '.zip'
# 定义压缩操作
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter(filename_fullpath_zip, options) as writer:
    for i in range(3):
        writer.write(serialized_example)
print('压缩测试完毕' + '*' * 100)

dataset_zip = tf.data.TFRecordDataset([filename_fullpath_zip],
                                      compression_type='GZIP')
for serialized_example_tensor in dataset_zip:
    # 将example解析
    example = tf.io.parse_single_example(
        serialized_example_tensor,
        expected_feature)
    # 将稀疏矩阵解析出来
    books = tf.sparse.to_dense(example["favorite_books"])
    for book in books:
        print(book.numpy().decode("UTF-8"))
print('解压缩解析完毕' + '*' * 100)
