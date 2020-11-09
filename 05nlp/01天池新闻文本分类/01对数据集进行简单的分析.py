import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
import numpy as np

mpl.rcParams["font.family"] = 'Arial Unicode MS'

train_df = pd.read_csv('/Users/yowasa/Documents/天池入门NLP - 新闻文本分类/train_set.csv', sep='\t')

# 查看数据格式
# print(train_df.head(10))

# 首先对句子长度进行分析
# train_df['text_len'] = train_df['text'].apply(lambda x: len(x.split(' ')))
# print(train_df['text_len'].describe())
# 最大长度5w 但是75在1100以内

# 画出直方图
# plt.hist(train_df['text_len'], bins=200,range=(0,10000))
# plt.show()
# 发现大部分数据长度在2000以内

# 查看类别分布情况
# print(train_df['label'].value_counts())
# train_df['label'].value_counts().plot(kind='bar')
# plt.show()
# 发现分布不均匀 最多的3.8w最低的只有908 可能需要数据复制

from collections import Counter


# 分析使用词的情况
all_lines = ' '.join(list(train_df['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d: d[1], reverse=True)
# 一共多少词
print(len(word_count))
# 使用最多的词
print(word_count[0])
# 使用最少的词
print(word_count[-1])

# 分析哪些词出现的范围（覆盖的文章）最大
# train_df['text_unique'] = train_df['text'].apply(lambda x: ' '.join(list(set(x.split(' ')))))
# all_lines = ' '.join(list(train_df['text_unique']))
# word_count = Counter(all_lines.split(" "))
# word_count = sorted(word_count.items(), key=lambda d: int(d[1]), reverse=True)
# print(word_count[0])  # ('3750', 197997)
# print(word_count[1])  # ('900', 197653)
# print(word_count[2])  # ('648', 191975)

# 作业
# 假设3750 900 和648是标点符号 分析每篇新闻平均由多少个句子构成

# def text_count(text):
#     tls = text.split(' ')
#     sum = 1
#     for i in ['3750', '900', '648']:
#         sum += tls.count(i)
#     return sum
#
# train_df['text_count'] = train_df['text'].apply(lambda x: text_count(x))
# print(np.mean(list(train_df['text_count'])))

