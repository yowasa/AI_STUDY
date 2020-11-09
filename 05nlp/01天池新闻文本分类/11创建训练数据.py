from tensorflow.python.keras.preprocessing.text import Tokenizer
import logging
from collections import Counter

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
import numpy as np
import pickle


# # 对数据进行token
# data_file = '/Users/yowasa/Documents/天池入门NLP - 新闻文本分类/train_set.csv'
# train_df = pd.read_csv(data_file, sep='\t', encoding='UTF-8')
# # 6874为vocab行数 使用[UNK]填充袋外数据 用空格作为分割符
# tokenizer = Tokenizer(num_words=6874, oov_token='[UNK]', split=" ")
# tokenizer.fit_on_texts(train_df['text'])
# with open('./data/tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, file=handle, protocol=pickle.HIGHEST_PROTOCOL)

# loading tokenizer
with open('./data/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

data_file = '/Users/yowasa/Documents/天池入门NLP - 新闻文本分类/train_set.csv'
train_df = pd.read_csv(data_file, sep='\t', encoding='UTF-8', nrows=1000)
data = train_df['text']
print(f'原始数据：{data[1]}')
sequence = tokenizer.texts_to_sequences(train_df['text'])
print(f'token转化后的数据：{sequence[1]}')
# 对数据进行处理
