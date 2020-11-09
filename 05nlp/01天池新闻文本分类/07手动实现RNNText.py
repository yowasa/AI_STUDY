import logging
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.python import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

# 1.读取文件
train_df = pd.read_csv('/Users/yowasa/Documents/天池入门NLP - 新闻文本分类/train_set.csv', sep='\t', nrows=20000)

# 2.统计分析
all_lines = ' '.join(list(train_df['text']))
word_count = Counter(all_lines.split(" "))
word_count = sorted(word_count.items(), key=lambda d: d[1], reverse=True)
print(f'单词种类{len(word_count)}')
# 保留90%的词
num_words = int(len(word_count) * 0.9)

# 文字编码
# 单词转化 正常如果是中文需要给他编码 虽然案例中已经将其编码为数字 我这当成中文来考虑
# 保留一定的词 舍弃部分低频词
'''
num_words=None,保留多少词 无论训练多少最多保留这么多词 基于词频
filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',过滤的字符
lower=True,是否转化为小写
split=' ',分隔符
char_level=False, 是否将每个字符识别为token
oov_token=None,未识别单词的替代
'''
token = Tokenizer(num_words=num_words, oov_token='<OOV>')
token.fit_on_texts(train_df['text'])

sequence = token.texts_to_sequences(train_df['text'])
padded = pad_sequences(sequence, padding='post', maxlen=3000, truncating='post')

# 4.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(np.array(padded), np.array(train_df['label']), test_size=0.25,
                                                    random_state=22)
# 5.模型构建
input_size = 3000
model = keras.models.Sequential([
    keras.layers.embeddings.Embedding(input_dim=num_words, output_dim=128, input_length=input_size),
    keras.layers.Bidirectional(keras.layers.LSTM(units=128, return_sequences=False)),
    # keras.layers.Bidirectional(keras.layers.LSTM(units=64, return_sequences=False, go_backwards=True)),
    # keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu', kernel_regularizer=keras.regularizers.l2()),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(14, activation='softmax'),
])

model.summary()

model.compile(loss="sparse_categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

# 训练
rnn_his = model.fit(X_train, Y_train,
                    epochs=5,
                    validation_split=0.2)

val_pred = model.predict(X_test)


print(f1_score(Y_test, np.argmax(val_pred, axis=1), average='macro'))


