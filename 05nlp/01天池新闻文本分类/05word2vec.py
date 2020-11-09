import logging

import pandas as pd
from gensim.models.word2vec import Word2Vec

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')

#
# '''
# sentences=None, 词列表
# corpus_file=None,
# size=100, 词向量维度
# alpha=0.025,
# window=5, window是句子中当前词与目标词之间的最大距离 最多向前向后看几个词
# min_count=5, 低频词过滤
# sample=1e-3, 表示更高频率的词被随机下采样到所设置的阈值
# workers=3, 线程数
# min_alpha=0.0001,
# sg=0, sg=1是skip-gram算法，对低频词敏感；默认sg=0为CBOW算法。
# hs=0, hs=1表示层级softmax将会被使用，默认hs=0且negative不为0，则负采样将会被选择使用。
# '''
train_df = pd.read_csv('/Users/yowasa/Documents/天池入门NLP - 新闻文本分类/train_set.csv', sep='\t', nrows=20000)

train_df['text_list'] = train_df['text'].apply(lambda x: x.split(' '))

model = Word2Vec(train_df['text_list'], workers=8, size=100)
model.init_sims(replace=True)
model.save('w2v.bin')
