import fasttext
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

mpl.rcParams["font.family"] = 'Arial Unicode MS'

train_df = pd.read_csv('/Users/yowasa/Documents/天池入门NLP - 新闻文本分类/train_set.csv', sep='\t', nrows=15000)

train_df['label_ft'] = '__label__' + train_df['label'].astype(str)
train_df[['text', 'label_ft']].iloc[:-5000].to_csv('train.csv', index=None, header=None, sep='\t')

model = fasttext.train_supervised('train.csv', lr=1, wordNgrams=3,dim=500,
                                  verbose=2, minCount=1, epoch=25, loss="softmax")

val_pred = [model.predict(x)[0][0].split('__')[-1] for x in train_df.iloc[-5000:]['text']]

print(f1_score(train_df['label'].values[-5000:].astype(str), val_pred, average='macro'))
