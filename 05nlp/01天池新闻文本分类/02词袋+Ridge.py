import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import f1_score
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rcParams["font.family"] = 'Arial Unicode MS'


train_df = pd.read_csv('/Users/yowasa/Documents/天池入门NLP - 新闻文本分类/train_set.csv', sep='\t' ,nrows=15000)

vectorizer = CountVectorizer(max_features=3000)
train_test = vectorizer.fit_transform(train_df['text'])

clf = RidgeClassifier()
clf.fit(train_test[:10000], train_df['label'].values[:10000])
val_pred = clf.predict(train_test[10000:])
print(f1_score(train_df['label'].values[10000:], val_pred, average='macro'))

