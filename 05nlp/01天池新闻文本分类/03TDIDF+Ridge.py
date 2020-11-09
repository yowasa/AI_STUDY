import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
import matplotlib as mpl
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

mpl.rcParams["font.family"] = 'Arial Unicode MS'

train_df = pd.read_csv('/Users/yowasa/Documents/天池入门NLP - 新闻文本分类/train_set.csv', sep='\t', nrows=20000)

# ngram_range代表组合形式 1，3 是最小1最大3都组合 abc会被分成 a ab bc abc
# 如果是2，2就是 ab bc
vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 3))
X = vectorizer.fit_transform(train_df['text'])
# 4.数据分割
X_train, X_test, Y_train, Y_test = train_test_split(X, train_df['label'], test_size=0.25,
                                                    random_state=22)
clf = RidgeClassifier()
clf.fit(X_train, Y_train)

val_pred = clf.predict(X_test)
print(f1_score(Y_test, val_pred, average='macro'))

# pipeline = Pipeline(steps=[
#     ("vec", vectorizer),
#     ("clf", clf)
# ])

# param = {
# 'vec__ngram_range': [(1, 3)]
# 'vec__max_features': [2000, 3000, 4000, 5000],
# 'clf__alpha': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0],
# 'clf__fit_intercept': [True, False],
# }
# algo = GridSearchCV(estimator=pipeline, cv=3, param_grid=param, scoring='f1_macro')

# algo.fit(X_train, Y_train)
#
# # 8.模型效果评估
# print(f'模型在训练集上的效果r:{algo.score(X_train, Y_train)}')
# print(f'模型在测试集上的效果r:{algo.score(X_test, Y_test)}')
# print(f'模型参数：{algo.best_params_}')
# best_pipeline = algo.best_estimator_
#
# print(f'各个特征属性的权重:{best_pipeline.get_params()["vec"].ngram_range}')
