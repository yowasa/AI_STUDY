import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report

# 设置中文支持
mpl.rcParams["font.family"] = 'Arial Unicode MS'

names = ['A', 'B', 'C', 'D', 'cla', ]
df = pd.read_csv('../../data_set/iris.data', names=names)
df.info()

X = df[names[0:-1]]
Y = df[names[-1]]
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

knn=KNeighborsClassifier(n_neighbors=10)
"""
base_estimator: Any = None, 基础分类器 默认为决策树
n_estimators: Any = 10,最终训练的子模型数量
max_samples: Any = 1.0, 每个子模型训练时使用多少原始数据（百分比 1.0是100%）
max_features: Any = 1.0,每个子模型训练时使用多少特征（百分比 1.0是100%）
bootstrap: Any = True, 每个子模型的训练数据时是否做有放回的重采样 默认重采样
bootstrap_features: Any = False, 每个子模型的特征属性是否使用使用重采样的方式（特征可能会被重复使用） 默认不采样
oob_score: Any = False,是否计算袋外准确率（没有参与模型训练的数据是否参与计算准确率）
warm_start: Any = False, 是否预热（重用之前的模型进行训练） 默认否
n_jobs: Any = None, 线程数
random_state: Any = None, 随机数种子
verbose: Any = 0  是否打印训练过程 0不打印 1打印
"""
algo = BaggingClassifier(base_estimator=knn,n_estimators=10,oob_score=True)

algo.fit(X_train, Y_train)
train_predict = algo.predict(X_train)
test_predict = algo.predict(X_test)

print(f'模型在训练集上的效果r:{algo.score(X_train, Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test, Y_test)}')
print(f'分类评估报告（训练集）:{classification_report(Y_train,train_predict)}')
print(f'分类评估报告（测试集）:{classification_report(Y_test,test_predict)}')

print(f'子模型列表：{algo.estimators_}')
print(f'子模型使用的样本：{algo.estimators_samples_}')
print(f'子模型使用的特征：{algo.estimators_features_}')
print(f'子模型的袋外准确率：{algo.oob_score_}')

