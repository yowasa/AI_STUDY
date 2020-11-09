import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
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

"""
base_estimator: Any = None,基础学习器 默认为决策树
n_estimators: Any = 50,训练多少个子模型
learning_rate: Any = 1.,学习率
algorithm: Any = 'SAMME.R', 执行方式 SAMME或SAMME.R（快一些，要求子模型可以计算概率）
random_state: Any = None 随机数种子
"""
algo = AdaBoostClassifier()

algo.fit(X_train, Y_train)
train_predict = algo.predict(X_train)
test_predict = algo.predict(X_test)

print(f'模型在训练集上的效果r:{algo.score(X_train, Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test, Y_test)}')
print(f'分类评估报告（训练集）:{classification_report(Y_train,train_predict)}')
print(f'分类评估报告（测试集）:{classification_report(Y_test,test_predict)}')


