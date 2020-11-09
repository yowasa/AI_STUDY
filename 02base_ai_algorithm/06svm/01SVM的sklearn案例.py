import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
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
C=1.0, 惩罚项系数 分错容忍度 越大越不容许分错
kernel='rbf', 核函数 rbf为高斯核函数
degree=3, 多项式核函数的深度
gamma='scale',核函数的gamma值
coef0=0.0, 多项式核函数中的截距
shrinking=True, 
probability=False, 是否计算概率值
tol=1e-3, 
cache_size=200, 
class_weight=None, 特征权重
verbose=False, 是否打印过程
max_iter=-1, 最大迭代次数
decision_function_shape='ovr', 多分类时采用的算法 ovo或ovr
break_ties=False,
random_state=None 随机数种子
"""
algo = SVC(probability=True)

algo.fit(X_train, Y_train)
train_predict = algo.predict(X_train)
test_predict = algo.predict(X_test)

print(f'模型在训练集上的效果r:{algo.score(X_train, Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test, Y_test)}')
print(f'分类评估报告（训练集）:{classification_report(Y_train,train_predict)}')
print(f'分类评估报告（测试集）:{classification_report(Y_test,test_predict)}')
print(f'支持向量下标：{algo.support_}')
print(f'支持向量值：{algo.support_vectors_}')

# print(f'SVM训练得出的权重系数（仅LearnerSVM可以）：{algo.coef_}')
# print(f'SVM训练得出的偏置系数（仅LearnerSVM可以）：{algo.intercept_}')
# print(f'返回的预测概率：{algo.predict_proba(X_test)}')

