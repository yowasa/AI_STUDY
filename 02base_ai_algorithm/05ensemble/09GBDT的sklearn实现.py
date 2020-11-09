import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
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
loss: Any = 'deviance',采用对数计算分类情况
learning_rate: Any = 0.1,学习率
n_estimators: Any = 100,最终训练的子模型数量
subsample: Any = 1.0,结合bagging思想采样
criterion: Any = 'friedman_mse',衡量指标 mse方式（差平方)
min_samples_split: Any = 2,树分裂的最小样本数目
min_samples_leaf: Any = 1,叶子节点最小样本数目
min_weight_fraction_leaf: Any = 0.,样本权重的最小加成参数（暂无用）
max_depth: Any = 3,最大树深
min_impurity_decrease: Any = 0.,分裂导致不纯度减少大于等于该值则分裂
min_impurity_split: Any = None,分裂提前停止阈值，一个节点不纯度大于此阈值才能分裂
init: Any = None,初始值 可以不给
random_state: Any = None,随机数种子
max_features: Any = None,从多少个特征中寻找最优划分 None 全部特征 int 具体值 float 占比 auto 为sqrt sqrt 特征开根号 log2 就log2
verbose: Any = 0,是否打印过程
max_leaf_nodes: Any = None,最多允许的叶子节点数目 None 不限制
warm_start: Any = False,是否预热（重用之前的模型进行训练） 默认否
presort: Any = 'deprecated',
validation_fraction: Any = 0.1,
n_iter_no_change: Any = None,
tol: Any = 1e-4,
ccp_alpha: Any = 0.0 最小剪枝系数 α大于该值则不进行修剪
"""
algo = GradientBoostingClassifier()

algo.fit(X_train, Y_train)
train_predict = algo.predict(X_train)
test_predict = algo.predict(X_test)

print(f'模型在训练集上的效果r:{algo.score(X_train, Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test, Y_test)}')
print(f'分类评估报告（训练集）:{classification_report(Y_train,train_predict)}')
print(f'分类评估报告（测试集）:{classification_report(Y_test,test_predict)}')
print(f'特征重要性权重：{algo.feature_importances_}')

