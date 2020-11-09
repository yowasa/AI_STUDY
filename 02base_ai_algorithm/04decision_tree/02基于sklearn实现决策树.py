import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
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
criterion: Any = "gini", 纯度衡量标准 gini为基尼系数 entropy为信息熵
splitter: Any = "best",选择特征属性的划分方式 best最优划分 random随机划分
max_depth: Any = None, 剪枝参数 允许最大深度（前置剪枝） None为不限制
min_samples_split: Any = 2, 剪枝参数，在构建决策树时如果当前数据集中的样本数目小于该值则停止划分
min_samples_leaf: Any = 1,剪枝参数，在叶子节点中至少要求有多少个样本 int表示数量 float表示占比
min_weight_fraction_leaf: Any = 0.,样本权重的最小加成参数（暂无用）
max_features: Any = None,从多少个特征中寻找最优划分 None 全部特征 int 具体值 float 占比 auto 为sqrt sqrt 特征开根号 log2 就log2
random_state: Any = None,随机数种子
max_leaf_nodes: Any = None,最多允许的叶子节点数目 None 不限制
min_impurity_decrease: Any = 0., 杂质的减少量小于该值则分裂
min_impurity_split: Any = None, 停止生长的阈值 杂质大于该值才能分裂
class_weight: Any = None,特征权重
presort: Any = 'deprecated', 已经被移除 不再使用
ccp_alpha: Any = 0.0 最小剪枝系数 α大于该值则不进行修剪
"""
algo = DecisionTreeClassifier()

algo.fit(X_train, Y_train)
train_predict = algo.predict(X_train)
test_predict = algo.predict(X_test)

print(f'特征重要程度:{algo.feature_importances_}')
print(f'模型在训练集上的效果r:{algo.score(X_train, Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test, Y_test)}')
print(f'分类评估报告（训练集）:{classification_report(Y_train,train_predict)}')
print(f'分类评估报告（测试集）:{classification_report(Y_test,test_predict)}')

