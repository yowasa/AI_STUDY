import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
n_estimators: Any = 100,最终训练的子模型数量
criterion: Any = "gini",计算损失方法 gini为基尼系数 entropy为信息熵
max_depth: Any = None,最大树深
min_samples_split: Any = 2,树分裂的最小样本数目
min_samples_leaf: Any = 1,叶子节点最小样本数目
min_weight_fraction_leaf: Any = 0.,样本权重的最小加成参数（暂无用）
max_features: Any = "auto",从多少个特征中寻找最优划分 None 全部特征 int 具体值 float 占比 auto 为sqrt sqrt 特征开根号 log2 就log2
max_leaf_nodes: Any = None,最多允许的叶子节点数目 None 不限制
min_impurity_decrease: Any = 0.,分裂导致不纯度减少大于等于该值则分裂
min_impurity_split: Any = None,分裂提前停止阈值，一个节点不纯度大于此阈值才能分裂
bootstrap: Any = True, 每个子模型的训练数据时是否做有放回的重采样 默认重采样
oob_score: Any = False,是否计算袋外准确率（没有参与模型训练的数据是否参与计算准确率）
warm_start: Any = False, 是否预热（重用之前的模型进行训练） 默认否
n_jobs: Any = None, 线程数
random_state: Any = None, 随机数种子
verbose: Any = 0  是否打印训练过程 0不打印 1打印
class_weight: Any = None,样本权重设置
ccp_alpha: Any = 0.0 最小剪枝系数 α大于该值则不进行修剪
max_samples: Any = None 每个子模型训练时使用多少原始数据
"""
algo = RandomForestClassifier(n_estimators=10,oob_score=True)

algo.fit(X_train, Y_train)
train_predict = algo.predict(X_train)
test_predict = algo.predict(X_test)

print(f'模型在训练集上的效果r:{algo.score(X_train, Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test, Y_test)}')
print(f'分类评估报告（训练集）:{classification_report(Y_train,train_predict)}')
print(f'分类评估报告（测试集）:{classification_report(Y_test,test_predict)}')
print(f'特征重要性权重：{algo.feature_importances_}')
print(f'模型的袋外准确率：{algo.oob_score_}')

