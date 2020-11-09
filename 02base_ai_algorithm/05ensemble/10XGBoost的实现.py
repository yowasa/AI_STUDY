import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
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
max_depth=None, 树深
learning_rate=None, 学习率
n_estimators=100, 模型数
verbosity=None, 
objective=None, 损失函数
booster=None, 求解方式
tree_method=None, 
n_jobs=None, 执行线程数目
gamma=None, gama值的大小 越大受叶子节点数目的制约越大
min_child_weight=None, 
max_delta_step=None, 
subsample=None,
colsample_bytree=None, 
colsample_bylevel=None,
colsample_bynode=None, 
reg_alpha=None, 
reg_lambda=None,
scale_pos_weight=None, 
base_score=None, 
random_state=None,
missing=np.nan, 
num_parallel_tree=None,
monotone_constraints=None, 
interaction_constraints=None,
importance_type="gain", 
gpu_id=None, 使用哪个gpu进行运算
validate_parameters=None
"""
algo = XGBClassifier(max_depth=3, n_estimators=10)


algo.fit(X_train, Y_train)
train_predict = algo.predict(X_train)
test_predict = algo.predict(X_test)

print(f'模型在训练集上的效果r:{algo.score(X_train, Y_train)}')
print(f'模型在测试集上的效果r:{algo.score(X_test, Y_test)}')
print(f'分类评估报告（训练集）:{classification_report(Y_train, train_predict)}')
print(f'分类评估报告（测试集）:{classification_report(Y_test, test_predict)}')
print(f'特征重要性权重：{algo.feature_importances_}')
