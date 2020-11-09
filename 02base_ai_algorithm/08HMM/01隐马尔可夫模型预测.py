from hmmlearn import hmm
import numpy as np

states = ['盒子1', '盒子2', '盒子3']

obs = ['白球', '黑球']

# 可观测状态数
n = 2
# 隐状态数
m = 3

# 定义训练数据
# 第一个序列为 白黑白白黑 01001
# 第二个序列为 01101
# 第三个序列为 011
# 第四个序列为 010000
train = np.array([
    [0], [1], [0], [0], [1],
    [0], [1], [1], [0], [1],
    [0], [1], [1],
    [0], [1], [0], [0], [0], [0]
])

# 定义模型
# 多项式隐马尔可夫模型
"""
MultinomialHMM 观测值是离散值的hmm模型
n_components=1,隐状态数目
startprob_prior=1.0, 初始概率的模型参数 一般不变
transmat_prior=1.0,初始转移矩阵概率参数 一般不变
algorithm="viterbi", 预测方式 还有map（每个时刻算出最大概率为最终状态）
random_state=None, 随机数种子
n_iter=10, 迭代次数
tol=1e-2, 收敛值
verbose=False, 打印过程
params="ste", 
init_params="ste"
"""
model = hmm.MultinomialHMM(n_components=m, n_iter=10, tol=0.01, random_state=22, verbose=True)
# lengths给定各个子序列长度
model.fit(train, lengths=[5, 5, 3, 6])

print(f'训练得到的初始概率：{model.startprob_}')
print(f'训练得到的转移矩阵：{model.transmat_}')
print(f'训练得到的发射矩阵：{model.emissionprob_}')

# 做一个viterbi算法预测
# 观测序列 白黑白白黑
test = np.array([[0, 1, 0, 0, 1]]).reshape(5, 1)
print(f'观测序列为：{test}')
print(f'预测序列（盒子编号）为：{model.predict(test)}')
print(f'概率值为：{model.predict_proba(test)}')
logprob, boxindex = model.decode(test, algorithm='viterbi')
print(f'预测序列（盒子编号）为：{boxindex}')
print(f'概率值为(hmm对概率做了log转化)：{np.exp(logprob)}')
