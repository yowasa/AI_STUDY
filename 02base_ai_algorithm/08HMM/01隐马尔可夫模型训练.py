from hmmlearn import hmm
import numpy as np

states = ['盒子1', '盒子2', '盒子3']

obs = ['白球', '黑球']

# 可观测状态数
n = 2
# 隐状态数
m = 3

# 初始概率
pi = np.array([0.2, 0.5, 0.3])

# 状态转移矩阵
a = np.array([
    [0.5, 0.4, 0.1],
    [0.2, 0.2, 0.6],
    [0.2, 0.5, 0.3]
])

# 混淆矩阵
b = np.array([
    [0.4, 0.6],
    [0.8, 0.2],
    [0.5, 0.5]
])

# 定义模型
# 多项式隐马尔可夫模型
"""
MultinomialHMM 观测值是离散值的hmm模型
n_components：隐状态数目
"""
model = hmm.MultinomialHMM(n_components=m)

# 直接给定模型参数
model.startprob_ = pi
model.transmat_ = a
model.emissionprob_ = b

# 做一个viterbi算法预测
# 观测序列 白黑白白黑
test = np.array([[0, 1, 0, 0, 1]]).reshape(5, 1)
print(f'观测序列为：{test}')
print(f'预测序列（盒子编号）为：{model.predict(test)}')
print(f'概率值为：{model.predict_proba(test)}')
logprob, boxindex = model.decode(test, algorithm='viterbi')
print(f'预测序列（盒子编号）为：{boxindex}')
print(f'概率值为(hmm对概率做了log转化)：{np.exp(logprob)}')