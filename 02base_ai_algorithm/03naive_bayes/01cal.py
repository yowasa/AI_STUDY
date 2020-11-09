# 朴素贝叶斯
import numpy as np

org_data = [
    [1, 's', -1],
    [1, 'm', -1],
    [1, 'm', 1],
    [1, 's', 1],
    [1, 's', -1],
    [2, 's', -1],
    [2, 'm', -1],
    [2, 'm', 1],
    [2, 'l', 1],
    [2, 'l', 1],
    [3, 'l', 1],
    [3, 'm', 1],
    [3, 'm', 1],
    [3, 'l', 1],
    [3, 'l', -1],
]
# 求出联合概率矩阵
data = np.array(org_data)
print(data[:, :-1].shape[1])

demo = []

for i in range(data.shape[1] - 1):
    r = []
    for j in range(data.shape[0]):
        r.append(org_data[j][i])
    r = list(set(r))
    demo.append(r)

# p(y)存储
per_y_data = []
total = len(org_data)
y = list(set(data[:, -1]))
for i in range(len(y)):
    top = len(list(filter(lambda n: n[-1] == int(y[i]), org_data)))
    button = total
    # 拉普拉斯平滑
    per_y_data.append((top + 1) / (button + len(y)))

result = {}
for i in range(len(y)):
    y_i = list(filter(lambda n: n[-1] == int(y[i]), org_data))
    top = len(y_i)
    for j in range(len(demo)):
        for k in range(len(demo[j])):
            count_x_i = len(list(filter(lambda n: n[j] == demo[j][k], y_i)))
            # 拉普拉斯平滑
            pre = (count_x_i + 1) / (top + len(demo[j]))
            key = str([j, demo[j][k], y[i]])
            result[key] = pre


def predict(px):
    p_y = []
    for i in range(len(y)):
        p_multi = 1
        for j in range(len(px)):
            pre = result.get(str([j, px[j], y[i]]))
            p_multi *= pre
        p_multi *= per_y_data[i]
        p_y.append([int(y[i]), p_multi])
    cal = sum(np.array(p_y)[:, -1])
    for i in p_y:
        i[-1] = i[-1] / cal
    return p_y


# 预测测试
p_x = [1, 's']

test = predict(p_x)
print(test)
