# 简单的最近邻
data = [[1, 4, 1], [0.5, 2, 1], [2, 2.3, 1], [1, 0.5, -1], [2, 1, -1],
        [4, 1, -1], [3.5, 4, 1], [3, 2.2, -1]]


def cal_distance(x, y):
    sum = 0
    for i in range(len(x)):
        sum += (x[i] - y[i]) ** 2
    return sum


def judje(x):
    flag = 0
    nearest = -1
    for i in data:
        dis = cal_distance(x, i[0:-1])
        if nearest == -1 or dis < nearest:
            nearest = dis
            flag = i[-1]
    return flag

print(judje([4,2]))
