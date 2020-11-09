import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

flag = 3
# 回归模型
if flag == 1:
    # 造数据
    np.random.seed(22)
    x = np.random.randn(10, 2) * 5
    y = np.random.randn(10, 1) * 3
    # 学习率
    lr = 0.1
    # 均值
    models = []
    y_true = y
    pred_m = np.mean(y)
    n = 1000
    for i in range(1000):
        if i == 0:
            y = y - lr * pred_m
            models.append(pred_m)
        else:
            y = y - lr * pred_m.predict(x).reshape(y.shape)
        model = DecisionTreeRegressor(max_depth=1)
        model.fit(x, y)
        models.append(model)
        pred_m = model
    # 构建完成

    # 开始预测
    # 构建一个和y一直的0矩阵
    y_pre = np.zeros_like(y)
    for i in range(1001):
        model = models[i]
        if i == 0:
            y_pre += lr * model
        else:
            y_pre += lr * model.predict(x).reshape(y.shape)
    # 输出模型效果
    print(f'构建模型完成 总模型数：{len(models)}')
    print(f'GBDT效果：{r2_score(y_true, y_pre)}')
    print(f'实际值：{y_true}')
    print(f'预测值：{y_pre}')
# 二分类情况
elif flag == 2:
    # 造数据
    np.random.seed(22)
    x = np.random.randn(10, 2) * 5
    y = np.array([1] * 6 + [0] * 4).astype(np.int)
    # 学习率
    lr = 0.1
    # 均值
    models = []
    y_true = y
    pred_m = np.log(6.0 / 4.0)
    n = 1000
    for i in range(1000):
        if i == 0:
            y = y - lr * pred_m
            models.append(pred_m)
        else:
            y = y - lr * pred_m.predict(x).reshape(y.shape)
        model = DecisionTreeRegressor(max_depth=1)
        model.fit(x, y)
        models.append(model)
        pred_m = model
    # 构建完成

    # 开始预测
    # 构建一个和y一直的0矩阵
    y_pre = np.zeros_like(y)
    for i in range(1001):
        model = models[i]
        if i == 0:
            y_pre += lr * model
        else:
            y_pre += lr * model.predict(x).reshape(y.shape)
    y_hat = np.zeros_like(y_pre, np.int)
    y_hat[y_pre >= 0.5] = 1
    y_hat[y_pre < 0.5] = 0
    # 输出模型效果
    print(f'构建模型完成 总模型数：{len(models)}')
    print(f'GBDT效果：{r2_score(y_true, y_pre)}')
    print(f'实际值：{y_true}')
    print(f'预测值：{y_hat}')
    print(f'决策函数值：{y_pre}')
# 多分类情况
elif flag == 3:
    # 造数据
    np.random.seed(22)
    x = np.random.randn(10, 2) * 5
    y = np.array([0] * 2 + [1] * 4 + [2] * 4).astype(np.int)
    y_true = y
    # 转化为二分类情况
    y0 = np.array([1] * 2 + [0] * 8).astype(np.int)
    y1 = np.array([0] * 2 + [1] * 4 + [0] * 4).astype(np.int)
    y2 = np.array([0] * 6 + [1] * 4).astype(np.int)
    ys = [y0, y1, y2]
    # 学习率
    lr = 0.1
    # 均值
    models = []
    pred_m = np.asarray([0, 0, 0])
    n = 1000
    for i in range(1000):
        # 保存地i此迭代时产生的所有子模型
        tmp_algos = []
        if i == 0:
            pred_y = pred_m
            models.append(pred_m)
        else:
            pred_y = np.asarray(list(map(lambda algo: algo.predict(x), pred_m)))
        for k in range(len(ys)):
            p = np.exp(pred_y[k]) / np.sum(np.exp(pred_y))
            ys[k] = ys[k] - p
            model = DecisionTreeRegressor(max_depth=1)
            model.fit(x, ys[k])
            tmp_algos.append(model)
            pred_m = tmp_algos
        models.append(tmp_algos)
    # 构建完成

    # 开始预测
    # 构建一个和y一直的0矩阵
    y_pre = np.zeros((3,y.shape[0]))
    for i in range(1001):
        model = models[i]
        for k in range(len(ys)):
            if i ==0:
                y_pre[k]+=model[k]
            else:
                y_pre[k]+=model[k].predict(x).reshape(y.shape)
    y_hat=np.argmax(y_pre,axis=0).astype(np.int)
    # 输出模型效果
    print(f'构建模型完成 总模型数：{len(models)*len(ys)}')
    print(f'GBDT效果：{r2_score(y_true, y_hat)}')
    print(f'实际值：{y_true}')
    print(f'预测值：{y_hat}')
    print(f'决策函数值：{y_pre}')
