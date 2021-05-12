import math


# 欧姆极化损失
def ohmicLoss(Current, Ri):
    vLoss = Current * Ri
    return vLoss


# 浓差极化损失
def concentrationLoss(Current, iL, Temperature):
    F = 96485
    R = 8.314  # J/mol*K
    n = 2
    vLoss = (R * Temperature / (n * F)) * np.log(iL / (iL - Current))
    return vLoss


# 活化极化修正
def activationLoss(Current, i0, Temperature, a):
    i_internal = 0.003
    F = 96485
    R = 8.314
    vLoss = (R * Temperature / (a * F)) * np.log((Current + i_internal) / i0)
    return vLoss


# 最大可逆电压
def max_reversible_voltage(T, p_h2, p_o2):
    p0 = 101.325  # kPa
    p_h2 /= p0
    p_o2 /= p0
    T_ref = 298.15  # K
    n = 2  # mol
    F = 96485
    R = 8.314  # J/mol*K
    S_divided_by_nF = 0.000345882779707
    Er = 1.229 - S_divided_by_nF * (T - T_ref) + ((R * T) / (n * F)) * (math.log(p_h2) + 0.5 * math.log(p_o2))
    return Er


# 极化曲线
def operatingVoltage(i, T, p_h2, p_o2, Ri, i0, a, iL):
    Er = max_reversible_voltage(T, p_h2, p_o2)
    V_ohmic = ohmicLoss(i, Ri)
    V_act = activationLoss(i, i0, T, a)
    V_conc = concentrationLoss(i, iL, T)
    E = Er - V_act - V_ohmic - V_conc
    print(f'电流密度为{i}下 最大可逆电压为{Er}  欧姆极化损失为{V_ohmic} 活化极化损失为{V_act} 浓差极化损失为{V_conc} 输出电压{E}')
    return E


import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams["font.family"] = 'Arial Unicode MS'
# # 给图像增加图例
plt.legend()
# 轴名称
plt.xlabel('电流密度：A/cm^2')
plt.ylabel('电压：V')
# 0到4 生成等差数列 其中有100个点
i = np.linspace(0, 1.4, 100)
T = 353
p_h2 = 202
p_o2 = 202
Ri = 0.2
i0 = 0.00008
a = 0.5
iL = 1.4
# 根据20个点的x和y 带入公式画线
plt.plot(i, operatingVoltage(i, T, p_h2, p_o2, Ri, i0, a, iL))

# 开路电压
x1 = 0
y1 = operatingVoltage(x1, T, p_h2, p_o2, Ri, i0, a, iL)

# 开路电压
x1 = 0
y1 = operatingVoltage(x1, T, p_h2, p_o2, Ri, i0, a, iL)

plt.annotate(f"开路电压{round(y1, 2)}V", xy=(x1, y1))

# 截止电压
x2 = 1.39
y2 = operatingVoltage(x2, T, p_h2, p_o2, Ri, i0, a, iL)

plt.annotate(f"截止电压{round(y2, 2)}V", xy=(x2, y2))

# 0.2
x3 = 0.2
y3 = operatingVoltage(x3, T, p_h2, p_o2, Ri, i0, a, iL)

plt.annotate(f"{round(y3, 2)}V", xy=(x3, y3))

# 0.6
x4 = 0.6
y4 = operatingVoltage(x4, T, p_h2, p_o2, Ri, i0, a, iL)

plt.annotate(f"{round(y4, 2)}V", xy=(x4, y4))

# 1.0
x5 = 1.0
y5 = operatingVoltage(x5, T, p_h2, p_o2, Ri, i0, a, iL)

plt.annotate(f"{round(y5, 2)}V", xy=(x5, y5))

plt.scatter([x1,x2,x3,x4,x5], [y1,y2,y3,y4,y5], label='正例')


# 显示图片
plt.show()
