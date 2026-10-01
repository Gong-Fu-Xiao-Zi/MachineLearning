'''import random as rd
from math import sqrt
import matplotlib.pyplot as plt


num = 500  # 投掷点数为1000的整数倍
for t in range(1, 4+1):  # 分别观察1000、2000、3000、4000个投掷点的结果
    plt.subplot(2, 2, t)
    area = 0    # 落在1/4单位圆内的点数
    for i in range(0, num*t):
        x = rd.random()      # 随机生成投掷点的横坐标
        y = rd.random()
        r = sqrt(x**2 + y**2)   # 计算当前点到坐标原点距离
        if r <= 1:              # 判断当前点是否在单位圆内
            area += 1           # 在单位圆内，计数+1
            plt.scatter(x, y, color='blue', marker='.')   # 画散点图，单位圆内画蓝色
        else:
            plt.scatter(x, y, color='red', marker='.')    # 单位圆外画红色
    plt.xlabel('x')      # 坐标横轴标签
    plt.ylabel('y')
    plt.title("{} points Monte Carlo".format(t*num))    # 图名
    pi = area*4/(t*num)  # 根据蒙特卡罗法原理计算圆周率
    print('pi = %f' % pi)
plt.tight_layout()
plt.show()'''

import random
import matplotlib.pyplot as plt


def monte_carlo_ellipse_area(num_points):
    a = 3
    b = 2
    area = 0

    for _ in range(num_points):
        x = random.uniform(-a, a)
        y = random.uniform(-b, b)

        if (x / a) ** 2 + (y / b) ** 2 <= 1:
            area += 1
            plt.scatter(x, y, color='blue', marker='.')
        else:
            plt.scatter(x, y, color='red', marker='.')

    estimated_area = area * (4 * a * b) / num_points
    return estimated_area


num_points = 500  # 投掷点数
for t in range(1, 5):  # 分别观察500、1000、1500、2000个投掷点的结果
    plt.subplot(2, 2, t)
    estimated_area = monte_carlo_ellipse_area(num_points * t)
    plt.title(f"{num_points * t} points Monte Carlo - Area: {estimated_area:.2f}")
    plt.xlabel('x')
    plt.ylabel('y')

plt.tight_layout()
plt.show()