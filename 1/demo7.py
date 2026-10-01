#横坐标为日期，可以为年月日，时分秒
#将横坐标使用datetime64函数

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
dates = np.arange(np.datetime64('2021-11'), np.datetime64('2022-11'),
                  np.timedelta64(1,'D'));   '''日期的格式嵌套进arange函数，日期函数中timedelta64第一个参数是间隔，第二个是单位，年月日第一个字母大写表示，时分秒首字母小写'''
data = np.cumsum(np.random.randn(len(dates)))
print(len(dates))
print('---------------------------------------------------------------')
print(np.random.randn(len(dates)))
print('---------------------------------------------------------------')
print(data)
ax.plot(dates, data,color='red')
cdf = mpl.dates.ConciseDateFormatter(ax.xaxis.get_major_locator())
ax.xaxis.set_major_formatter(cdf);

plt.show()