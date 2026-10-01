'''import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filname = "年度数据.csv"
data = pd.read_csv(filname)
data_my = data.iloc[200:220]
#print(data.iloc[0:5])


# 准备数据
#data1,data2,data3,data4=np.random.randn(4,150);
data1 = data_my['TotalUSDPrize']
data2 = data_my['PlayerId'] * 20
print(data1)


# 创建画布
fig,ax=plt.subplots(figsize=(10,5.4),layout='constrained')

# 绘制图像
#x=np.arange(1000)
x = data_my['CurrentHandle']
ax.plot(x,data1,color='blue',linewidth=1.5,linestyle=':',label='TotalUSDPrize');
ax.plot(x,data2,'d',label='Games')
#ax.plot(data3,'s',label='data3')
ax.legend();

# 设置横纵轴以及标题
ax.set_xlabel('Player ID')
ax.set_ylabel('label y')
ax.set_title('highest_earning_players')

# plt.savefig('D:\Python pro\exercise\pythonProject1/0.Mat---Picture/figure1');

# 图像展示
plt.show()

fig, ax = plt.subplots(figsize=(5, 2.7))
ax.scatter(data1, data2,s=10,  facecolor='white', edgecolor='blue');
plt.show()

x=150
fig, ax = plt.subplots(figsize=(5, 2.7))
ax.plot(data1, 'o', label='data1')
ax.plot(data2, 'd', label='data2')
ax.plot(data3, 'v', label='data3')
ax.plot(data4, 's', label='data4')
ax.legend();

# plt.show()
'''

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据
data = {
    '指标': ['国民总收入(亿元)', '国内生产总值(亿元)', '第一产业增加值(亿元)', '第二产业增加值(亿元)', '第三产业增加值(亿元)', '人均国内生产总值(元)'],
    '2024年': [1339814.6, 1349083.5, 91413.9, 492087.1, 765582.5, 95749],
    '2023年': [1284773.9, 1294271.7, 89169.1, 475936.1, 729166.5, 91746],
    '2022年': [1223706.8, 1234029.4, 88207.0, 467629.6, 678192.7, 87385],
    '2021年': [1165816.8, 1173823.0, 83216.5, 447138.2, 643468.4, 83111],
    '2020年': [1026751.9, 1034867.6, 78030.9, 381985.8, 574850.9, 73338],
    '2019年': [1003108.4, 1005872.4, 70473.6, 379860.0, 555538.9, 71453],
    '2018年': [931972.5, 936010.1, 64745.2, 364835.2, 506429.8, 66726],
    '2017年': [846292.7, 847382.9, 62099.5, 331580.5, 453702.9, 60691],
    '2016年': [757492.0, 761193.0, 60139.2, 295427.8, 405626.0, 54849],
    '2015年': [699224.5, 702511.5, 57774.6, 281338.9, 363397.9, 50912],
    '2014年': [656600.0, 655782.9, 55626.3, 277282.8, 322873.8, 47802],
    '2013年': [598838.4, 603660.4, 53028.1, 261951.6, 288680.7, 44281],
    '2012年': [546259.7, 547510.6, 49084.6, 244639.1, 253786.9, 40431],
    '2011年': [491160.2, 495707.6, 44781.5, 227035.1, 223891.0, 36855],
    '2010年': [417488.1, 419253.3, 38430.8, 191626.5, 189195.9, 31341],
    '2009年': [353938.8, 354521.6, 33583.8, 160168.8, 160769.0, 26631],
    '2008年': [326302.7, 324317.8, 32464.1, 149952.9, 141900.7, 24483],
    '2007年': [274791.4, 274179.7, 27674.1, 126630.5, 119875.0, 20805],
    '2006年': [222168.4, 222578.4, 23317.0, 104359.2, 94902.2, 16977],
    '2005年': [188587.6, 189907.5, 21806.7, 88082.2, 80018.6, 14567]
}

df = pd.DataFrame(data)
df.set_index('指标', inplace=True)

df_t = df.T

df_t = df_t.iloc[::-1]

years = np.arange(2005, 2025)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))

# 第一个子图：人均国内生产总值和国内生产总值的折线图（使用双Y轴）
# 绘制国内生产总值（左轴）
color = 'tab:blue'
ax1.plot(years, df_t['国内生产总值(亿元)'], marker='o', color=color, label='国内生产总值(亿元)', linewidth=2)
ax1.set_xlabel('年份', fontsize=12)
ax1.set_ylabel('国内生产总值(亿元)', color=color, fontsize=12)
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.7)

# 创建第二个Y轴，用于人均国内生产总值
ax1_right = ax1.twinx()
color = 'tab:red'
ax1_right.plot(years, df_t['人均国内生产总值(元)'], marker='s', color=color, label='人均国内生产总值(元)', linewidth=2)
ax1_right.set_ylabel('人均国内生产总值(元)', color=color, fontsize=12)
ax1_right.tick_params(axis='y', labelcolor=color)

# 添加图例
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax1_right.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=12)

ax1.set_title('近二十年国内生产总值与人均国内生产总值趋势', fontsize=16)

# 第二个子图：三大产业增加值的散点图
ax2.scatter(years, df_t['第一产业增加值(亿元)'], label='第一产业增加值(亿元)', s=100, alpha=0.7)
ax2.scatter(years, df_t['第二产业增加值(亿元)'], label='第二产业增加值(亿元)', s=100, alpha=0.7)
ax2.scatter(years, df_t['第三产业增加值(亿元)'], label='第三产业增加值(亿元)', s=100, alpha=0.7)
ax2.set_title('近二十年三大产业增加值分布', fontsize=16)
ax2.set_xlabel('年份', fontsize=12)
ax2.set_ylabel('亿元', fontsize=12)
ax2.legend(fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()
