'''
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 数据生成
mu, sigma = 115, 15
x = mu + sigma * np.random.randn(10000)

# 创建画布
fig, ax = plt.subplots(figsize=(5, 2.7), layout='constrained')
# the histogram of the data,hist——直方图
n, bins, patches = ax.hist(x, 150, density=1, facecolor='green', alpha=0.75)

# 设置横纵轴及标题
ax.set_xlabel('Length [cm]')
ax.set_ylabel('Probability')
ax.set_title('Aardvark lengths\n (not really)')
# 添加某点的文本说明
ax.text(75, .025, r'$\mu=115,\ \sigma=15$')
ax.axis([55, 175, 0, 0.03])
ax.grid(True);
plt.grid(linestyle='-.',color='red') ;
# plt.savefig('D:\Python pro\exercise\pythonProject1/0.Mat---Picture/figure2.png')
plt.show()
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据
years = ['2022年', '2023年', '2024年']
age_groups = ['0-14岁', '15-64岁', '65岁及以上']

population_data = {
    '2022年': [23908, 96289, 20978],
    '2023年': [23063, 96228, 21676],
    '2024年': [22240, 96565, 22023]
}

percentage_data = {}
for year, data in population_data.items():
    total = sum(data)
    percentage_data[year] = [round((x / total) * 100, 1) for x in data]

fig, axes = plt.subplots(1, 3, figsize=(15, 6), layout='constrained')

colors = ['#FF9999', '#66B2FF', '#99FF99']

for i, year in enumerate(years):
    ax = axes[i]
    data = percentage_data[year]

    # 绘制直方图
    bars = ax.bar(age_groups, data, color=colors, alpha=0.7, edgecolor='black')

    # 设置标题和标签
    ax.set_title(f'{year}人口年龄结构(%)', fontsize=14)
    ax.set_xlabel('年龄组', fontsize=12)
    ax.set_ylabel('人口占比(%)', fontsize=12)

    # 在每个柱子上添加百分比标签
    for j, bar in enumerate(bars):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 1,
                f'{data[j]}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_ylim(0, max(data) * 1.2)


    ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.suptitle('近三年人口年龄结构百分比分布', fontsize=16, fontweight='bold')

plt.tight_layout()

plt.show()

fig2, ax2 = plt.subplots(figsize=(10, 6))

x = np.arange(len(age_groups))
width = 0.25

for i, year in enumerate(years):
    data = percentage_data[year]
    ax2.bar(x + i * width, data, width, label=year, color=colors[i], alpha=0.7, edgecolor='black')

ax2.set_title('近三年人口年龄结构百分比对比', fontsize=16)
ax2.set_xlabel('年龄组', fontsize=12)
ax2.set_ylabel('人口占比(%)', fontsize=12)
ax2.set_xticks(x + width)
ax2.set_xticklabels(age_groups)
ax2.legend(fontsize=12)

ax2.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()

plt.show()