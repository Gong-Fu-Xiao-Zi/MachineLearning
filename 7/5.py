import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_csv('5_心血管疾病预测.csv')
print("数据形状:", data.shape)
print("\n数据基本信息:")
print(data.info())
print("\n前5行数据:")
print(data.head())

# 数据基本信息
print(f"\n数据集包含 {len(data)} 条记录")
print(f"心血管疾病比例: {data['cardio'].mean():.2%}")

# 创建可视化
color_map = {0: 'lightblue', 1: 'red'}
colors = list(map(lambda x: color_map[x], data['cardio']))

# 创建4x3的子图布局
fig, axs = plt.subplots(4, 3, figsize=(18, 12))

# 1. 年龄分布 (注意: 年龄数据需要转换，看起来是以天为单位)
data['age_years'] = data['age'] / 365.25  # 转换为年
axs[0,0].scatter(np.arange(len(data)), data['age_years'], c=colors, s=8, alpha=0.6)
axs[0,0].set_title('年龄分布 (年)')
axs[0,0].set_ylabel('年龄 (年)')

# 2. 性别分布
axs[0,1].scatter(np.arange(len(data)), data['gender'], c=colors, s=8, alpha=0.6)
axs[0,1].set_title('性别分布')
axs[0,1].set_yticks([1, 2])
axs[0,1].set_yticklabels(['男', '女'])

# 3. 身高分布
axs[0,2].scatter(np.arange(len(data)), data['height'], c=colors, s=8, alpha=0.6)
axs[0,2].set_title('身高分布')
axs[0,2].set_ylabel('身高 (cm)')

# 4. 体重分布
axs[1,0].scatter(np.arange(len(data)), data['weight'], c=colors, s=8, alpha=0.6)
axs[1,0].set_title('体重分布')
axs[1,0].set_ylabel('体重 (kg)')

# 5. 收缩压分布
axs[1,1].scatter(np.arange(len(data)), data['ap_hi'], c=colors, s=8, alpha=0.6)
axs[1,1].set_title('收缩压分布')
axs[1,1].set_ylabel('收缩压')

# 6. 舒张压分布
axs[1,2].scatter(np.arange(len(data)), data['ap_lo'], c=colors, s=8, alpha=0.6)
axs[1,2].set_title('舒张压分布')
axs[1,2].set_ylabel('舒张压')

# 7. 胆固醇水平
axs[2,0].scatter(np.arange(len(data)), data['cholesterol'], c=colors, s=8, alpha=0.6)
axs[2,0].set_title('胆固醇水平')
axs[2,0].set_yticks([1, 2, 3])
axs[2,0].set_yticklabels(['正常', '偏高', '很高'])

# 8. 血糖水平
axs[2,1].scatter(np.arange(len(data)), data['gluc'], c=colors, s=8, alpha=0.6)
axs[2,1].set_title('血糖水平')
axs[2,1].set_yticks([1, 2, 3])
axs[2,1].set_yticklabels(['正常', '偏高', '很高'])

# 9. 吸烟情况
axs[2,2].scatter(np.arange(len(data)), data['smoke'], c=colors, s=8, alpha=0.6)
axs[2,2].set_title('吸烟情况')
axs[2,2].set_yticks([0, 1])
axs[2,2].set_yticklabels(['不吸烟', '吸烟'])

# 10. 饮酒情况
axs[3,0].scatter(np.arange(len(data)), data['alco'], c=colors, s=8, alpha=0.6)
axs[3,0].set_title('饮酒情况')
axs[3,0].set_yticks([0, 1])
axs[3,0].set_yticklabels(['不饮酒', '饮酒'])

# 11. 运动情况
axs[3,1].scatter(np.arange(len(data)), data['active'], c=colors, s=8, alpha=0.6)
axs[3,1].set_title('运动情况')
axs[3,1].set_yticks([0, 1])
axs[3,1].set_yticklabels(['不运动', '运动'])

# 12. 目标变量分布
cardio_counts = data['cardio'].value_counts()
axs[3,2].bar(['无疾病', '有疾病'], cardio_counts.values,
            color=['lightblue', 'red'], alpha=0.7)
axs[3,2].set_title('心血管疾病分布')
for i, v in enumerate(cardio_counts.values):
    axs[3,2].text(i, v, str(v), ha='center', va='bottom')

plt.tight_layout()
plt.show()

# 创建相关性热力图
plt.figure(figsize=(12, 8))
correlation_matrix = data.drop('id', axis=1).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
           square=True, fmt='.2f', linewidths=0.5)
plt.title('特征相关性热力图')
plt.tight_layout()
plt.show()

# 创建BMI特征并分析
data['bmi'] = data['weight'] / ((data['height'] / 100) ** 2)

# 创建年龄分组
data['age_group'] = pd.cut(data['age_years'],
                         bins=[0, 40, 50, 60, 70, 100],
                         labels=['<40', '40-50', '50-60', '60-70', '70+'])

# 分析不同年龄组的心血管疾病发病率
plt.figure(figsize=(10, 6))
age_cardio_rate = data.groupby('age_group')['cardio'].mean() * 100
age_cardio_rate.plot(kind='bar', color='skyblue', alpha=0.7)
plt.title('不同年龄组心血管疾病发病率')
plt.ylabel('发病率 (%)')
plt.xlabel('年龄组')
for i, v in enumerate(age_cardio_rate):
    plt.text(i, v + 0.5, f'{v:.1f}%', ha='center')
plt.tight_layout()
plt.show()

# BMI与心血管疾病的关系
plt.figure(figsize=(10, 6))
bmi_cardio = data.groupby(pd.cut(data['bmi'], bins=10))['cardio'].mean() * 100
bmi_cardio.plot(kind='bar', color='lightgreen', alpha=0.7)
plt.title('不同BMI区间心血管疾病发病率')
plt.ylabel('发病率 (%)')
plt.xlabel('BMI区间')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 血压与心血管疾病的关系
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 收缩压
systolic_bins = [0, 120, 130, 140, 160, 200, 500]
systolic_labels = ['<120', '120-129', '130-139', '140-159', '160-199', '200+']
data['systolic_group'] = pd.cut(data['ap_hi'], bins=systolic_bins, labels=systolic_labels)
systolic_rate = data.groupby('systolic_group')['cardio'].mean() * 100
systolic_rate.plot(kind='bar', ax=ax1, color='coral', alpha=0.7)
ax1.set_title('不同收缩压水平心血管疾病发病率')
ax1.set_ylabel('发病率 (%)')

# 舒张压
diastolic_bins = [0, 80, 85, 90, 100, 200, 1500]
diastolic_labels = ['<80', '80-84', '85-89', '90-99', '100-199', '200+']
data['diastolic_group'] = pd.cut(data['ap_lo'], bins=diastolic_bins, labels=diastolic_labels)
diastolic_rate = data.groupby('diastolic_group')['cardio'].mean() * 100
diastolic_rate.plot(kind='bar', ax=ax2, color='lightcoral', alpha=0.7)
ax2.set_title('不同舒张压水平心血管疾病发病率')
ax2.set_ylabel('发病率 (%)')

plt.tight_layout()
plt.show()

# 打印一些统计信息
print("\n=== 数据集统计信息 ===")
print(f"总样本数: {len(data)}")
print(f"心血管疾病患者: {data['cardio'].sum()} ({data['cardio'].mean():.2%})")
print(f"平均年龄: {data['age_years'].mean():.1f} 岁")
print(f"平均BMI: {data['bmi'].mean():.1f}")
print(f"吸烟者比例: {data['smoke'].mean():.2%}")
print(f"饮酒者比例: {data['alco'].mean():.2%}")
print(f"运动者比例: {data['active'].mean():.2%}")

# 按性别统计
print("\n=== 按性别统计 ===")
gender_stats = data.groupby('gender').agg({
    'cardio': ['count', 'mean'],
    'age_years': 'mean',
    'bmi': 'mean'
}).round(3)
print(gender_stats)