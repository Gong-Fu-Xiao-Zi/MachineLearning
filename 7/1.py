import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载和预处理
print("=== 心血管疾病预测分析 ===")

# 加载数据
df = pd.read_csv('5_心血管疾病预测.csv')

print(f"数据集形状: {df.shape}")
print("\n数据前5行:")
print(df.head())

print("\n数据基本信息:")
print(df.info())

print("\n缺失值统计:")
print(df.isnull().sum())

print("\n目标变量分布:")
print(df['cardio'].value_counts())

# 数据预处理
# 将年龄从天数转换为年数
df['age'] = df['age'] / 365.25

# 计算BMI
df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

# 处理血压异常值
df = df[(df['ap_hi'] >= 50) & (df['ap_hi'] <= 250)]
df = df[(df['ap_lo'] >= 30) & (df['ap_lo'] <= 150)]

print(f"清洗后数据形状: {df.shape}")

# 2. 数据可视化
print("\n=== 数据可视化 ===")

# 目标变量分布
plt.figure(figsize=(15, 12))

plt.subplot(3, 3, 1)
df['cardio'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('目标变量分布 (cardio)')
plt.xlabel('是否有心血管疾病')
plt.ylabel('数量')
plt.xticks(rotation=0)

# 年龄分布
plt.subplot(3, 3, 2)
plt.hist(df['age'], bins=30, alpha=0.7, color='lightgreen')
plt.title('年龄分布')
plt.xlabel('年龄 (年)')
plt.ylabel('频数')

# 性别分布
plt.subplot(3, 3, 3)
df['gender'].value_counts().plot(kind='bar', color=['lightpink', 'lightblue'])
plt.title('性别分布')
plt.xlabel('性别')
plt.ylabel('数量')
plt.xticks(rotation=0)

# BMI分布
plt.subplot(3, 3, 4)
plt.hist(df['bmi'], bins=30, alpha=0.7, color='orange')
plt.title('BMI分布')
plt.xlabel('BMI')
plt.ylabel('频数')

# 收缩压分布
plt.subplot(3, 3, 5)
plt.hist(df['ap_hi'], bins=30, alpha=0.7, color='red')
plt.title('收缩压分布')
plt.xlabel('收缩压 (ap_hi)')
plt.ylabel('频数')

# 舒张压分布
plt.subplot(3, 3, 6)
plt.hist(df['ap_lo'], bins=30, alpha=0.7, color='purple')
plt.title('舒张压分布')
plt.xlabel('舒张压 (ap_lo)')
plt.ylabel('频数')

# 胆固醇水平
plt.subplot(3, 3, 7)
df['cholesterol'].value_counts().sort_index().plot(kind='bar', color='lightseagreen')
plt.title('胆固醇水平分布')
plt.xlabel('胆固醇水平')
plt.ylabel('数量')

# 血糖水平
plt.subplot(3, 3, 8)
df['gluc'].value_counts().sort_index().plot(kind='bar', color='gold')
plt.title('血糖水平分布')
plt.xlabel('血糖水平')
plt.ylabel('数量')

plt.tight_layout()
plt.show()

# 相关性热力图
plt.figure(figsize=(12, 10))
numeric_cols = ['age', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'cholesterol', 'gluc', 'cardio']
correlation_matrix = df[numeric_cols].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('特征相关性热力图')
plt.tight_layout()
plt.show()

# 3. 特征工程和数据划分
print("\n=== 特征工程和数据划分 ===")

# 选择特征
features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi']
X = df[features]
y = df['cardio']

print(f"特征形状: {X.shape}")
print(f"目标变量形状: {y.shape}")

# 划分数据集：训练集60%，验证集20%，测试集20%
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print(f"训练集大小: {X_train.shape}")
print(f"验证集大小: {X_val.shape}")
print(f"测试集大小: {X_test.shape}")

# 处理不平衡数据 - 使用SMOTE过采样
print("\n应用SMOTE过采样...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

print(f"过采样后训练集形状: {X_train_resampled.shape}")
print(f"过采样后目标变量分布: {pd.Series(y_train_resampled).value_counts()}")

# 4. 决策树和随机森林的深度比较
print("\n=== 深度比较和过拟合分析 ===")

# 测试不同的深度
depths = range(1, 21)
dt_train_scores = []
dt_val_scores = []
rf_train_scores = []
rf_val_scores = []

for depth in depths:
    # 决策树
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train_resampled, y_train_resampled)
    dt_train_score = dt.score(X_train_resampled, y_train_resampled)
    dt_val_score = dt.score(X_val, y_val)
    dt_train_scores.append(dt_train_score)
    dt_val_scores.append(dt_val_score)

    # 随机森林
    rf = RandomForestClassifier(max_depth=depth, n_estimators=100, random_state=42)
    rf.fit(X_train_resampled, y_train_resampled)
    rf_train_score = rf.score(X_train_resampled, y_train_resampled)
    rf_val_score = rf.score(X_val, y_val)
    rf_train_scores.append(rf_train_score)
    rf_val_scores.append(rf_val_score)

# 绘制深度比较图
plt.figure(figsize=(15, 6))

plt.subplot(1, 2, 1)
plt.plot(depths, dt_train_scores, 'o-', label='决策树训练集', color='blue')
plt.plot(depths, dt_val_scores, 's-', label='决策树验证集', color='red')
plt.xlabel('最大深度')
plt.ylabel('准确率')
plt.title('决策树不同深度的性能')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(depths, rf_train_scores, 'o-', label='随机森林训练集', color='green')
plt.plot(depths, rf_val_scores, 's-', label='随机森林验证集', color='orange')
plt.xlabel('最大深度')
plt.ylabel('准确率')
plt.title('随机森林不同深度的性能')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 找到最佳深度
dt_best_depth = depths[np.argmax(dt_val_scores)]
rf_best_depth = depths[np.argmax(rf_val_scores)]

print(f"决策树最佳深度: {dt_best_depth}, 验证集准确率: {max(dt_val_scores):.4f}")
print(f"随机森林最佳深度: {rf_best_depth}, 验证集准确率: {max(rf_val_scores):.4f}")

# 5. 参数调优
print("\n=== 参数调优 ===")

# 决策树参数调优
print("正在进行决策树参数调优...")
dt_param_grid = {
    'max_depth': [dt_best_depth - 1, dt_best_depth, dt_best_depth + 1, dt_best_depth + 2],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42),
                       dt_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
dt_grid.fit(X_train_resampled, y_train_resampled)

print(f"决策树最佳参数: {dt_grid.best_params_}")
print(f"决策树最佳交叉验证分数: {dt_grid.best_score_:.4f}")

# 随机森林参数调优
print("\n正在进行随机森林参数调优...")
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [rf_best_depth - 1, rf_best_depth, rf_best_depth + 1],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

rf_grid = GridSearchCV(RandomForestClassifier(random_state=42),
                       rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train_resampled, y_train_resampled)

print(f"随机森林最佳参数: {rf_grid.best_params_}")
print(f"随机森林最佳交叉验证分数: {rf_grid.best_score_:.4f}")

# 6. 使用最佳模型进行预测
print("\n=== 模型评估 ===")

# 使用最佳参数训练模型
best_dt = dt_grid.best_estimator_
best_rf = rf_grid.best_estimator_

# 在测试集上进行预测
y_pred_dt = best_dt.predict(X_test)
y_pred_rf = best_rf.predict(X_test)

# 计算准确率
dt_accuracy = accuracy_score(y_test, y_pred_dt)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"决策树测试集准确率: {dt_accuracy:.4f}")
print(f"随机森林测试集准确率: {rf_accuracy:.4f}")

# 7. 结果可视化
print("\n=== 结果可视化 ===")

# 混淆矩阵
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 决策树混淆矩阵
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'决策树混淆矩阵\n准确率: {dt_accuracy:.4f}')
axes[0].set_xlabel('预测标签')
axes[0].set_ylabel('真实标签')

# 随机森林混淆矩阵
cm_rf = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f'随机森林混淆矩阵\n准确率: {rf_accuracy:.4f}')
axes[1].set_xlabel('预测标签')
axes[1].set_ylabel('真实标签')

plt.tight_layout()
plt.show()

# 分类报告
print("\n决策树分类报告:")
print(classification_report(y_test, y_pred_dt))

print("\n随机森林分类报告:")
print(classification_report(y_test, y_pred_rf))

# 8. 特征重要性
print("\n=== 特征重要性分析 ===")

# 决策树特征重要性
dt_importance = best_dt.feature_importances_
dt_feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': dt_importance
}).sort_values('importance', ascending=False)

# 随机森林特征重要性
rf_importance = best_rf.feature_importances_
rf_feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': rf_importance
}).sort_values('importance', ascending=False)

# 绘制特征重要性
fig, axes = plt.subplots(1, 2, figsize=(15, 8))

# 决策树特征重要性
axes[0].barh(dt_feature_importance_df['feature'], dt_feature_importance_df['importance'], color='skyblue')
axes[0].set_title('决策树特征重要性')
axes[0].set_xlabel('重要性')

# 随机森林特征重要性
axes[1].barh(rf_feature_importance_df['feature'], rf_feature_importance_df['importance'], color='lightgreen')
axes[1].set_title('随机森林特征重要性')
axes[1].set_xlabel('重要性')

plt.tight_layout()
plt.show()

print("\n决策树特征重要性排序:")
print(dt_feature_importance_df)

print("\n随机森林特征重要性排序:")
print(rf_feature_importance_df)

# 9. ROC曲线
print("\n=== ROC曲线分析 ===")

# 获取预测概率
y_pred_proba_dt = best_dt.predict_proba(X_test)[:, 1]
y_pred_proba_rf = best_rf.predict_proba(X_test)[:, 1]

# 计算ROC曲线
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_proba_dt)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_proba_rf)

# 计算AUC
roc_auc_dt = auc(fpr_dt, tpr_dt)
roc_auc_rf = auc(fpr_rf, tpr_rf)

# 绘制ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr_dt, tpr_dt, color='blue', lw=2, label=f'决策树 (AUC = {roc_auc_dt:.4f})')
plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'随机森林 (AUC = {roc_auc_rf:.4f})')
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机分类器')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正率 (False Positive Rate)')
plt.ylabel('真正率 (True Positive Rate)')
plt.title('ROC曲线比较')
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

print(f"决策树AUC: {roc_auc_dt:.4f}")
print(f"随机森林AUC: {roc_auc_rf:.4f}")

# 10. 总结
print("\n=== 实验总结 ===")
print("1. 数据预处理: 成功处理了年龄转换、BMI计算和异常值")
print("2. 数据可视化: 完成了目标变量分布、特征分布和相关性分析")
print("3. 数据划分: 按60%-20%-20%划分训练集、验证集和测试集")
print("4. 不平衡处理: 使用SMOTE成功处理了类别不平衡问题")
print("5. 深度分析: 找到了决策树和随机森林的最佳深度")
print("6. 参数调优: 使用网格搜索优化了模型参数")
print("7. 模型评估: 在测试集上评估了模型性能")
print("8. 特征重要性: 分析了各特征对预测的贡献度")

print(f"\n最终结果:")
print(f"决策树 - 准确率: {dt_accuracy:.4f}, AUC: {roc_auc_dt:.4f}")
print(f"随机森林 - 准确率: {rf_accuracy:.4f}, AUC: {roc_auc_rf:.4f}")
print(f"最佳模型: {'随机森林' if rf_accuracy > dt_accuracy else '决策树'}")