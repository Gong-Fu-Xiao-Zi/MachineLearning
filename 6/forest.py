import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def analyze_overfitting_underfitting(X_train, y_train, X_val, y_val, dataset_name, model_type='both'):
    """
    分析过拟合和欠拟合情况
    """
    depths = range(1, 21)
    train_scores = []
    val_scores = []

    if model_type == 'dt' or model_type == 'both':
        print(f"\n{dataset_name} - 决策树深度分析:")
        for depth in depths:
            dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
            dt.fit(X_train, y_train)
            train_score = dt.score(X_train, y_train)
            val_score = dt.score(X_val, y_val)
            train_scores.append(train_score)
            val_scores.append(val_score)

        # 找到最佳深度
        best_depth_idx = np.argmax(val_scores)
        best_depth = depths[best_depth_idx]
        print(f"决策树最佳深度: {best_depth}, 验证集准确率: {val_scores[best_depth_idx]:.4f}")

        # 绘制过拟合/欠拟合分析图
        plt.figure(figsize=(10, 6))
        plt.plot(depths, train_scores, 'o-', label='训练集准确率', color='blue')
        plt.plot(depths, val_scores, 's-', label='验证集准确率', color='red')
        plt.axvline(x=best_depth, color='green', linestyle='--', label=f'最佳深度: {best_depth}')
        plt.xlabel('树深度')
        plt.ylabel('准确率')
        plt.title(f'{dataset_name} - 决策树过拟合/欠拟合分析')
        plt.legend()
        plt.grid(True)
        plt.show()

        # 分析过拟合程度
        overfitting_gap = train_scores[best_depth_idx] - val_scores[best_depth_idx]
        print(f"过拟合程度（训练集-验证集准确率差）: {overfitting_gap:.4f}")
        if overfitting_gap > 0.1:
            print("警告：存在明显过拟合")
        elif overfitting_gap < 0.02:
            print("模型拟合良好")
        else:
            print("存在轻微过拟合")

    # 随机森林分析
    if model_type == 'rf' or model_type == 'both':
        print(f"\n{dataset_name} - 随机森林深度分析:")
        rf_train_scores = []
        rf_val_scores = []

        for depth in depths:
            rf = RandomForestClassifier(max_depth=depth, n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            train_score = rf.score(X_train, y_train)
            val_score = rf.score(X_val, y_val)
            rf_train_scores.append(train_score)
            rf_val_scores.append(val_score)

        best_depth_idx_rf = np.argmax(rf_val_scores)
        best_depth_rf = depths[best_depth_idx_rf]
        print(f"随机森林最佳深度: {best_depth_rf}, 验证集准确率: {rf_val_scores[best_depth_idx_rf]:.4f}")

        plt.figure(figsize=(10, 6))
        plt.plot(depths, rf_train_scores, 'o-', label='训练集准确率', color='blue')
        plt.plot(depths, rf_val_scores, 's-', label='验证集准确率', color='red')
        plt.axvline(x=best_depth_rf, color='green', linestyle='--', label=f'最佳深度: {best_depth_rf}')
        plt.xlabel('树深度')
        plt.ylabel('准确率')
        plt.title(f'{dataset_name} - 随机森林过拟合/欠拟合分析')
        plt.legend()
        plt.grid(True)
        plt.show()

        overfitting_gap_rf = rf_train_scores[best_depth_idx_rf] - rf_val_scores[best_depth_idx_rf]
        print(f"过拟合程度: {overfitting_gap_rf:.4f}")
        if overfitting_gap_rf > 0.1:
            print("警告：存在明显过拟合")
        elif overfitting_gap_rf < 0.02:
            print("模型拟合良好")
        else:
            print("存在轻微过拟合")

        return best_depth, best_depth_rf


# =============================================================================
# 1. 冠心病数据集分析
# =============================================================================
print("\n1. 冠心病数据集分析")
print("-" * 40)

# 加载冠心病数据
heart_data = pd.read_csv('heart_improve.csv')
print(f"冠心病数据集形状: {heart_data.shape}")
print("数据集基本信息:")
print(heart_data.info())
print("\n前5行数据:")
print(heart_data.head())

# 数据预处理 - 处理缺失值
heart_processed = heart_data.copy()
numeric_columns = ['glucose', 'heartRate', 'education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI']
for col in numeric_columns:
    heart_processed[col].fillna(value=heart_processed[col].mean(), inplace=True)

print(f"\n处理后的冠心病数据形状: {heart_processed.shape}")
print("缺失值情况:")
print(heart_processed.isnull().sum())

# 冠心病数据可视化
x_heart = heart_processed.iloc[:, 0:-1]
y_heart = heart_processed['TenYearCHD']

color_map_heart = {0: 'yellow', 1: 'red'}  # 0:健康(黄色), 1:冠心病(红色)
colors_heart = list(map(lambda x: color_map_heart[x], heart_processed['TenYearCHD']))

fig, axs = plt.subplots(4, 4, figsize=(20, 10.8))
features_heart = list(x_heart.columns)

for i in range(4):
    for j in range(4):
        idx = i * 4 + j
        if idx < len(features_heart):
            axs[i, j].scatter(np.linspace(0, len(x_heart), len(x_heart)),
                              x_heart[features_heart[idx]], color=colors_heart, s=3, alpha=0.6)
            axs[i, j].set_title(features_heart[idx])
            axs[i, j].set_xlabel('样本索引')
            axs[i, j].set_ylabel(features_heart[idx])

plt.tight_layout()
plt.show()

# 冠心病数据集建模与比较
print("\n冠心病数据集建模与比较")
print("-" * 40)

# 提前划分训练集、验证集和测试集
X_temp, X_test_heart, y_temp, y_test_heart = train_test_split(
    x_heart, y_heart, test_size=0.2, random_state=42, stratify=y_heart
)
X_train_heart, X_val_heart, y_train_heart, y_val_heart = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)  # 最终划分：60%训练，20%验证，20%测试

print(f"训练集形状: {X_train_heart.shape}")
print(f"验证集形状: {X_val_heart.shape}")
print(f"测试集形状: {X_test_heart.shape}")

# 处理不平衡问题 - 使用SMOTE过采样
print("\n应用SMOTE过采样处理不平衡问题...")
smote = SMOTE(random_state=42)
X_train_heart_resampled, y_train_heart_resampled = smote.fit_resample(X_train_heart, y_train_heart)
print(f"过采样后训练集形状: {X_train_heart_resampled.shape}")
print(f"过采样后类别分布: {pd.Series(y_train_heart_resampled).value_counts()}")

# 特征标准化
scaler_heart = StandardScaler()
X_train_heart_scaled = scaler_heart.fit_transform(X_train_heart_resampled)
X_val_heart_scaled = scaler_heart.transform(X_val_heart)
X_test_heart_scaled = scaler_heart.transform(X_test_heart)

# 深度分析与过拟合/欠拟合比较
best_depth_dt_heart, best_depth_rf_heart = analyze_overfitting_underfitting(
    X_train_heart_scaled, y_train_heart_resampled,
    X_val_heart_scaled, y_val_heart,
    "冠心病数据集"
)

# 参数调优 - 决策树
print("\n冠心病数据集 - 决策树参数调优:")
dt_heart_param_grid = {
    'max_depth': [best_depth_dt_heart - 1, best_depth_dt_heart, best_depth_dt_heart + 1, best_depth_dt_heart + 2],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_heart_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_heart_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
dt_heart_grid.fit(X_train_heart_scaled, y_train_heart_resampled)

print(f"决策树最优参数: {dt_heart_grid.best_params_}")
print(f"决策树最优交叉验证分数: {dt_heart_grid.best_score_:.4f}")

best_dt_heart = dt_heart_grid.best_estimator_
y_val_pred_dt_heart = best_dt_heart.predict(X_val_heart_scaled)
dt_heart_val_accuracy = accuracy_score(y_val_heart, y_val_pred_dt_heart)
print(f"决策树验证集准确率: {dt_heart_val_accuracy:.4f}")

# 参数调优 - 随机森林
print("\n冠心病数据集 - 随机森林参数调优:")
rf_heart_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [best_depth_rf_heart - 1, best_depth_rf_heart, best_depth_rf_heart + 1],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_heart_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_heart_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_heart_grid.fit(X_train_heart_scaled, y_train_heart_resampled)

print(f"随机森林最优参数: {rf_heart_grid.best_params_}")
print(f"随机森林最优交叉验证分数: {rf_heart_grid.best_score_:.4f}")

best_rf_heart = rf_heart_grid.best_estimator_
y_val_pred_rf_heart = best_rf_heart.predict(X_val_heart_scaled)
rf_heart_val_accuracy = accuracy_score(y_val_heart, y_val_pred_rf_heart)
print(f"随机森林验证集准确率: {rf_heart_val_accuracy:.4f}")

# 在测试集上评估最终模型
print("\n冠心病数据集 - 测试集最终评估:")
y_test_pred_dt_heart = best_dt_heart.predict(X_test_heart_scaled)
y_test_pred_rf_heart = best_rf_heart.predict(X_test_heart_scaled)

dt_heart_test_accuracy = accuracy_score(y_test_heart, y_test_pred_dt_heart)
rf_heart_test_accuracy = accuracy_score(y_test_heart, y_test_pred_rf_heart)

print(f"决策树测试集准确率: {dt_heart_test_accuracy:.4f}")
print(f"随机森林测试集准确率: {rf_heart_test_accuracy:.4f}")

# 可视化预测结果对比
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 决策树预测结果
axes[0].scatter(x=np.arange(len(y_test_heart)), y=y_test_heart, s=10, c='blue', alpha=0.7, label='真实值')
axes[0].scatter(x=np.arange(len(y_test_heart)), y=y_test_pred_dt_heart, s=10, c='red', alpha=0.7, label='预测值')
axes[0].set_title(f'冠心病 - 决策树预测结果 (准确率: {dt_heart_test_accuracy:.4f})')
axes[0].legend()
axes[0].set_xlabel('样本索引')
axes[0].set_ylabel('冠心病风险')

# 随机森林预测结果
axes[1].scatter(x=np.arange(len(y_test_heart)), y=y_test_heart, s=10, c='blue', alpha=0.7, label='真实值')
axes[1].scatter(x=np.arange(len(y_test_heart)), y=y_test_pred_rf_heart, s=10, c='red', alpha=0.7, label='预测值')
axes[1].set_title(f'冠心病 - 随机森林预测结果 (准确率: {rf_heart_test_accuracy:.4f})')
axes[1].legend()
axes[1].set_xlabel('样本索引')
axes[1].set_ylabel('冠心病风险')

plt.tight_layout()
plt.show()

# 特征重要性分析
print("\n冠心病数据集 - 特征重要性分析:")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 决策树特征重要性
dt_heart_importance = pd.DataFrame({
    'feature': x_heart.columns,
    'importance': best_dt_heart.feature_importances_
}).sort_values('importance', ascending=True)

axes[0].barh(dt_heart_importance['feature'], dt_heart_importance['importance'])
axes[0].set_title('冠心病 - 决策树特征重要性')
axes[0].set_xlabel('重要性')

# 随机森林特征重要性
rf_heart_importance = pd.DataFrame({
    'feature': x_heart.columns,
    'importance': best_rf_heart.feature_importances_
}).sort_values('importance', ascending=True)

axes[1].barh(rf_heart_importance['feature'], rf_heart_importance['importance'])
axes[1].set_title('冠心病 - 随机森林特征重要性')
axes[1].set_xlabel('重要性')

plt.tight_layout()
plt.show()

# =============================================================================
# 2. 泰坦尼克号数据集分析
# =============================================================================
print("\n2. 泰坦尼克号数据集分析")
print("-" * 40)

# 加载泰坦尼克号数据
titanic_data = pd.read_csv('titanic_data.csv')
print(f"泰坦尼克号数据集形状: {titanic_data.shape}")
print("数据集基本信息:")
print(titanic_data.info())
print("\n前5行数据:")
print(titanic_data.head())

# 数据预处理
titanic_processed = titanic_data.copy()
titanic_processed.replace(to_replace='male', value=0, inplace=True)
titanic_processed.replace(to_replace='female', value=1, inplace=True)

# 处理缺失值
titanic_processed['Age'].fillna(value=titanic_processed['Age'].mean(), inplace=True)
titanic_processed['Embarked'].fillna(titanic_processed['Embarked'].mode()[0], inplace=True)
titanic_processed['Fare'].fillna(titanic_processed['Fare'].median(), inplace=True)

# 选择更多特征用于分析
features_to_use = ['Pclass', 'Age', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked']
x_titanic = titanic_processed[features_to_use]
y_titanic = titanic_processed['Survived']

# 编码分类变量
label_encoders = {}
for col in ['Embarked']:
    le = LabelEncoder()
    x_titanic[col] = le.fit_transform(x_titanic[col].astype(str))
    label_encoders[col] = le

print(f"\n处理后的特征数据形状: {x_titanic.shape}")
print("缺失值情况:")
print(x_titanic.isnull().sum())

# 泰坦尼克号数据可视化
color_map = {0: 'red', 1: 'green'}  # 0:未生存(红色), 1:生存(绿色)
colors_titanic = list(map(lambda x: color_map[x], titanic_processed['Survived']))

fig, axs = plt.subplots(2, 4, figsize=(20, 10))
features_titanic = list(x_titanic.columns)

for i in range(2):
    for j in range(4):
        idx = i * 4 + j
        if idx < len(features_titanic):
            axs[i, j].scatter(np.linspace(0, len(x_titanic), len(x_titanic)),
                              x_titanic[features_titanic[idx]], color=colors_titanic, s=5, alpha=0.6)
            axs[i, j].set_title(f'{features_titanic[idx]} 特征分布')
            axs[i, j].set_xlabel('样本索引')
            axs[i, j].set_ylabel(features_titanic[idx])
        else:
            axs[i, j].axis('off')  # 隐藏多余的子图

# 添加图例
from matplotlib.patches import Patch

legend_elements = [Patch(facecolor='red', label='未生存'),
                   Patch(facecolor='green', label='生存')]
axs[1, 3].legend(handles=legend_elements, loc='center')
axs[1, 3].set_title('图例')
axs[1, 3].axis('off')

plt.tight_layout()
plt.show()

# 泰坦尼克号数据集建模与比较
print("\n泰坦尼克号数据集建模与比较")
print("-" * 40)

# 提前划分训练集、验证集和测试集
X_temp_t, X_test_titanic, y_temp_t, y_test_titanic = train_test_split(
    x_titanic, y_titanic, test_size=0.2, random_state=42, stratify=y_titanic
)
X_train_titanic, X_val_titanic, y_train_titanic, y_val_titanic = train_test_split(
    X_temp_t, y_temp_t, test_size=0.25, random_state=42, stratify=y_temp_t
)

print(f"训练集形状: {X_train_titanic.shape}")
print(f"验证集形状: {X_val_titanic.shape}")
print(f"测试集形状: {X_test_titanic.shape}")

# 处理不平衡问题 - 使用SMOTE过采样
print("\n应用SMOTE过采样处理不平衡问题...")
smote_titanic = SMOTE(random_state=42)
X_train_titanic_resampled, y_train_titanic_resampled = smote_titanic.fit_resample(X_train_titanic, y_train_titanic)
print(f"过采样后训练集形状: {X_train_titanic_resampled.shape}")
print(f"过采样后类别分布: {pd.Series(y_train_titanic_resampled).value_counts()}")

# 深度分析与过拟合/欠拟合比较
best_depth_dt_titanic, best_depth_rf_titanic = analyze_overfitting_underfitting(
    X_train_titanic_resampled, y_train_titanic_resampled,
    X_val_titanic, y_val_titanic,
    "泰坦尼克号数据集"
)

# 参数调优 - 决策树
print("\n泰坦尼克号数据集 - 决策树参数调优:")
dt_titanic_param_grid = {
    'max_depth': [best_depth_dt_titanic - 1, best_depth_dt_titanic, best_depth_dt_titanic + 1,
                  best_depth_dt_titanic + 2],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_titanic_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    dt_titanic_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
dt_titanic_grid.fit(X_train_titanic_resampled, y_train_titanic_resampled)

print(f"决策树最优参数: {dt_titanic_grid.best_params_}")
print(f"决策树最优交叉验证分数: {dt_titanic_grid.best_score_:.4f}")

best_dt_titanic = dt_titanic_grid.best_estimator_
y_val_pred_dt_titanic = best_dt_titanic.predict(X_val_titanic)
dt_titanic_val_accuracy = accuracy_score(y_val_titanic, y_val_pred_dt_titanic)
print(f"决策树验证集准确率: {dt_titanic_val_accuracy:.4f}")

# 参数调优 - 随机森林
print("\n泰坦尼克号数据集 - 随机森林参数调优:")
rf_titanic_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [best_depth_rf_titanic - 1, best_depth_rf_titanic, best_depth_rf_titanic + 1],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_titanic_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    rf_titanic_param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
rf_titanic_grid.fit(X_train_titanic_resampled, y_train_titanic_resampled)

print(f"随机森林最优参数: {rf_titanic_grid.best_params_}")
print(f"随机森林最优交叉验证分数: {rf_titanic_grid.best_score_:.4f}")

best_rf_titanic = rf_titanic_grid.best_estimator_
y_val_pred_rf_titanic = best_rf_titanic.predict(X_val_titanic)
rf_titanic_val_accuracy = accuracy_score(y_val_titanic, y_val_pred_rf_titanic)
print(f"随机森林验证集准确率: {rf_titanic_val_accuracy:.4f}")

# 在测试集上评估最终模型
print("\n泰坦尼克号数据集 - 测试集最终评估:")
y_test_pred_dt_titanic = best_dt_titanic.predict(X_test_titanic)
y_test_pred_rf_titanic = best_rf_titanic.predict(X_test_titanic)

dt_titanic_test_accuracy = accuracy_score(y_test_titanic, y_test_pred_dt_titanic)
rf_titanic_test_accuracy = accuracy_score(y_test_titanic, y_test_pred_rf_titanic)

print(f"决策树测试集准确率: {dt_titanic_test_accuracy:.4f}")
print(f"随机森林测试集准确率: {rf_titanic_test_accuracy:.4f}")

# 可视化预测结果对比
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 决策树预测结果
axes[0].scatter(x=np.arange(len(y_test_titanic)), y=y_test_titanic, s=10, c='blue', alpha=0.7, label='真实值')
axes[0].scatter(x=np.arange(len(y_test_titanic)), y=y_test_pred_dt_titanic, s=10, c='red', alpha=0.7, label='预测值')
axes[0].set_title(f'泰坦尼克号 - 决策树预测结果 (准确率: {dt_titanic_test_accuracy:.4f})')
axes[0].legend()
axes[0].set_xlabel('样本索引')
axes[0].set_ylabel('生存情况')

# 随机森林预测结果
axes[1].scatter(x=np.arange(len(y_test_titanic)), y=y_test_titanic, s=10, c='blue', alpha=0.7, label='真实值')
axes[1].scatter(x=np.arange(len(y_test_titanic)), y=y_test_pred_rf_titanic, s=10, c='red', alpha=0.7, label='预测值')
axes[1].set_title(f'泰坦尼克号 - 随机森林预测结果 (准确率: {rf_titanic_test_accuracy:.4f})')
axes[1].legend()
axes[1].set_xlabel('样本索引')
axes[1].set_ylabel('生存情况')

plt.tight_layout()
plt.show()

# 特征重要性分析
print("\n泰坦尼克号数据集 - 特征重要性分析:")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 决策树特征重要性
dt_titanic_importance = pd.DataFrame({
    'feature': x_titanic.columns,
    'importance': best_dt_titanic.feature_importances_
}).sort_values('importance', ascending=True)

axes[0].barh(dt_titanic_importance['feature'], dt_titanic_importance['importance'])
axes[0].set_title('泰坦尼克号 - 决策树特征重要性')
axes[0].set_xlabel('重要性')

# 随机森林特征重要性
rf_titanic_importance = pd.DataFrame({
    'feature': x_titanic.columns,
    'importance': best_rf_titanic.feature_importances_
}).sort_values('importance', ascending=True)

axes[1].barh(rf_titanic_importance['feature'], rf_titanic_importance['importance'])
axes[1].set_title('泰坦尼克号 - 随机森林特征重要性')
axes[1].set_xlabel('重要性')

plt.tight_layout()
plt.show()

# =============================================================================
# 3. 综合比较结果
# =============================================================================
print("\n3. 综合比较结果")
print("-" * 40)

# 创建比较表格
comparison_data = {
    '数据集': ['冠心病', '冠心病', '泰坦尼克号', '泰坦尼克号'],
    '模型': ['决策树', '随机森林', '决策树', '随机森林'],
    '验证集准确率': [dt_heart_val_accuracy, rf_heart_val_accuracy, dt_titanic_val_accuracy, rf_titanic_val_accuracy],
    '测试集准确率': [dt_heart_test_accuracy, rf_heart_test_accuracy, dt_titanic_test_accuracy,
                     rf_titanic_test_accuracy],
    '最优深度': [best_depth_dt_heart, best_depth_rf_heart, best_depth_dt_titanic, best_depth_rf_titanic]
}

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df)

# 可视化比较结果
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 测试集准确率比较
sns.barplot(x='数据集', y='测试集准确率', hue='模型', data=comparison_df, ax=axes[0])
axes[0].set_title('两个数据集上模型测试集准确率比较')
axes[0].set_ylim(0, 1)

# 最优深度比较
sns.barplot(x='数据集', y='最优深度', hue='模型', data=comparison_df, ax=axes[1])
axes[1].set_title('两个数据集上模型最优深度比较')

plt.tight_layout()
plt.show()

# 4. 详细评估报告
print("\n4. 详细模型评估报告")
print("-" * 40)

print("冠心病数据集 - 决策树:")
print(classification_report(y_test_heart, y_test_pred_dt_heart))

print("冠心病数据集 - 随机森林:")
print(classification_report(y_test_heart, y_test_pred_rf_heart))

print("泰坦尼克号数据集 - 决策树:")
print(classification_report(y_test_titanic, y_test_pred_dt_titanic))

print("泰坦尼克号数据集 - 随机森林:")
print(classification_report(y_test_titanic, y_test_pred_rf_titanic))

# 混淆矩阵可视化
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 冠心病决策树
cm1 = confusion_matrix(y_test_heart, y_test_pred_dt_heart)
sns.heatmap(cm1, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
axes[0, 0].set_title('冠心病 - 决策树')

# 冠心病随机森林
cm2 = confusion_matrix(y_test_heart, y_test_pred_rf_heart)
sns.heatmap(cm2, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1])
axes[0, 1].set_title('冠心病 - 随机森林')

# 泰坦尼克号决策树
cm3 = confusion_matrix(y_test_titanic, y_test_pred_dt_titanic)
sns.heatmap(cm3, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0])
axes[1, 0].set_title('泰坦尼克号 - 决策树')

# 泰坦尼克号随机森林
cm4 = confusion_matrix(y_test_titanic, y_test_pred_rf_titanic)
sns.heatmap(cm4, annot=True, fmt='d', cmap='Greens', ax=axes[1, 1])
axes[1, 1].set_title('泰坦尼克号 - 随机森林')

plt.tight_layout()
plt.show()
