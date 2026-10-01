import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, roc_curve
from sklearn import tree
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 读取数据
data = pd.read_csv('heart_improve.csv')
data = pd.DataFrame(data)
print("数据形状:", data.shape)
print("目标变量分布:\n", data['TenYearCHD'].value_counts())

# 处理缺失值
numeric_columns = ['glucose', 'heartRate', 'education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI']
for col in numeric_columns:
    data[col].fillna(value=data[col].mean(), inplace=True)

# 准备特征和目标变量
x = data.iloc[:, 0:-1]
y = data['TenYearCHD']

print(f"目标变量分布: 0-{sum(y == 0)}个, 1-{sum(y == 1)}个")
print(f"正负样本比例: {sum(y == 1) / sum(y == 0):.3f}")

# 数据标准化
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
x_scaled = pd.DataFrame(x_scaled, columns=x.columns)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, random_state=22, test_size=0.25, stratify=y)

print(f"训练集形状: {x_train.shape}, 测试集形状: {x_test.shape}")
print(f"训练集目标分布: {pd.Series(y_train).value_counts().to_dict()}")

# 1. 处理不平衡数据 - 使用SMOTE过采样
print("\n" + "=" * 50)
print("1. 处理不平衡数据")
print("=" * 50)

# 原始数据训练
dt_original = DecisionTreeClassifier(max_depth=8, random_state=42)
dt_original.fit(x_train, y_train)
y_pred_original = dt_original.predict(x_test)

print("原始数据 - 准确率:", dt_original.score(x_test, y_test))
print("原始数据 - AUC:", roc_auc_score(y_test, dt_original.predict_proba(x_test)[:, 1]))

# 使用SMOTE过采样
smote = SMOTE(random_state=42)
x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)

print(f"过采样后训练集形状: {x_train_smote.shape}")
print(f"过采样后目标分布: {pd.Series(y_train_smote).value_counts().to_dict()}")

dt_smote = DecisionTreeClassifier(max_depth=8, random_state=42)
dt_smote.fit(x_train_smote, y_train_smote)
y_pred_smote = dt_smote.predict(x_test)

print("SMOTE过采样 - 准确率:", dt_smote.score(x_test, y_test))
print("SMOTE过采样 - AUC:", roc_auc_score(y_test, dt_smote.predict_proba(x_test)[:, 1]))

# 2. 不同深度决策树比较（过拟合 vs 欠拟合）
print("\n" + "=" * 50)
print("2. 不同深度决策树比较")
print("=" * 50)

depths = [2, 4, 6, 8, 10, 12, 15, 20, None]
train_scores = []
test_scores = []
auc_scores = []

for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(x_train_smote, y_train_smote)

    train_score = dt.score(x_train_smote, y_train_smote)
    test_score = dt.score(x_test, y_test)
    auc_score = roc_auc_score(y_test, dt.predict_proba(x_test)[:, 1])

    train_scores.append(train_score)
    test_scores.append(test_score)
    auc_scores.append(auc_score)

    print(f"深度 {depth}: 训练集准确率={train_score:.4f}, 测试集准确率={test_score:.4f}, AUC={auc_score:.4f}")

# 绘制不同深度的性能比较
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot([str(d) for d in depths], train_scores, 'o-', label='训练集准确率')
plt.plot([str(d) for d in depths], test_scores, 'o-', label='测试集准确率')
plt.xlabel('树深度')
plt.ylabel('准确率')
plt.title('不同深度决策树性能比较')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot([str(d) for d in depths], auc_scores, 'o-', color='red', label='AUC')
plt.xlabel('树深度')
plt.ylabel('AUC')
plt.title('不同深度决策树AUC比较')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# 3. 预剪枝 vs 后剪枝比较
print("\n" + "=" * 50)
print("3. 预剪枝 vs 后剪枝比较")
print("=" * 50)

# 预剪枝参数
pre_pruning_params = {
    'max_depth': [4, 6, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_leaf_nodes': [None, 20, 50]
}

# 使用网格搜索找到最佳预剪枝参数
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=42),
    pre_pruning_params,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1
)
grid_search.fit(x_train_smote, y_train_smote)

print("最佳预剪枝参数:", grid_search.best_params_)
print("最佳预剪枝分数:", grid_search.best_score_)

best_pre_pruned = grid_search.best_estimator_
y_pred_pre_pruned = best_pre_pruned.predict(x_test)

# 后剪枝 (Cost Complexity Pruning)
dt_full = DecisionTreeClassifier(random_state=42, max_depth=15)
dt_full.fit(x_train_smote, y_train_smote)

# 获取剪枝路径
path = dt_full.cost_complexity_pruning_path(x_train_smote, y_train_smote)
ccp_alphas = path.ccp_alphas

# 尝试不同的alpha值进行后剪枝
post_pruning_scores = []
for ccp_alpha in ccp_alphas[:20]:  # 只尝试前20个alpha值
    if ccp_alpha >= 0:
        dt_post_pruned = DecisionTreeClassifier(
            random_state=42,
            ccp_alpha=ccp_alpha
        )
        dt_post_pruned.fit(x_train_smote, y_train_smote)
        score = roc_auc_score(y_test, dt_post_pruned.predict_proba(x_test)[:, 1])
        post_pruning_scores.append((ccp_alpha, score))

# 找到最佳后剪枝参数
if post_pruning_scores:
    best_alpha, best_score = max(post_pruning_scores, key=lambda x: x[1])
    print(f"最佳后剪枝alpha: {best_alpha:.6f}, AUC: {best_score:.4f}")

    best_post_pruned = DecisionTreeClassifier(random_state=42, ccp_alpha=best_alpha)
    best_post_pruned.fit(x_train_smote, y_train_smote)
    y_pred_post_pruned = best_post_pruned.predict(x_test)
else:
    print("后剪枝失败，使用预剪枝模型")
    best_post_pruned = best_pre_pruned
    y_pred_post_pruned = y_pred_pre_pruned

# 绘制剪枝比较
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
if post_pruning_scores:
    alphas = [x[0] for x in post_pruning_scores]
    scores = [x[1] for x in post_pruning_scores]
    plt.plot(alphas, scores, 'o-')
    plt.xlabel('CCP Alpha')
    plt.ylabel('AUC')
    plt.title('后剪枝参数选择')
    plt.grid(True)

# 4. 模型比较和特征重要性
print("\n" + "=" * 50)
print("4. 最终模型比较")
print("=" * 50)

models = {
    '原始数据': dt_original,
    'SMOTE过采样': dt_smote,
    '预剪枝': best_pre_pruned,
    '后剪枝': best_post_pruned
}

results = []
for name, model in models.items():
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)[:, 1]

    accuracy = model.score(x_test, y_test)
    auc = roc_auc_score(y_test, y_prob)

    # 计算召回率（对少数类很重要）
    cm = confusion_matrix(y_test, y_pred)
    recall_1 = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0

    results.append({
        '模型': name,
        '准确率': accuracy,
        'AUC': auc,
        '召回率(1)': recall_1
    })

results_df = pd.DataFrame(results)
print(results_df)

# 绘制模型比较
plt.subplot(1, 2, 2)
x_pos = np.arange(len(models))
metrics = ['准确率', 'AUC', '召回率(1)']

for i, metric in enumerate(metrics):
    values = [results_df[results_df['模型'] == model][metric].values[0] for model in models.keys()]
    plt.bar(x_pos + i * 0.25, values, width=0.25, label=metric, alpha=0.8)

plt.xlabel('模型')
plt.ylabel('分数')
plt.title('不同模型性能比较')
plt.xticks(x_pos + 0.25, list(models.keys()), rotation=45)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 5. 特征重要性分析
print("\n" + "=" * 50)
print("5. 特征重要性分析")
print("=" * 50)

# 使用最佳模型分析特征重要性
best_model = best_post_pruned
feature_importance = pd.DataFrame({
    'feature': x.columns,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

print("特征重要性排序:")
print(feature_importance)

# 绘制特征重要性
plt.figure(figsize=(10, 8))
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('特征重要性')
plt.title('决策树特征重要性排序')
plt.tight_layout()
plt.show()

# 6. 详细分类报告和ROC曲线
print("\n" + "=" * 50)
print("6. 最佳模型详细评估")
print("=" * 50)

print("最佳模型 (后剪枝) 分类报告:")
print(classification_report(y_test, y_pred_post_pruned, target_names=["健康", "亚健康"]))

# 绘制混淆矩阵和ROC曲线
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred_post_pruned)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['预测健康', '预测亚健康'],
            yticklabels=['实际健康', '实际亚健康'])
plt.title('混淆矩阵')

plt.subplot(1, 2, 2)
# 绘制ROC曲线
for name, model in models.items():
    y_prob = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc_score:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='随机分类器')
plt.xlabel('假正率')
plt.ylabel('真正率')
plt.title('ROC曲线比较')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
