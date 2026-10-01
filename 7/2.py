import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, \
    precision_recall_curve
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import warnings
import time

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载和预处理
print("=== 心血管疾病预测分析 - 支持向量机 ===")

# 加载数据
df = pd.read_csv('5_心血管疾病预测.csv')

print(f"数据集形状: {df.shape}")
print("\n数据前5行:")
print(df.head())

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
plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
df['cardio'].value_counts().plot(kind='bar', color=['skyblue', 'lightcoral'])
plt.title('目标变量分布 (cardio)')
plt.xlabel('是否有心血管疾病')
plt.ylabel('数量')
plt.xticks(rotation=0)

# 年龄与心血管疾病的关系
plt.subplot(2, 2, 2)
sns.boxplot(x='cardio', y='age', data=df, palette=['skyblue', 'lightcoral'])
plt.title('年龄与心血管疾病的关系')
plt.xlabel('是否有心血管疾病')
plt.ylabel('年龄 (年)')

# BMI与心血管疾病的关系
plt.subplot(2, 2, 3)
sns.boxplot(x='cardio', y='bmi', data=df, palette=['skyblue', 'lightcoral'])
plt.title('BMI与心血管疾病的关系')
plt.xlabel('是否有心血管疾病')
plt.ylabel('BMI')

# 收缩压与心血管疾病的关系
plt.subplot(2, 2, 4)
sns.boxplot(x='cardio', y='ap_hi', data=df, palette=['skyblue', 'lightcoral'])
plt.title('收缩压与心血管疾病的关系')
plt.xlabel('是否有心血管疾病')
plt.ylabel('收缩压')

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

# 数据标准化 (SVM对特征尺度敏感)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# 处理不平衡数据 - 使用SMOTE过采样
print("\n应用SMOTE过采样...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"过采样后训练集形状: {X_train_resampled.shape}")
print(f"过采样后目标变量分布: {pd.Series(y_train_resampled).value_counts()}")

# 4. 不同核函数的SVM性能比较
print("\n=== 不同核函数性能比较 ===")

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_scores = {}

for kernel in kernels:
    print(f"训练 {kernel} 核SVM...")
    svm = SVC(kernel=kernel, random_state=42, probability=True)
    svm.fit(X_train_resampled, y_train_resampled)
    val_score = svm.score(X_val_scaled, y_val)
    kernel_scores[kernel] = val_score
    print(f"{kernel} 核SVM验证集准确率: {val_score:.4f}")

# 可视化不同核函数的性能
plt.figure(figsize=(10, 6))
plt.bar(kernel_scores.keys(), kernel_scores.values(), color=['blue', 'green', 'red', 'purple'])
plt.title('不同核函数SVM在验证集上的性能')
plt.ylabel('准确率')
plt.ylim(0, 1)
for i, v in enumerate(kernel_scores.values()):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')
plt.grid(True, alpha=0.3)
plt.show()

# 选择最佳核函数
best_kernel = max(kernel_scores, key=kernel_scores.get)
print(f"最佳核函数: {best_kernel}, 准确率: {kernel_scores[best_kernel]:.4f}")

# 5. 参数调优
print("\n=== SVM参数调优 ===")

# 根据最佳核函数设置不同的参数网格
if best_kernel == 'linear':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'class_weight': [None, 'balanced']
    }
elif best_kernel == 'poly':
    param_grid = {
        'C': [0.1, 1, 10],
        'degree': [2, 3, 4],
        'gamma': ['scale', 'auto']
    }
elif best_kernel == 'rbf':
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'class_weight': [None, 'balanced']
    }
else:  # sigmoid
    param_grid = {
        'C': [0.1, 1, 10],
        'gamma': ['scale', 'auto', 0.1],
        'coef0': [0, 1]
    }
start_time = time.time()
print(f"使用 {best_kernel} 核进行参数调优...")
svm_grid = GridSearchCV(
    SVC(kernel=best_kernel, random_state=42, probability=True),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=1
)

svm_grid.fit(X_train_resampled, y_train_resampled)

print(f"SVM最佳参数: {svm_grid.best_params_}")
print(f"SVM最佳交叉验证分数: {svm_grid.best_score_:.4f}")
end_time = time.time()
print("耗时: {:.2f}秒".format(end_time - start_time))
# 6. 模型评估
print("\n=== 模型评估 ===")

# 使用最佳参数训练最终模型
best_svm = svm_grid.best_estimator_

# 在验证集上评估
y_val_pred = best_svm.predict(X_val_scaled)
y_val_pred_proba = best_svm.predict_proba(X_val_scaled)[:, 1]
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"SVM验证集准确率: {val_accuracy:.4f}")

# 在测试集上评估
y_test_pred = best_svm.predict(X_test_scaled)
y_test_pred_proba = best_svm.predict_proba(X_test_scaled)[:, 1]
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"SVM测试集准确率: {test_accuracy:.4f}")

# 7. 结果可视化
print("\n=== 结果可视化 ===")

# 混淆矩阵
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 验证集混淆矩阵
cm_val = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'SVM验证集混淆矩阵\n准确率: {val_accuracy:.4f}')
axes[0].set_xlabel('预测标签')
axes[0].set_ylabel('真实标签')

# 测试集混淆矩阵
cm_test = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f'SVM测试集混淆矩阵\n准确率: {test_accuracy:.4f}')
axes[1].set_xlabel('预测标签')
axes[1].set_ylabel('真实标签')

plt.tight_layout()
plt.show()

# 分类报告
print("\nSVM验证集分类报告:")
print(classification_report(y_val, y_val_pred))

print("\nSVM测试集分类报告:")
print(classification_report(y_test, y_test_pred))

# 8. ROC曲线和PR曲线
print("\n=== ROC曲线和PR曲线 ===")

# ROC曲线
fpr_val, tpr_val, _ = roc_curve(y_val, y_val_pred_proba)
fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred_proba)

roc_auc_val = auc(fpr_val, tpr_val)
roc_auc_test = auc(fpr_test, tpr_test)

# PR曲线
precision_val, recall_val, _ = precision_recall_curve(y_val, y_val_pred_proba)
precision_test, recall_test, _ = precision_recall_curve(y_test, y_test_pred_proba)

pr_auc_val = auc(recall_val, precision_val)
pr_auc_test = auc(recall_test, precision_test)

# 绘制ROC和PR曲线
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# ROC曲线
axes[0].plot(fpr_val, tpr_val, color='blue', lw=2,
             label=f'验证集 ROC (AUC = {roc_auc_val:.4f})')
axes[0].plot(fpr_test, tpr_test, color='red', lw=2,
             label=f'测试集 ROC (AUC = {roc_auc_test:.4f})')
axes[0].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='随机分类器')
axes[0].set_xlim([0.0, 1.0])
axes[0].set_ylim([0.0, 1.05])
axes[0].set_xlabel('假正率 (False Positive Rate)')
axes[0].set_ylabel('真正率 (True Positive Rate)')
axes[0].set_title('SVM ROC曲线')
axes[0].legend(loc="lower right")
axes[0].grid(True)

# PR曲线
axes[1].plot(recall_val, precision_val, color='blue', lw=2,
             label=f'验证集 PR (AUC = {pr_auc_val:.4f})')
axes[1].plot(recall_test, precision_test, color='red', lw=2,
             label=f'测试集 PR (AUC = {pr_auc_test:.4f})')
axes[1].set_xlim([0.0, 1.0])
axes[1].set_ylim([0.0, 1.05])
axes[1].set_xlabel('召回率 (Recall)')
axes[1].set_ylabel('精确率 (Precision)')
axes[1].set_title('SVM PR曲线')
axes[1].legend(loc="lower left")
axes[1].grid(True)

plt.tight_layout()
plt.show()

print(f"验证集ROC AUC: {roc_auc_val:.4f}")
print(f"测试集ROC AUC: {roc_auc_test:.4f}")
print(f"验证集PR AUC: {pr_auc_val:.4f}")
print(f"测试集PR AUC: {pr_auc_test:.4f}")

# 9. 学习曲线分析
print("\n=== 学习曲线分析 ===")

from sklearn.model_selection import learning_curve

train_sizes, train_scores, test_scores = learning_curve(
    best_svm, X_train_resampled, y_train_resampled,
    cv=5, n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 10),
    scoring='accuracy'
)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
test_scores_mean = np.mean(test_scores, axis=1)
test_scores_std = np.std(test_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label='训练得分')
plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                 train_scores_mean + train_scores_std, alpha=0.1, color='blue')
plt.plot(train_sizes, test_scores_mean, 'o-', color='green', label='交叉验证得分')
plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                 test_scores_mean + test_scores_std, alpha=0.1, color='green')
plt.xlabel('训练样本数')
plt.ylabel('准确率')
plt.title('SVM学习曲线')
plt.legend(loc='best')
plt.grid(True)
plt.show()

# 10. 特征重要性分析（对于线性SVM）
print("\n=== 特征重要性分析 ===")

if best_kernel == 'linear':
    # 线性SVM的特征权重
    feature_importance = np.abs(best_svm.coef_[0])
    feature_importance_df = pd.DataFrame({
        'feature': features,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'], color='lightcoral')
    plt.title('线性SVM特征重要性（权重绝对值）')
    plt.xlabel('重要性')
    plt.tight_layout()
    plt.show()

    print("特征重要性排序:")
    print(feature_importance_df)
else:
    print(f"对于 {best_kernel} 核SVM，无法直接获取特征重要性")

    # 使用排列重要性
    from sklearn.inspection import permutation_importance

    print("计算排列重要性...")
    perm_importance = permutation_importance(
        best_svm, X_val_scaled, y_val,
        n_repeats=10, random_state=42, n_jobs=-1
    )

    perm_importance_df = pd.DataFrame({
        'feature': features,
        'importance': perm_importance.importances_mean
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 8))
    plt.barh(perm_importance_df['feature'], perm_importance_df['importance'], color='lightseagreen')
    plt.title(f'{best_kernel}核SVM排列重要性')
    plt.xlabel('排列重要性')
    plt.tight_layout()
    plt.show()

    print("排列重要性排序:")
    print(perm_importance_df)

# 11. 交叉验证评估
print("\n=== 交叉验证评估 ===")

cv_scores = cross_val_score(best_svm, X_train_resampled, y_train_resampled,
                            cv=5, scoring='accuracy', n_jobs=-1)

print(f"5折交叉验证准确率: {cv_scores}")
print(f"交叉验证平均准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 12. 总结
print("\n=== 实验总结 ===")
print("1. 数据预处理: 完成了年龄转换、BMI计算和异常值处理")
print("2. 特征标准化: 对特征进行了标准化处理（SVM对尺度敏感）")
print("3. 不平衡处理: 使用SMOTE成功处理了类别不平衡问题")
print("4. 核函数比较: 比较了linear、poly、rbf、sigmoid四种核函数")
print(f"5. 参数调优: 使用网格搜索优化了{best_kernel}核SVM的参数")
print("6. 模型评估: 在验证集和测试集上评估了模型性能")
print("7. 曲线分析: 绘制了ROC曲线和PR曲线")
print("8. 学习曲线: 分析了模型的学习过程")
print("9. 特征重要性: 分析了各特征对预测的贡献度")
print("10. 交叉验证: 使用5折交叉验证验证了模型稳定性")

print(f"\n最终结果:")
print(f"最佳核函数: {best_kernel}")
print(f"最佳参数: {svm_grid.best_params_}")
print(f"测试集准确率: {test_accuracy:.4f}")
print(f"测试集ROC AUC: {roc_auc_test:.4f}")
print(f"测试集PR AUC: {pr_auc_test:.4f}")
print(f"交叉验证平均准确率: {cv_scores.mean():.4f}")

# 保存模型和结果
import joblib
import datetime

# 创建结果字典
results = {
    'model': best_svm,
    'scaler': scaler,
    'features': features,
    'best_kernel': best_kernel,
    'best_params': svm_grid.best_params_,
    'test_accuracy': test_accuracy,
    'roc_auc': roc_auc_test,
    'pr_auc': pr_auc_test,
    'cv_mean_score': cv_scores.mean(),
    'timestamp': datetime.datetime.now()
}

# 保存模型
joblib.dump(results, 'svm_cardio_model.pkl')
print("\n模型已保存为 'svm_cardio_model.pkl'")