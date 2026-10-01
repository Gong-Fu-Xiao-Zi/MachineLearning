import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, \
    precision_recall_curve
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l1_l2
from sklearn.inspection import permutation_importance
import warnings
import time

warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
tf.random.set_seed(42)
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 数据加载和预处理
print("=== 心血管疾病预测分析 - BP神经网络 ===")

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

# 创建新的特征
df['bp_ratio'] = df['ap_hi'] / df['ap_lo']  # 血压比
df['age_bmi'] = df['age'] * df['bmi']  # 年龄和BMI的交互项

# 处理血压异常值
df = df[(df['ap_hi'] >= 50) & (df['ap_hi'] <= 250)]
df = df[(df['ap_lo'] >= 30) & (df['ap_lo'] <= 150)]
df = df[(df['bp_ratio'] >= 0.5) & (df['bp_ratio'] <= 3)]

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

# BMI分布
plt.subplot(3, 3, 3)
plt.hist(df['bmi'], bins=30, alpha=0.7, color='orange')
plt.title('BMI分布')
plt.xlabel('BMI')
plt.ylabel('频数')

# 血压比分布
plt.subplot(3, 3, 4)
plt.hist(df['bp_ratio'], bins=30, alpha=0.7, color='red')
plt.title('血压比分布')
plt.xlabel('血压比 (ap_hi/ap_lo)')
plt.ylabel('频数')

# 年龄与心血管疾病的关系
plt.subplot(3, 3, 5)
sns.boxplot(x='cardio', y='age', data=df, palette=['skyblue', 'lightcoral'])
plt.title('年龄与心血管疾病的关系')
plt.xlabel('是否有心血管疾病')
plt.ylabel('年龄 (年)')

# BMI与心血管疾病的关系
plt.subplot(3, 3, 6)
sns.boxplot(x='cardio', y='bmi', data=df, palette=['skyblue', 'lightcoral'])
plt.title('BMI与心血管疾病的关系')
plt.xlabel('是否有心血管疾病')
plt.ylabel('BMI')

plt.tight_layout()
plt.show()

# 3. 特征工程和数据划分
print("\n=== 特征工程和数据划分 ===")

# 选择特征
features = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bmi', 'bp_ratio', 'age_bmi']
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

# 数据标准化 (神经网络对特征尺度敏感)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

print(f"标准化后训练集范围: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

# 处理不平衡数据 - 使用SMOTE过采样
print("\n应用SMOTE过采样...")
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

print(f"过采样后训练集形状: {X_train_resampled.shape}")
print(f"过采样后目标变量分布: {pd.Series(y_train_resampled).value_counts()}")

# 4. 构建BP神经网络模型
print("\n=== 构建BP神经网络模型 ===")


def create_model(hidden_layers=2, units=64, dropout_rate=0.3, learning_rate=0.001, l1_reg=0.01, l2_reg=0.01):
    """创建BP神经网络模型"""
    model = Sequential()

    # 输入层
    model.add(Dense(units, activation='relu', input_shape=(X_train_resampled.shape[1],),
                    kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
    model.add(BatchNormalization())
    model.add(Dropout(dropout_rate))

    # 隐藏层
    for i in range(hidden_layers - 1):
        model.add(Dense(units // (2 ** (i + 1)), activation='relu',
                        kernel_regularizer=l1_l2(l1=l1_reg, l2=l2_reg)))
        model.add(BatchNormalization())
        model.add(Dropout(dropout_rate))

    # 输出层
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer,
                  loss='binary_crossentropy',
                  metrics=['accuracy',
                           tf.keras.metrics.Precision(name='precision'),
                           tf.keras.metrics.Recall(name='recall'),
                           tf.keras.metrics.AUC(name='auc')])

    return model


# 5. 不同网络结构比较
print("\n=== 不同网络结构比较 ===")

# 测试不同的网络结构
network_structures = [
    {'hidden_layers': 1, 'units': 32, 'name': '浅层网络(32)'},
    {'hidden_layers': 2, 'units': 64, 'name': '中等网络(64-32)'},
    {'hidden_layers': 3, 'units': 128, 'name': '深层网络(128-64-32)'},
    {'hidden_layers': 4, 'units': 256, 'name': '深度网络(256-128-64-32)'}
]

history_dict = {}
val_accuracies = []

# 回调函数
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7)

for structure in network_structures:
    print(f"\n训练 {structure['name']}...")

    model = create_model(hidden_layers=structure['hidden_layers'],
                         units=structure['units'])

    history = model.fit(
        X_train_resampled, y_train_resampled,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    history_dict[structure['name']] = history
    val_accuracy = max(history.history['val_accuracy'])
    val_accuracies.append(val_accuracy)

    print(f"{structure['name']} - 最佳验证准确率: {val_accuracy:.4f}")

# 可视化不同网络结构的性能
plt.figure(figsize=(12, 10))

# 准确率比较
plt.subplot(2, 2, 1)
plt.bar([s['name'] for s in network_structures], val_accuracies, color=['blue', 'green', 'red', 'purple'])
plt.title('不同网络结构在验证集上的性能')
plt.ylabel('最佳验证准确率')
plt.xticks(rotation=45)
plt.ylim(0, 1)
for i, v in enumerate(val_accuracies):
    plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

# 训练过程可视化
colors = ['blue', 'green', 'red', 'purple']
for i, (name, history) in enumerate(history_dict.items()):
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], color=colors[i], alpha=0.7, label=f'{name}训练')
    plt.plot(history.history['val_accuracy'], color=colors[i], linestyle='--', alpha=0.7, label=f'{name}验证')

    plt.subplot(2, 2, 3)
    plt.plot(history.history['loss'], color=colors[i], alpha=0.7, label=f'{name}训练')
    plt.plot(history.history['val_loss'], color=colors[i], linestyle='--', alpha=0.7, label=f'{name}验证')

plt.subplot(2, 2, 2)
plt.title('不同网络结构的准确率曲线')
plt.ylabel('准确率')
plt.xlabel('轮次')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.title('不同网络结构的损失曲线')
plt.ylabel('损失')
plt.xlabel('轮次')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 选择最佳网络结构
best_structure_idx = np.argmax(val_accuracies)
best_structure = network_structures[best_structure_idx]
print(f"最佳网络结构: {best_structure['name']}")

# 6. 参数调优
print("\n=== 参数调优 ===")

# 使用最佳网络结构进行参数调优
param_combinations = [
    {'units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.001},
    {'units': 128, 'dropout_rate': 0.3, 'learning_rate': 0.001},
    {'units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.0005},
    {'units': 128, 'dropout_rate': 0.4, 'learning_rate': 0.0005}
]

best_val_accuracy = 0
best_params = None
best_model = None

for params in param_combinations:
    print(f"\n测试参数: {params}")

    model = create_model(
        hidden_layers=best_structure['hidden_layers'],
        units=params['units'],
        dropout_rate=params['dropout_rate'],
        learning_rate=params['learning_rate']
    )

    history = model.fit(
        X_train_resampled, y_train_resampled,
        epochs=100,
        batch_size=32,
        validation_data=(X_val_scaled, y_val),
        callbacks=[early_stopping, reduce_lr],
        verbose=0
    )

    val_accuracy = max(history.history['val_accuracy'])
    print(f"验证准确率: {val_accuracy:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        best_params = params
        best_model = model

print(f"\n最佳参数: {best_params}")
print(f"最佳验证准确率: {best_val_accuracy:.4f}")

# 7. 使用最佳模型进行最终训练
print("\n=== 最终模型训练 ===")

# 使用最佳参数创建最终模型
final_model = create_model(
    hidden_layers=best_structure['hidden_layers'],
    units=best_params['units'],
    dropout_rate=best_params['dropout_rate'],
    learning_rate=best_params['learning_rate']
)

# 模型检查点
model_checkpoint = ModelCheckpoint('best_bp_model.h5',
                                   monitor='val_accuracy',
                                   save_best_only=True,
                                   mode='max',
                                   verbose=1)

# 训练最终模型
final_history = final_model.fit(
    X_train_resampled, y_train_resampled,
    epochs=150,
    batch_size=32,
    validation_data=(X_val_scaled, y_val),
    callbacks=[early_stopping, reduce_lr, model_checkpoint],
    verbose=1
)

# 8. 模型评估
print("\n=== 模型评估 ===")

# 加载最佳模型
final_model.load_weights('best_bp_model.h5')

# 在验证集上评估
y_val_pred_proba = final_model.predict(X_val_scaled)
y_val_pred = (y_val_pred_proba > 0.5).astype(int)
val_accuracy = accuracy_score(y_val, y_val_pred)

# 在测试集上评估
y_test_pred_proba = final_model.predict(X_test_scaled)
y_test_pred = (y_test_pred_proba > 0.5).astype(int)
test_accuracy = accuracy_score(y_test, y_test_pred)

print(f"BP神经网络验证集准确率: {val_accuracy:.4f}")
print(f"BP神经网络测试集准确率: {test_accuracy:.4f}")

# 9. 结果可视化
print("\n=== 结果可视化 ===")

# 训练历史
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(final_history.history['accuracy'], label='训练准确率')
plt.plot(final_history.history['val_accuracy'], label='验证准确率')
plt.title('模型准确率')
plt.ylabel('准确率')
plt.xlabel('轮次')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(final_history.history['loss'], label='训练损失')
plt.plot(final_history.history['val_loss'], label='验证损失')
plt.title('模型损失')
plt.ylabel('损失')
plt.xlabel('轮次')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(final_history.history['precision'], label='训练精确率')
plt.plot(final_history.history['val_precision'], label='验证精确率')
plt.plot(final_history.history['recall'], label='训练召回率')
plt.plot(final_history.history['val_recall'], label='验证召回率')
plt.title('精确率和召回率')
plt.ylabel('分数')
plt.xlabel('轮次')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 混淆矩阵
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# 验证集混淆矩阵
cm_val = confusion_matrix(y_val, y_val_pred)
sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[0])
axes[0].set_title(f'BP神经网络验证集混淆矩阵\n准确率: {val_accuracy:.4f}')
axes[0].set_xlabel('预测标签')
axes[0].set_ylabel('真实标签')

# 测试集混淆矩阵
cm_test = confusion_matrix(y_test, y_test_pred)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=axes[1])
axes[1].set_title(f'BP神经网络测试集混淆矩阵\n准确率: {test_accuracy:.4f}')
axes[1].set_xlabel('预测标签')
axes[1].set_ylabel('真实标签')

plt.tight_layout()
plt.show()

# 分类报告
print("\nBP神经网络验证集分类报告:")
print(classification_report(y_val, y_val_pred))

print("\nBP神经网络测试集分类报告:")
print(classification_report(y_test, y_test_pred))

# 10. ROC曲线和PR曲线
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
axes[0].set_title('BP神经网络 ROC曲线')
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
axes[1].set_title('BP神经网络 PR曲线')
axes[1].legend(loc="lower left")
axes[1].grid(True)

plt.tight_layout()
plt.show()

print(f"验证集ROC AUC: {roc_auc_val:.4f}")
print(f"测试集ROC AUC: {roc_auc_test:.4f}")
print(f"验证集PR AUC: {pr_auc_val:.4f}")
print(f"测试集PR AUC: {pr_auc_test:.4f}")

# 11. 特征重要性分析
print("\n=== 特征重要性分析 ===")


# 使用排列重要性分析特征重要性
def nn_predict(X):
    return final_model.predict(X).flatten()


perm_importance = permutation_importance(
    nn_predict, X_val_scaled, y_val,
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

feature_importance_df = pd.DataFrame({
    'feature': features,
    'importance': perm_importance.importances_mean,
    'std': perm_importance.importances_std
}).sort_values('importance', ascending=False)

plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['feature'], feature_importance_df['importance'],
         xerr=feature_importance_df['std'], color='lightcoral')
plt.title('BP神经网络特征重要性（排列重要性）')
plt.xlabel('排列重要性')
plt.tight_layout()
plt.show()

print("特征重要性排序:")
print(feature_importance_df)

# 12. 模型架构可视化
print("\n=== 模型架构 ===")
final_model.summary()

# 13. 预测概率分布
print("\n=== 预测概率分布 ===")

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(y_val_pred_proba[y_val == 0], bins=30, alpha=0.7, label='无疾病', color='skyblue')
plt.hist(y_val_pred_proba[y_val == 1], bins=30, alpha=0.7, label='有疾病', color='lightcoral')
plt.title('验证集预测概率分布')
plt.xlabel('预测概率')
plt.ylabel('频数')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(y_test_pred_proba[y_test == 0], bins=30, alpha=0.7, label='无疾病', color='skyblue')
plt.hist(y_test_pred_proba[y_test == 1], bins=30, alpha=0.7, label='有疾病', color='lightcoral')
plt.title('测试集预测概率分布')
plt.xlabel('预测概率')
plt.ylabel('频数')
plt.legend()

plt.tight_layout()
plt.show()

# 14. 总结
print("\n=== 实验总结 ===")
print("1. 数据预处理: 完成了年龄转换、BMI计算、新特征创建和异常值处理")
print("2. 特征工程: 创建了血压比和年龄-BMI交互项等新特征")
print("3. 特征标准化: 对特征进行了标准化处理")
print("4. 不平衡处理: 使用SMOTE成功处理了类别不平衡问题")
print("5. 网络结构比较: 比较了不同深度和宽度的神经网络")
print(f"6. 参数调优: 优化了神经元数量、dropout率和学习率")
print(f"7. 最佳网络结构: {best_structure['name']}")
print(f"8. 最佳参数: {best_params}")
print("9. 正则化技术: 使用了L1/L2正则化、Dropout和BatchNormalization")
print("10. 早停法: 使用早停防止过拟合")

print(f"\n最终结果:")
print(f"测试集准确率: {test_accuracy:.4f}")
print(f"测试集ROC AUC: {roc_auc_test:.4f}")
print(f"测试集PR AUC: {pr_auc_test:.4f}")
print(f"最佳验证准确率: {best_val_accuracy:.4f}")

print(f"\n最重要的5个特征:")
for i, row in feature_importance_df.head().iterrows():
    print(f"  {row['feature']}: {row['importance']:.4f}")

# 保存模型和预处理对象
import joblib
import datetime

# 创建结果字典
results = {
    'model': final_model,
    'scaler': scaler,
    'features': features,
    'best_structure': best_structure,
    'best_params': best_params,
    'test_accuracy': test_accuracy,
    'roc_auc': roc_auc_test,
    'pr_auc': pr_auc_test,
    'feature_importance': feature_importance_df,
    'timestamp': datetime.datetime.now()
}

# 保存模型
final_model.save('bp_neural_network_cardio.h5')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(results, 'bp_nn_results.pkl')

print("\n模型已保存为 'bp_neural_network_cardio.h5'")
print("预处理对象已保存为 'scaler.pkl'")
print("结果字典已保存为 'bp_nn_results.pkl'")