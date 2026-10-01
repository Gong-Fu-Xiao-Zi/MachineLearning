import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc, \
    precision_recall_curve, average_precision_score
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.inspection import permutation_importance
import warnings
import time
import json
from pathlib import Path

warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
torch.manual_seed(42)
np.random.seed(42)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 检查GPU可用性
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


class CardiovascularPredictor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = []
        self.results = {}

    def load_and_preprocess_data(self, file_path):
        """加载并预处理数据"""
        print("=== 数据加载和预处理 ===")

        # 加载数据
        df = pd.read_csv(file_path)

        print(f"数据集形状: {df.shape}")
        print("\n数据前5行:")
        print(df.head())

        print("\n目标变量分布:")
        target_dist = df['cardio'].value_counts()
        print(target_dist)

        # 数据预处理
        # 将年龄从天数转换为年数
        df['age'] = df['age'] / 365.25

        # 计算BMI
        df['bmi'] = df['weight'] / (df['height'] / 100) ** 2

        # 创建新的特征
        df['bp_ratio'] = df['ap_hi'] / df['ap_lo']  # 血压比
        df['age_bmi'] = df['age'] * df['bmi']  # 年龄和BMI的交互项
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']  # 脉压
        df['map'] = (df['ap_hi'] + 2 * df['ap_lo']) / 3  # 平均动脉压

        # 处理血压异常值
        df = df[(df['ap_hi'] >= 50) & (df['ap_hi'] <= 250)]
        df = df[(df['ap_lo'] >= 30) & (df['ap_lo'] <= 150)]
        df = df[(df['bp_ratio'] >= 0.5) & (df['bp_ratio'] <= 3)]
        df = df[(df['bmi'] >= 15) & (df['bmi'] <= 50)]

        print(f"清洗后数据形状: {df.shape}")

        return df

    def visualize_data_distribution(self, df):
        """可视化数据分布"""
        print("\n=== 数据分布可视化 ===")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('心血管疾病数据分布', fontsize=16, fontweight='bold')

        # 目标变量分布
        target_counts = df['cardio'].value_counts()
        axes[0, 0].pie(target_counts.values, labels=['无心血管疾病', '有心血管疾病'],
                       autopct='%1.1f%%', colors=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('目标变量分布')

        # 年龄分布
        axes[0, 1].hist(df['age'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('年龄分布')
        axes[0, 1].set_xlabel('年龄(年)')
        axes[0, 1].set_ylabel('频数')

        # 性别分布
        gender_counts = df['gender'].value_counts()
        axes[0, 2].bar(gender_counts.index, gender_counts.values,
                       color=['lightblue', 'lightpink'])
        axes[0, 2].set_title('性别分布')
        axes[0, 2].set_xlabel('性别')
        axes[0, 2].set_ylabel('频数')

        # BMI分布
        axes[1, 0].hist(df['bmi'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('BMI分布')
        axes[1, 0].set_xlabel('BMI')
        axes[1, 0].set_ylabel('频数')

        # 收缩压分布
        axes[1, 1].hist(df['ap_hi'], bins=30, color='salmon', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('收缩压分布')
        axes[1, 1].set_xlabel('收缩压')
        axes[1, 1].set_ylabel('频数')

        # 胆固醇水平分布
        cholesterol_counts = df['cholesterol'].value_counts().sort_index()
        axes[1, 2].bar(cholesterol_counts.index, cholesterol_counts.values,
                       color=['lightyellow', 'orange', 'red'])
        axes[1, 2].set_title('胆固醇水平分布')
        axes[1, 2].set_xlabel('胆固醇水平')
        axes[1, 2].set_ylabel('频数')

        plt.tight_layout()
        plt.show()

        # 特征与目标变量关系
        self._visualize_feature_target_relationship(df)

    def _visualize_feature_target_relationship(self, df):
        """可视化特征与目标变量的关系"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('特征与心血管疾病的关系', fontsize=16, fontweight='bold')

        # 年龄 vs 心血管疾病
        sns.boxplot(x='cardio', y='age', data=df, ax=axes[0, 0],
                    palette=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('年龄与心血管疾病')
        axes[0, 0].set_xlabel('心血管疾病')
        axes[0, 0].set_ylabel('年龄(年)')

        # BMI vs 心血管疾病
        sns.boxplot(x='cardio', y='bmi', data=df, ax=axes[0, 1],
                    palette=['lightblue', 'lightcoral'])
        axes[0, 1].set_title('BMI与心血管疾病')
        axes[0, 1].set_xlabel('心血管疾病')
        axes[0, 1].set_ylabel('BMI')

        # 收缩压 vs 心血管疾病
        sns.boxplot(x='cardio', y='ap_hi', data=df, ax=axes[0, 2],
                    palette=['lightblue', 'lightcoral'])
        axes[0, 2].set_title('收缩压与心血管疾病')
        axes[0, 2].set_xlabel('心血管疾病')
        axes[0, 2].set_ylabel('收缩压')

        # 胆固醇水平 vs 心血管疾病患病率
        cholesterol_means = df.groupby('cholesterol')['cardio'].mean()
        axes[1, 0].bar(cholesterol_means.index, cholesterol_means.values,
                       color='purple', alpha=0.7)
        axes[1, 0].set_title('胆固醇水平与心血管疾病患病率')
        axes[1, 0].set_xlabel('胆固醇水平')
        axes[1, 0].set_ylabel('心血管疾病患病率')

        # 血糖水平 vs 心血管疾病患病率
        gluc_means = df.groupby('gluc')['cardio'].mean()
        axes[1, 1].bar(gluc_means.index, gluc_means.values,
                       color='orange', alpha=0.7)
        axes[1, 1].set_title('血糖水平与心血管疾病患病率')
        axes[1, 1].set_xlabel('血糖水平')
        axes[1, 1].set_ylabel('心血管疾病患病率')

        # 吸烟 vs 心血管疾病患病率
        smoke_means = df.groupby('smoke')['cardio'].mean()
        axes[1, 2].bar(smoke_means.index, smoke_means.values,
                       color='red', alpha=0.7)
        axes[1, 2].set_title('吸烟与心血管疾病患病率')
        axes[1, 2].set_xlabel('吸烟')
        axes[1, 2].set_ylabel('心血管疾病患病率')

        plt.tight_layout()
        plt.show()

    def prepare_features(self, df):
        """特征工程和数据准备"""
        print("\n=== 特征工程和数据划分 ===")

        # 选择特征
        self.feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
                              'cholesterol', 'gluc', 'smoke', 'alco', 'active',
                              'bmi', 'bp_ratio', 'age_bmi', 'pulse_pressure', 'map']

        X = df[self.feature_names]
        y = df['cardio']

        print(f"特征形状: {X.shape}")
        print(f"目标变量形状: {y.shape}")

        # 划分数据集：训练集60%，验证集20%，测试集20%
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

        print(f"训练集大小: {X_train.shape}")
        print(f"验证集大小: {X_val.shape}")
        print(f"测试集大小: {X_test.shape}")

        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)

        print(f"标准化后训练集范围: [{X_train_scaled.min():.2f}, {X_train_scaled.max():.2f}]")

        # 处理不平衡数据 - 使用SMOTE过采样
        print("\n应用SMOTE过采样...")
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)

        print(f"过采样前训练集分布: {pd.Series(y_train).value_counts().to_dict()}")
        print(f"过采样后训练集分布: {pd.Series(y_train_resampled).value_counts().to_dict()}")

        return (X_train_resampled, y_train_resampled,
                X_val_scaled, y_val,
                X_test_scaled, y_test)

    def create_data_loaders(self, X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
        """创建PyTorch数据加载器"""
        print("\n=== 创建PyTorch数据加载器 ===")

        # 转换为PyTorch张量
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        y_train_tensor = torch.FloatTensor(y_train.values).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        y_val_tensor = torch.FloatTensor(y_val.values).to(device)
        X_test_tensor = torch.FloatTensor(X_test).to(device)
        y_test_tensor = torch.FloatTensor(y_test.values).to(device)

        # 创建数据集
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # 创建数据加载器
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f"训练批次数量: {len(train_loader)}")
        print(f"验证批次数量: {len(val_loader)}")
        print(f"测试批次数量: {len(test_loader)}")

        return train_loader, val_loader, test_loader


class BPNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_layers=2, units=64, dropout_rate=0.3):
        super(BPNeuralNetwork, self).__init__()

        self.hidden_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # 输入层到第一个隐藏层
        self.hidden_layers.append(nn.Linear(input_size, units))
        self.batch_norms.append(nn.BatchNorm1d(units))
        self.dropouts.append(nn.Dropout(dropout_rate))

        # 添加隐藏层
        current_units = units
        for i in range(hidden_layers - 1):
            next_units = max(current_units // 2, 8)  # 确保至少8个神经元
            self.hidden_layers.append(nn.Linear(current_units, next_units))
            self.batch_norms.append(nn.BatchNorm1d(next_units))
            self.dropouts.append(nn.Dropout(dropout_rate))
            current_units = next_units

        # 输出层
        self.output_layer = nn.Linear(current_units, 1)

        # 激活函数
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i, (layer, bn, dropout) in enumerate(zip(self.hidden_layers, self.batch_norms, self.dropouts)):
            x = layer(x)
            if x.size(0) > 1:  # BatchNorm需要至少2个样本
                x = bn(x)
            x = self.relu(x)
            x = dropout(x)

        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x


def train_model(model, train_loader, val_loader, epochs=100, learning_rate=0.001, patience=10):
    """训练模型"""
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, min_lr=1e-7)

    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    learning_rates = []

    best_val_accuracy = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # 训练阶段
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs.squeeze(), batch_y)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs.squeeze() > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()

        train_accuracy = train_correct / train_total
        train_losses.append(train_loss / len(train_loader))
        train_accuracies.append(train_accuracy)

        # 验证阶段
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                val_loss += loss.item()

                predicted = (outputs.squeeze() > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

        val_accuracy = val_correct / val_total
        val_losses.append(val_loss / len(val_loader))
        val_accuracies.append(val_accuracy)
        learning_rates.append(optimizer.param_groups[0]['lr'])

        # 学习率调度
        scheduler.step(val_loss)

        # 早停法
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            print(f'Epoch [{epoch + 1}/{epochs}], '
                  f'Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_losses[-1]:.4f}, '
                  f'Train Acc: {train_accuracy:.4f}, '
                  f'Val Acc: {val_accuracy:.4f}, '
                  f'LR: {learning_rates[-1]:.6f}')

        if patience_counter >= patience:
            print(f"早停于第 {epoch + 1} 轮")
            break

    # 加载最佳模型
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'learning_rates': learning_rates,
        'best_val_accuracy': best_val_accuracy
    }


def evaluate_model(model, data_loader):
    """评估模型"""
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []

    with torch.no_grad():
        for batch_X, batch_y in data_loader:
            outputs = model(batch_X)
            probabilities = outputs.squeeze().cpu().numpy()
            predictions = (probabilities > 0.5).astype(int)

            all_probabilities.extend(probabilities)
            all_predictions.extend(predictions)
            all_labels.extend(batch_y.cpu().numpy())

    return np.array(all_predictions), np.array(all_probabilities), np.array(all_labels)


def visualize_training_history(history, model_name):
    """可视化训练历史"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{model_name} - 训练过程', fontsize=16, fontweight='bold')

    # 损失曲线
    axes[0, 0].plot(history['train_losses'], label='训练损失', color='blue')
    axes[0, 0].plot(history['val_losses'], label='验证损失', color='red')
    axes[0, 0].set_title('损失曲线')
    axes[0, 0].set_xlabel('轮次')
    axes[0, 0].set_ylabel('损失')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 准确率曲线
    axes[0, 1].plot(history['train_accuracies'], label='训练准确率', color='blue')
    axes[0, 1].plot(history['val_accuracies'], label='验证准确率', color='red')
    axes[0, 1].set_title('准确率曲线')
    axes[0, 1].set_xlabel('轮次')
    axes[0, 1].set_ylabel('准确率')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 学习率变化
    axes[1, 0].plot(history['learning_rates'], color='green')
    axes[1, 0].set_title('学习率变化')
    axes[1, 0].set_xlabel('轮次')
    axes[1, 0].set_ylabel('学习率')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_yscale('log')

    # 最佳验证准确率
    axes[1, 1].bar(['最佳验证准确率'], [history['best_val_accuracy']], color='orange')
    axes[1, 1].set_title(f'最佳验证准确率: {history["best_val_accuracy"]:.4f}')
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].text(0, history['best_val_accuracy'] + 0.01,
                    f'{history["best_val_accuracy"]:.4f}', ha='center', va='bottom')

    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("=== 心血管疾病预测 - PyTorch BP神经网络 ===")

    # 初始化预测器
    predictor = CardiovascularPredictor()

    # 1. 加载和预处理数据
    df = predictor.load_and_preprocess_data('5_心血管疾病预测.csv')

    # 2. 数据可视化
    predictor.visualize_data_distribution(df)

    # 3. 特征工程和数据准备
    X_train, y_train, X_val, y_val, X_test, y_test = predictor.prepare_features(df)

    # 4. 创建数据加载器
    train_loader, val_loader, test_loader = predictor.create_data_loaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32)

    # 5. 网络结构比较
    print("\n=== 网络结构比较 ===")

    network_structures = [
        {'hidden_layers': 1, 'units': 32, 'name': '浅层网络(32)'},
        {'hidden_layers': 2, 'units': 64, 'name': '中等网络(64-32)'},
        {'hidden_layers': 3, 'units': 128, 'name': '深层网络(128-64-32)'},
        {'hidden_layers': 4, 'units': 256, 'name': '深度网络(256-128-64-32)'}
    ]

    input_size = X_train.shape[1]
    history_dict = {}
    val_accuracies = []

    for structure in network_structures:
        print(f"\n训练 {structure['name']}...")

        model = BPNeuralNetwork(
            input_size=input_size,
            hidden_layers=structure['hidden_layers'],
            units=structure['units']
        ).to(device)

        history = train_model(model, train_loader, val_loader, epochs=100)
        history_dict[structure['name']] = history
        val_accuracies.append(history['best_val_accuracy'])

        print(f"{structure['name']} - 最佳验证准确率: {history['best_val_accuracy']:.4f}")

    # 可视化网络结构比较
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.bar([s['name'] for s in network_structures], val_accuracies,
            color=['blue', 'green', 'red', 'purple'])
    plt.title('不同网络结构在验证集上的性能')
    plt.ylabel('最佳验证准确率')
    plt.xticks(rotation=45)
    plt.ylim(0, 1)
    for i, v in enumerate(val_accuracies):
        plt.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

    # 训练过程比较
    colors = ['blue', 'green', 'red', 'purple']
    for i, (name, history) in enumerate(history_dict.items()):
        plt.subplot(2, 2, 2)
        plt.plot(history['train_accuracies'], color=colors[i], alpha=0.7, label=f'{name}训练')
        plt.plot(history['val_accuracies'], color=colors[i], linestyle='--', alpha=0.7, label=f'{name}验证')

    plt.subplot(2, 2, 2)
    plt.title('不同网络结构的准确率曲线')
    plt.ylabel('准确率')
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

    param_combinations = [
        {'units': 64, 'dropout_rate': 0.2, 'learning_rate': 0.001},
        {'units': 128, 'dropout_rate': 0.3, 'learning_rate': 0.001},
        {'units': 64, 'dropout_rate': 0.3, 'learning_rate': 0.0005},
        {'units': 128, 'dropout_rate': 0.4, 'learning_rate': 0.0005}
    ]

    best_val_accuracy = 0
    best_params = None
    best_model = None

    start_time = time.time()
    for params in param_combinations:
        print(f"\n测试参数: {params}")

        model = BPNeuralNetwork(
            input_size=input_size,
            hidden_layers=best_structure['hidden_layers'],
            units=params['units'],
            dropout_rate=params['dropout_rate']
        ).to(device)

        history = train_model(model, train_loader, val_loader, epochs=100,
                              learning_rate=params['learning_rate'])
        val_accuracy = history['best_val_accuracy']
        print(f"验证准确率: {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_params = params
            best_model = model

    end_time = time.time()
    print(f"\n最佳参数: {best_params}")
    print(f"最佳验证准确率: {best_val_accuracy:.4f}")
    print(f"参数调优耗时: {end_time - start_time:.2f}秒")

    # 7. 最终模型训练
    print("\n=== 最终模型训练 ===")

    final_model = BPNeuralNetwork(
        input_size=input_size,
        hidden_layers=best_structure['hidden_layers'],
        units=best_params['units'],
        dropout_rate=best_params['dropout_rate']
    ).to(device)

    final_history = train_model(final_model, train_loader, val_loader, epochs=150,
                                learning_rate=best_params['learning_rate'])

    print(f"最终模型训练完成，最佳验证准确率: {final_history['best_val_accuracy']:.4f}")

    # 可视化最终训练过程
    visualize_training_history(final_history, "最终模型")

    # 8. 模型评估
    print("\n=== 模型评估 ===")

    # 在验证集和测试集上评估
    y_val_pred, y_val_pred_proba, y_val_true = evaluate_model(final_model, val_loader)
    y_test_pred, y_test_pred_proba, y_test_true = evaluate_model(final_model, test_loader)

    val_accuracy = accuracy_score(y_val_true, y_val_pred)
    test_accuracy = accuracy_score(y_test_true, y_test_pred)

    print(f"验证集准确率: {val_accuracy:.4f}")
    print(f"测试集准确率: {test_accuracy:.4f}")

    # 混淆矩阵
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    cm_val = confusion_matrix(y_val_true, y_val_pred)
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title(f'验证集混淆矩阵\n准确率: {val_accuracy:.4f}')
    axes[0].set_xlabel('预测标签')
    axes[0].set_ylabel('真实标签')

    cm_test = confusion_matrix(y_test_true, y_test_pred)
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title(f'测试集混淆矩阵\n准确率: {test_accuracy:.4f}')
    axes[1].set_xlabel('预测标签')
    axes[1].set_ylabel('真实标签')

    plt.tight_layout()
    plt.show()

    # 分类报告
    print("\n验证集分类报告:")
    print(classification_report(y_val_true, y_val_pred))

    print("\n测试集分类报告:")
    print(classification_report(y_test_true, y_test_pred))

    # 9. ROC曲线和PR曲线
    print("\n=== ROC曲线和PR曲线 ===")

    # ROC曲线
    fpr_val, tpr_val, _ = roc_curve(y_val_true, y_val_pred_proba)
    fpr_test, tpr_test, _ = roc_curve(y_test_true, y_test_pred_proba)

    roc_auc_val = auc(fpr_val, tpr_val)
    roc_auc_test = auc(fpr_test, tpr_test)

    # PR曲线
    precision_val, recall_val, _ = precision_recall_curve(y_val_true, y_val_pred_proba)
    precision_test, recall_test, _ = precision_recall_curve(y_test_true, y_test_pred_proba)

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
    axes[0].set_xlabel('假正率')
    axes[0].set_ylabel('真正率')
    axes[0].set_title('ROC曲线')
    axes[0].legend(loc="lower right")
    axes[0].grid(True)

    # PR曲线
    axes[1].plot(recall_val, precision_val, color='blue', lw=2,
                 label=f'验证集 PR (AUC = {pr_auc_val:.4f})')
    axes[1].plot(recall_test, precision_test, color='red', lw=2,
                 label=f'测试集 PR (AUC = {pr_auc_test:.4f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('召回率')
    axes[1].set_ylabel('精确率')
    axes[1].set_title('PR曲线')
    axes[1].legend(loc="lower left")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    print(f"验证集ROC AUC: {roc_auc_val:.4f}")
    print(f"测试集ROC AUC: {roc_auc_test:.4f}")
    print(f"验证集PR AUC: {pr_auc_val:.4f}")
    print(f"测试集PR AUC: {pr_auc_test:.4f}")

    # 10. 特征重要性分析
    print("\n=== 特征重要性分析 ===")

    def nn_predict(X):
        X_tensor = torch.FloatTensor(X).to(device)
        final_model.eval()
        with torch.no_grad():
            predictions = final_model(X_tensor).cpu().numpy()
        return predictions.flatten()

    # 使用验证集计算排列重要性
    perm_importance = permutation_importance(
        nn_predict, X_val, y_val,
        n_repeats=10,
        random_state=42,
        n_jobs=-1
    )

    feature_importance_df = pd.DataFrame({
        'feature': predictor.feature_names,
        'importance': perm_importance.importances_mean,
        'std': perm_importance.importances_std
    }).sort_values('importance', ascending=True)

    plt.figure(figsize=(12, 8))
    plt.barh(feature_importance_df['feature'], feature_importance_df['importance'],
             xerr=feature_importance_df['std'], color='lightcoral')
    plt.title('特征重要性（排列重要性）')
    plt.xlabel('排列重要性')
    plt.tight_layout()
    plt.show()

    print("特征重要性排序:")
    print(feature_importance_df.sort_values('importance', ascending=False))

    # 11. 预测概率分布
    print("\n=== 预测概率分布 ===")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(y_val_pred_proba[y_val_true == 0], bins=30, alpha=0.7, label='无疾病', color='skyblue')
    plt.hist(y_val_pred_proba[y_val_true == 1], bins=30, alpha=0.7, label='有疾病', color='lightcoral')
    plt.title('验证集预测概率分布')
    plt.xlabel('预测概率')
    plt.ylabel('频数')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.hist(y_test_pred_proba[y_test_true == 0], bins=30, alpha=0.7, label='无疾病', color='skyblue')
    plt.hist(y_test_pred_proba[y_test_true == 1], bins=30, alpha=0.7, label='有疾病', color='lightcoral')
    plt.title('测试集预测概率分布')
    plt.xlabel('预测概率')
    plt.ylabel('频数')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # 12. 保存结果
    predictor.model = final_model
    predictor.results = {
        'best_structure': best_structure,
        'best_params': best_params,
        'test_accuracy': test_accuracy,
        'val_accuracy': val_accuracy,
        'roc_auc_test': roc_auc_test,
        'pr_auc_test': pr_auc_test,
        'feature_importance': feature_importance_df.to_dict(),
        'training_time': end_time - start_time
    }

    # 保存模型
    torch.save(final_model.state_dict(), 'bp_neural_network.pth')
    print("\n模型已保存为 'bp_neural_network.pth'")

    # 13. 总结
    print("\n=== 实验总结 ===")
    print("1. 数据预处理: 完成了年龄转换、BMI计算、新特征创建和异常值处理")
    print("2. 特征工程: 创建了血压比、年龄-BMI交互项、脉压和平均动脉压等新特征")
    print("3. 特征标准化: 对特征进行了标准化处理")
    print("4. 不平衡处理: 使用SMOTE成功处理了类别不平衡问题")
    print("5. 网络结构比较: 比较了不同深度和宽度的神经网络")
    print(f"6. 最佳网络结构: {best_structure['name']}")
    print(f"7. 最佳参数: {best_params}")
    print("8. 正则化技术: 使用了BatchNormalization、Dropout和权重衰减")
    print("9. 优化技术: 使用了学习率调度、梯度裁剪和早停法")
    print(f"10. 使用设备: {device}")

    print(f"\n最终结果:")
    print(f"测试集准确率: {test_accuracy:.4f}")
    print(f"测试集ROC AUC: {roc_auc_test:.4f}")
    print(f"测试集PR AUC: {pr_auc_test:.4f}")
    print(f"最佳验证准确率: {best_val_accuracy:.4f}")

    print(f"\n最重要的5个特征:")
    top_features = feature_importance_df.nlargest(5, 'importance')
    for i, row in top_features.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    return predictor


if __name__ == "__main__":
    predictor = main()