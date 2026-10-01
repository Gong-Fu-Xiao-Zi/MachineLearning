import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, precision_recall_curve, auc, \
    roc_curve
from sklearn.datasets import load_diabetes
from imblearn.over_sampling import SMOTE
import warnings

warnings.filterwarnings('ignore')

pd.set_option('display.width', 1000)
pd.set_option('display.max_columns', None)

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class HeartDiseaseAnalysis:
    def __init__(self):
        self.data = None
        self.models = {}
        self.results = {}
        self.X_test = None
        self.y_test = None

    def load_and_preprocess_data(self):
        """加载和预处理心脏病数据"""
        self.data = pd.read_csv('heart_improve.csv')

        # 处理缺失值
        numeric_columns = ['glucose', 'heartRate', 'education', 'cigsPerDay', 'BPMeds', 'totChol', 'BMI']
        for col in numeric_columns:
            if col in self.data.columns:
                self.data[col].fillna(self.data[col].median(), inplace=True)

        print(f"数据形状: {self.data.shape}")
        print(f"目标变量分布:\n{self.data['TenYearCHD'].value_counts()}")
        print(f"阳性样本比例: {self.data['TenYearCHD'].mean():.3f}")

        return self.data

    def visualize_dataset(self):
        """使用示例代码风格可视化数据集"""
        print("生成心脏病数据集可视化...")

        x = self.data.iloc[:, 0:-1]
        y = self.data['TenYearCHD']

        color_map = {0: 'yellow', 1: 'red'}
        color1 = list(map(lambda x: color_map[x], self.data['TenYearCHD']))

        # 数据特征展示
        fig, axs = plt.subplots(4, 4, figsize=(20, 10.8), dpi=70)

        # 第一行
        axs[0, 0].scatter(np.linspace(0, len(x) - 1, len(x)), x['male'], color=color1, s=3)
        axs[0, 0].set_title('male')

        axs[0, 1].scatter(np.linspace(0, len(x) - 1, len(x)), x['age'], color=color1, s=3)
        axs[0, 1].set_title('age')

        axs[0, 2].scatter(np.linspace(0, len(x) - 1, len(x)), x['education'], color=color1, s=6)
        axs[0, 2].set_title('education')

        axs[0, 3].scatter(np.linspace(0, len(x) - 1, len(x)), x['currentSmoker'], color=color1, s=4)
        axs[0, 3].set_title('currentSmoker')

        # 第二行
        axs[1, 0].scatter(np.linspace(0, len(x) - 1, len(x)), x['cigsPerDay'], color=color1, s=4)
        axs[1, 0].set_title('cigsPerDay')

        axs[1, 1].scatter(np.linspace(0, len(x) - 1, len(x)), x['BPMeds'], color=color1, s=5)
        axs[1, 1].set_title('Blood Pressure Medicine服药史')

        axs[1, 2].scatter(np.linspace(0, len(x) - 1, len(x)), x['prevalentStroke'], color=color1, s=5)
        axs[1, 2].set_title('prevalentStroke中风史')

        axs[1, 3].scatter(np.linspace(0, len(x) - 1, len(x)), x['prevalentHyp'], color=color1, s=5)
        axs[1, 3].set_title('prevalentHyp高血压')

        # 第三行
        axs[2, 0].scatter(np.linspace(0, len(x) - 1, len(x)), x['diabetes'], color=color1, s=5)
        axs[2, 0].set_title('diabetes糖尿病')

        axs[2, 1].scatter(np.linspace(0, len(x) - 1, len(x)), x['totChol'], color=color1, s=3)
        axs[2, 1].set_title('totChol总胆固醇')

        axs[2, 2].scatter(np.linspace(0, len(x) - 1, len(x)), x['sysBP'], color=color1, s=3)
        axs[2, 2].set_title('sysBP收缩压')

        axs[2, 3].scatter(np.linspace(0, len(x) - 1, len(x)), x['diaBP'], color=color1, s=3)
        axs[2, 3].set_title('diaBP扩张压')

        # 第四行
        axs[3, 0].scatter(np.linspace(0, len(x) - 1, len(x)), x['BMI'], color=color1, s=3)
        axs[3, 0].set_title('BMI')

        axs[3, 1].scatter(np.linspace(0, len(x) - 1, len(x)), x['heartRate'], color=color1, s=3)
        axs[3, 1].set_title('heartRate心率')

        axs[3, 2].scatter(np.linspace(0, len(x) - 1, len(x)), x['glucose'], color=color1, s=3)
        axs[3, 2].set_title('glucose血糖')

        # 目标变量分布
        target_counts = self.data['TenYearCHD'].value_counts()
        axs[3, 3].pie(target_counts.values, labels=['健康', '患病'], autopct='%1.1f%%',
                      colors=['yellow', 'red'], startangle=90)
        axs[3, 3].set_title('目标变量分布')

        plt.tight_layout()
        plt.show()

        # 添加特征相关性热力图
        self.plot_correlation_heatmap()

    def plot_correlation_heatmap(self):
        """绘制特征相关性热力图"""
        print("生成特征相关性热力图...")

        plt.figure(figsize=(12, 10))

        # 计算相关性矩阵
        corr_matrix = self.data.corr()

        # 创建热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, cbar_kws={"shrink": .8})

        plt.title('心脏病数据集特征相关性热力图', fontsize=16)
        plt.tight_layout()
        plt.show()

    def handle_imbalance(self, method='smote'):
        """处理样本不平衡问题"""
        X = self.data.drop('TenYearCHD', axis=1)
        y = self.data['TenYearCHD']

        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y

        print(f"重采样后数据形状: {X_resampled.shape}")
        print(f"重采样后目标分布:\n{y_resampled.value_counts()}")

        return X_resampled, y_resampled

    def train_models_with_regularization(self, X, y):
        """使用不同正则化训练模型"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)

        self.X_test = X_test
        self.y_test = y_test

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        regularization_types = {
            'L2_C=1.0': {'penalty': 'l2', 'C': 1.0},
            'L2_C=0.1': {'penalty': 'l2', 'C': 0.1},
            'L2_C=10.0': {'penalty': 'l2', 'C': 10.0},
            'L1_C=1.0': {'penalty': 'l1', 'C': 1.0, 'solver': 'liblinear'},
        }

        models = {}
        results = {}

        for name, params in regularization_types.items():
            print(f"\n训练模型: {name}")
            try:
                model = LogisticRegression(**params, random_state=42, class_weight='balanced')
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                accuracy = model.score(X_test_scaled, y_test)
                auc_score = roc_auc_score(y_test, y_pred_proba)

                models[name] = model
                results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'y_test': y_test,
                }

                print(f"{name} - 准确率: {accuracy:.3f}, AUC: {auc_score:.3f}")

            except Exception as e:
                print(f"训练模型 {name} 时出错: {e}")

        self.models = models
        self.results = results
        return models, results

    def visualize_predictions(self):
        """使用示例代码风格可视化预测结果"""
        if not self.results:
            print("没有训练好的模型结果可可视化")
            return

        print("生成预测结果可视化...")

        # 选择最佳模型进行详细可视化
        best_model_name = max(self.results.keys(),
                              key=lambda x: self.results[x]['auc'])
        best_result = self.results[best_model_name]

        y_test = best_result['y_test']
        y_pred = best_result['y_pred']
        y_pred_proba = best_result['y_pred_proba']

        # 创建预测结果可视化 - 使用示例代码风格
        fig, axs = plt.subplots(2, 3, figsize=(18, 12), dpi=100)

        # 1. 真实值和预测值对比
        axs[0, 0].scatter(np.linspace(0, len(y_test) - 1, len(y_test)), y_test,
                          c='orange', s=2, label='real', alpha=0.75)
        axs[0, 0].scatter(np.linspace(0, len(y_pred) - 1, len(y_pred)), y_pred,
                          c='blue', s=2, label='predict', alpha=0.75)
        axs[0, 0].set_title('真实值和预测值对比')
        axs[0, 0].set_xlabel('样本索引')
        axs[0, 0].set_ylabel('类别 (0=健康, 1=患病)')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        # 2. 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=axs[0, 1],
                    xticklabels=['预测健康', '预测患病'],
                    yticklabels=['真实健康', '真实患病'])
        axs[0, 1].set_title('混淆矩阵')

        # 3. ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        axs[0, 2].plot(fpr, tpr, color='darkorange', lw=2,
                       label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        axs[0, 2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[0, 2].set_xlim([0.0, 1.0])
        axs[0, 2].set_ylim([0.0, 1.05])
        axs[0, 2].set_xlabel('假正率')
        axs[0, 2].set_ylabel('真正率')
        axs[0, 2].set_title('ROC曲线')
        axs[0, 2].legend(loc="lower right")
        axs[0, 2].grid(True, alpha=0.3)

        # 4. 预测概率分布
        axs[1, 0].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7,
                       color='yellow', label='健康样本')
        axs[1, 0].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7,
                       color='red', label='患病样本')
        axs[1, 0].set_title('预测概率分布')
        axs[1, 0].set_xlabel('预测概率')
        axs[1, 0].set_ylabel('频数')
        axs[1, 0].legend()

        # 5. 模型性能比较
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        auc_scores = [self.results[name]['auc'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        axs[1, 1].bar(x - width / 2, accuracies, width, label='准确率',
                      alpha=0.7, color='lightblue')
        axs[1, 1].bar(x + width / 2, auc_scores, width, label='AUC',
                      alpha=0.7, color='lightgreen')
        axs[1, 1].set_xlabel('模型')
        axs[1, 1].set_ylabel('分数')
        axs[1, 1].set_title('不同正则化模型性能比较')
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(model_names, rotation=45)
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)

        # 添加数值标签
        for i, v in enumerate(accuracies):
            axs[1, 1].text(i - width / 2, v + 0.01, f'{v:.3f}',
                           ha='center', va='bottom', fontsize=8)
        for i, v in enumerate(auc_scores):
            axs[1, 1].text(i + width / 2, v + 0.01, f'{v:.3f}',
                           ha='center', va='bottom', fontsize=8)

        # 6. 精确率-召回率曲线
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        axs[1, 2].plot(recall, precision, color='green', lw=2,
                       label=f'PR曲线 (AUC = {pr_auc:.3f})')
        axs[1, 2].set_xlim([0.0, 1.0])
        axs[1, 2].set_ylim([0.0, 1.05])
        axs[1, 2].set_xlabel('召回率')
        axs[1, 2].set_ylabel('精确率')
        axs[1, 2].set_title('精确率-召回率曲线')
        axs[1, 2].legend(loc="lower left")
        axs[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印详细分类报告
        print(f"\n最佳模型 ({best_model_name}) 详细分类报告:")
        print(classification_report(y_test, y_pred,
                                    target_names=['健康', '患病']))

        # 打印AUC分数
        print(f"AUC分数: {self.results[best_model_name]['auc']:.3f}")

    def run_analysis(self):
        """运行完整分析"""
        print("开始心脏病数据分析...")

        # 1. 加载数据
        self.load_and_preprocess_data()

        # 2. 数据集可视化
        self.visualize_dataset()

        # 3. 处理样本不平衡
        X_resampled, y_resampled = self.handle_imbalance('smote')

        # 4. 训练不同正则化模型
        self.train_models_with_regularization(X_resampled, y_resampled)

        # 5. 预测结果可视化
        self.visualize_predictions()


class DiabetesAnalysis:
    def __init__(self):
        self.data = None
        self.models = {}
        self.results = {}
        self.X_test = None
        self.y_test = None

    def load_and_preprocess_data(self):
        """加载和预处理糖尿病数据"""
        diabetes = load_diabetes()
        X = diabetes.data
        y = diabetes.target

        # 将回归问题转换为分类问题（根据中位数）
        threshold = np.median(y)
        y_binary = (y > threshold).astype(int)

        self.data = pd.DataFrame(X, columns=diabetes.feature_names)
        self.data['target'] = y_binary

        print(f"糖尿病数据形状: {self.data.shape}")
        print(f"目标变量分布:\n{self.data['target'].value_counts()}")
        print(f"阳性样本比例: {self.data['target'].mean():.3f}")

        return self.data

    def visualize_dataset(self):
        """使用示例代码风格可视化糖尿病数据集"""
        print("生成糖尿病数据集可视化...")

        x = self.data.iloc[:, 0:-1]
        y = self.data['target']

        color_map = {0: 'yellow', 1: 'red'}
        color1 = list(map(lambda x: color_map[x], self.data['target']))

        # 数据特征展示 - 使用3x4布局适应10个特征
        fig, axs = plt.subplots(3, 4, figsize=(20, 10.8), dpi=70)

        features = list(x.columns)

        for i, feature in enumerate(features):
            row, col = i // 4, i % 4
            axs[row, col].scatter(np.linspace(0, len(x) - 1, len(x)), x[feature],
                                  color=color1, s=3)
            axs[row, col].set_title(feature)

        # 目标变量分布
        target_counts = self.data['target'].value_counts()
        axs[2, 3].pie(target_counts.values, labels=['低风险', '高风险'],
                      autopct='%1.1f%%', colors=['yellow', 'red'],
                      startangle=90)
        axs[2, 3].set_title('目标变量分布')

        plt.tight_layout()
        plt.show()

        # 添加特征相关性热力图
        self.plot_correlation_heatmap()

    def plot_correlation_heatmap(self):
        """绘制特征相关性热力图"""
        print("生成糖尿病数据集特征相关性热力图...")

        plt.figure(figsize=(10, 8))

        # 计算相关性矩阵
        corr_matrix = self.data.corr()

        # 创建热力图
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, square=True, cbar_kws={"shrink": .8})

        plt.title('糖尿病数据集特征相关性热力图', fontsize=16)
        plt.tight_layout()
        plt.show()

    def handle_imbalance(self, method='smote'):
        """处理样本不平衡问题"""
        X = self.data.drop('target', axis=1)
        y = self.data['target']

        if method == 'smote':
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
        else:
            X_resampled, y_resampled = X, y

        print(f"重采样后数据形状: {X_resampled.shape}")
        print(f"重采样后目标分布:\n{y_resampled.value_counts()}")

        return X_resampled, y_resampled

    def train_models_with_regularization(self, X, y):
        """使用不同正则化训练模型"""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y)

        self.X_test = X_test
        self.y_test = y_test

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        regularization_types = {
            'L2_C=1.0': {'penalty': 'l2', 'C': 1.0},
            'L2_C=0.1': {'penalty': 'l2', 'C': 0.1},
            'L2_C=10.0': {'penalty': 'l2', 'C': 10.0},
            'L1_C=1.0': {'penalty': 'l1', 'C': 1.0, 'solver': 'liblinear'},
        }

        for name, params in regularization_types.items():
            print(f"\n训练糖尿病模型: {name}")
            try:
                model = LogisticRegression(**params, random_state=42, class_weight='balanced')
                model.fit(X_train_scaled, y_train)

                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                accuracy = model.score(X_test_scaled, y_test)
                auc_score = roc_auc_score(y_test, y_pred_proba)

                self.models[name] = model
                self.results[name] = {
                    'model': model,
                    'accuracy': accuracy,
                    'auc': auc_score,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba,
                    'y_test': y_test,
                }

                print(f"{name} - 准确率: {accuracy:.3f}, AUC: {auc_score:.3f}")

            except Exception as e:
                print(f"训练模型 {name} 时出错: {e}")

        return self.models, self.results

    def visualize_predictions(self):
        """使用示例代码风格可视化预测结果"""
        if not self.results:
            print("没有训练好的模型结果可可视化")
            return

        print("生成糖尿病预测结果可视化...")

        # 选择最佳模型
        best_model_name = max(self.results.keys(),
                              key=lambda x: self.results[x]['auc'])
        best_result = self.results[best_model_name]

        y_test = best_result['y_test']
        y_pred = best_result['y_pred']
        y_pred_proba = best_result['y_pred_proba']

        # 创建预测结果可视化 - 使用示例代码风格
        fig, axs = plt.subplots(2, 3, figsize=(18, 12), dpi=100)

        # 1. 真实值和预测值对比
        axs[0, 0].scatter(np.linspace(0, len(y_test) - 1, len(y_test)), y_test,
                          c='orange', s=2, label='real', alpha=0.75)
        axs[0, 0].scatter(np.linspace(0, len(y_pred) - 1, len(y_pred)), y_pred,
                          c='blue', s=2, label='predict', alpha=0.75)
        axs[0, 0].set_title('真实值和预测值对比')
        axs[0, 0].set_xlabel('样本索引')
        axs[0, 0].set_ylabel('类别 (0=低风险, 1=高风险)')
        axs[0, 0].legend()
        axs[0, 0].grid(True, alpha=0.3)

        # 2. 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlOrRd', ax=axs[0, 1],
                    xticklabels=['预测低风险', '预测高风险'],
                    yticklabels=['真实低风险', '真实高风险'])
        axs[0, 1].set_title('混淆矩阵')

        # 3. ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        axs[0, 2].plot(fpr, tpr, color='darkorange', lw=2,
                       label=f'ROC曲线 (AUC = {roc_auc:.3f})')
        axs[0, 2].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axs[0, 2].set_xlim([0.0, 1.0])
        axs[0, 2].set_ylim([0.0, 1.05])
        axs[0, 2].set_xlabel('假正率')
        axs[0, 2].set_ylabel('真正率')
        axs[0, 2].set_title('ROC曲线')
        axs[0, 2].legend(loc="lower right")
        axs[0, 2].grid(True, alpha=0.3)

        # 4. 预测概率分布
        axs[1, 0].hist(y_pred_proba[y_test == 0], bins=20, alpha=0.7,
                       color='yellow', label='低风险样本')
        axs[1, 0].hist(y_pred_proba[y_test == 1], bins=20, alpha=0.7,
                       color='red', label='高风险样本')
        axs[1, 0].set_title('预测概率分布')
        axs[1, 0].set_xlabel('预测概率')
        axs[1, 0].set_ylabel('频数')
        axs[1, 0].legend()

        # 5. 模型性能比较
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        auc_scores = [self.results[name]['auc'] for name in model_names]

        x = np.arange(len(model_names))
        width = 0.35

        axs[1, 1].bar(x - width / 2, accuracies, width, label='准确率',
                      alpha=0.7, color='lightblue')
        axs[1, 1].bar(x + width / 2, auc_scores, width, label='AUC',
                      alpha=0.7, color='lightgreen')
        axs[1, 1].set_xlabel('模型')
        axs[1, 1].set_ylabel('分数')
        axs[1, 1].set_title('不同正则化模型性能比较')
        axs[1, 1].set_xticks(x)
        axs[1, 1].set_xticklabels(model_names, rotation=45)
        axs[1, 1].legend()
        axs[1, 1].grid(True, alpha=0.3)

        # 添加数值标签
        for i, v in enumerate(accuracies):
            axs[1, 1].text(i - width / 2, v + 0.01, f'{v:.3f}',
                           ha='center', va='bottom', fontsize=8)
        for i, v in enumerate(auc_scores):
            axs[1, 1].text(i + width / 2, v + 0.01, f'{v:.3f}',
                           ha='center', va='bottom', fontsize=8)

        # 6. 精确率-召回率曲线
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = auc(recall, precision)

        axs[1, 2].plot(recall, precision, color='green', lw=2,
                       label=f'PR曲线 (AUC = {pr_auc:.3f})')
        axs[1, 2].set_xlim([0.0, 1.0])
        axs[1, 2].set_ylim([0.0, 1.05])
        axs[1, 2].set_xlabel('召回率')
        axs[1, 2].set_ylabel('精确率')
        axs[1, 2].set_title('精确率-召回率曲线')
        axs[1, 2].legend(loc="lower left")
        axs[1, 2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        # 打印详细分类报告
        print(f"\n最佳模型 ({best_model_name}) 详细分类报告:")
        print(classification_report(y_test, y_pred,
                                    target_names=['低风险', '高风险']))

        # 打印AUC分数
        print(f"AUC分数: {self.results[best_model_name]['auc']:.3f}")

    def run_analysis(self):
        """运行糖尿病分析"""
        print("开始糖尿病数据分析...")

        self.load_and_preprocess_data()
        self.visualize_dataset()
        X_resampled, y_resampled = self.handle_imbalance('smote')
        self.train_models_with_regularization(X_resampled, y_resampled)
        self.visualize_predictions()


def main():
    """主函数 - 交互界面"""
    print("=" * 60)
    print("          机器学习分类分析系统")
    print("=" * 60)
    print("1. 冠心病预测分析")
    print("2. 糖尿病预测分析")
    print("3. 退出")
    print("=" * 60)

    while True:
        try:
            choice = input("\n请选择分析类型 (1/2/3): ").strip()

            if choice == '1':
                print("\n开始冠心病预测分析...")
                heart_analysis = HeartDiseaseAnalysis()
                heart_analysis.run_analysis()

            elif choice == '2':
                print("\n开始糖尿病预测分析...")
                diabetes_analysis = DiabetesAnalysis()
                diabetes_analysis.run_analysis()

            elif choice == '3':
                print("谢谢使用！")
                break

            else:
                print("无效选择，请重新输入！")

            # 询问是否继续
            continue_choice = input("\n是否继续分析？(y/n): ").strip().lower()
            if continue_choice != 'y':
                print("谢谢使用！")
                break

        except KeyboardInterrupt:
            print("\n程序被用户中断")
            break
        except Exception as e:
            print(f"发生错误: {e}")
            continue_choice = input("是否继续？(y/n): ").strip().lower()
            if continue_choice != 'y':
                break


if __name__ == "__main__":
    main()