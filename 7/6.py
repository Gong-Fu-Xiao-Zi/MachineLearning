import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             roc_curve, auc, precision_recall_curve, average_precision_score,
                             roc_auc_score)
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
import warnings
import time
import joblib

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class CardiovascularLRClassifier:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.results = {}
        self.best_params = {}

    def load_and_preprocess_data(self, file_path):
        """加载并预处理数据"""
        print("=== 数据加载和预处理 ===")

        # 加载数据
        df = pd.read_csv(file_path)

        print(f"数据集形状: {df.shape}")
        print("\n数据前5行:")
        print(df.head())

        # 基本统计信息
        print("\n数据基本统计信息:")
        print(df.describe())

        # 检查缺失值
        print(f"\n缺失值统计:")
        print(df.isnull().sum())

        # 目标变量分布
        print("\n目标变量分布:")
        target_dist = df['cardio'].value_counts()
        print(target_dist)
        print(f"阳性比例: {df['cardio'].mean():.3f}")

        return df

    def visualize_data_distribution(self, df):
        """可视化数据分布"""
        print("\n=== 数据分布可视化 ===")

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('心血管疾病数据分布', fontsize=16, fontweight='bold')

        # 目标变量分布
        target_counts = df['cardio'].value_counts()
        axes[0, 0].pie(target_counts.values, labels=['无心血管疾病', '有心血管疾病'],
                       autopct='%1.1f%%', colors=['lightblue', 'lightcoral'],
                       startangle=90)
        axes[0, 0].set_title('目标变量分布')

        # 年龄分布（转换为年）
        df['age_years'] = df['age'] / 365.25
        axes[0, 1].hist(df['age_years'], bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        axes[0, 1].set_title('年龄分布')
        axes[0, 1].set_xlabel('年龄(年)')
        axes[0, 1].set_ylabel('频数')

        # 性别分布
        gender_mapping = {1: '男', 2: '女'}
        gender_counts = df['gender'].value_counts()
        axes[0, 2].bar([gender_mapping[x] for x in gender_counts.index],
                       gender_counts.values, color=['lightblue', 'lightpink'])
        axes[0, 2].set_title('性别分布')
        axes[0, 2].set_ylabel('频数')

        # BMI分布
        df['bmi'] = df['weight'] / (df['height'] / 100) ** 2
        axes[1, 0].hist(df['bmi'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
        axes[1, 0].set_title('BMI分布')
        axes[1, 0].set_xlabel('BMI')
        axes[1, 0].set_ylabel('频数')

        # 收缩压分布
        axes[1, 1].hist(df['ap_hi'], bins=30, color='salmon', edgecolor='black', alpha=0.7)
        axes[1, 1].set_title('收缩压分布')
        axes[1, 1].set_xlabel('收缩压 (mmHg)')
        axes[1, 1].set_ylabel('频数')

        # 胆固醇水平分布
        cholesterol_mapping = {1: '正常', 2: '偏高', 3: '很高'}
        cholesterol_counts = df['cholesterol'].value_counts().sort_index()
        axes[1, 2].bar([cholesterol_mapping[x] for x in cholesterol_counts.index],
                       cholesterol_counts.values, color=['lightgreen', 'orange', 'red'])
        axes[1, 2].set_title('胆固醇水平分布')
        axes[1, 2].set_ylabel('频数')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

        # 特征与目标变量关系
        self._visualize_feature_target_relationship(df)

        return df

    def _visualize_feature_target_relationship(self, df):
        """可视化特征与目标变量的关系"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('特征与心血管疾病的关系', fontsize=16, fontweight='bold')

        # 年龄 vs 心血管疾病
        sns.boxplot(x='cardio', y='age_years', data=df, ax=axes[0, 0],
                    palette=['lightblue', 'lightcoral'])
        axes[0, 0].set_title('年龄与心血管疾病')
        axes[0, 0].set_xlabel('心血管疾病 (0:无, 1:有)')
        axes[0, 0].set_ylabel('年龄(年)')

        # BMI vs 心血管疾病
        sns.boxplot(x='cardio', y='bmi', data=df, ax=axes[0, 1],
                    palette=['lightblue', 'lightcoral'])
        axes[0, 1].set_title('BMI与心血管疾病')
        axes[0, 1].set_xlabel('心血管疾病 (0:无, 1:有)')
        axes[0, 1].set_ylabel('BMI')

        # 收缩压 vs 心血管疾病
        sns.boxplot(x='cardio', y='ap_hi', data=df, ax=axes[0, 2],
                    palette=['lightblue', 'lightcoral'])
        axes[0, 2].set_title('收缩压与心血管疾病')
        axes[0, 2].set_xlabel('心血管疾病 (0:无, 1:有)')
        axes[0, 2].set_ylabel('收缩压 (mmHg)')

        # 胆固醇水平 vs 心血管疾病患病率
        cholesterol_means = df.groupby('cholesterol')['cardio'].mean()
        cholesterol_mapping = {1: '正常', 2: '偏高', 3: '很高'}
        axes[1, 0].bar([cholesterol_mapping[x] for x in cholesterol_means.index],
                       cholesterol_means.values, color='purple', alpha=0.7)
        axes[1, 0].set_title('胆固醇水平与心血管疾病患病率')
        axes[1, 0].set_xlabel('胆固醇水平')
        axes[1, 0].set_ylabel('心血管疾病患病率')
        plt.sca(axes[1, 0])
        plt.xticks(rotation=45)

        # 血糖水平 vs 心血管疾病患病率
        gluc_means = df.groupby('gluc')['cardio'].mean()
        gluc_mapping = {1: '正常', 2: '偏高', 3: '很高'}
        axes[1, 1].bar([gluc_mapping[x] for x in gluc_means.index],
                       gluc_means.values, color='orange', alpha=0.7)
        axes[1, 1].set_title('血糖水平与心血管疾病患病率')
        axes[1, 1].set_xlabel('血糖水平')
        axes[1, 1].set_ylabel('心血管疾病患病率')
        plt.sca(axes[1, 1])
        plt.xticks(rotation=45)

        # 吸烟 vs 心血管疾病患病率
        smoke_means = df.groupby('smoke')['cardio'].mean()
        smoke_mapping = {0: '不吸烟', 1: '吸烟'}
        axes[1, 2].bar([smoke_mapping[x] for x in smoke_means.index],
                       smoke_means.values, color='red', alpha=0.7)
        axes[1, 2].set_title('吸烟与心血管疾病患病率')
        axes[1, 2].set_xlabel('吸烟')
        axes[1, 2].set_ylabel('心血管疾病患病率')

        plt.tight_layout()
        plt.show()

        # 相关性热图
        self._plot_correlation_heatmap(df)

    def _plot_correlation_heatmap(self, df):
        """绘制相关性热图"""
        # 选择数值型特征
        numeric_features = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo',
                            'cholesterol', 'gluc', 'bmi', 'cardio']

        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_features].corr()

        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm',
                    center=0, square=True, linewidths=0.5)
        plt.title('特征相关性热图')
        plt.tight_layout()
        plt.show()

    def feature_engineering(self, df):
        """特征工程"""
        print("\n=== 特征工程 ===")

        # 创建新特征
        df_eng = df.copy()

        # 基础特征
        df_eng['bmi'] = df_eng['weight'] / (df_eng['height'] / 100) ** 2
        df_eng['age_years'] = df_eng['age'] / 365.25

        # 血压相关特征
        df_eng['bp_ratio'] = df_eng['ap_hi'] / df_eng['ap_lo']
        df_eng['pulse_pressure'] = df_eng['ap_hi'] - df_eng['ap_lo']  # 脉压
        df_eng['map'] = (df_eng['ap_hi'] + 2 * df_eng['ap_lo']) / 3  # 平均动脉压

        # 交互特征
        df_eng['age_bmi'] = df_eng['age_years'] * df_eng['bmi']
        df_eng['bp_bmi'] = df_eng['ap_hi'] * df_eng['bmi']

        # 风险评分特征
        df_eng['risk_score'] = (df_eng['cholesterol'] + df_eng['gluc'] +
                                df_eng['smoke'] * 2 + df_eng['alco'] * 1.5)

        # 处理异常值
        df_eng = df_eng[(df_eng['ap_hi'] >= 50) & (df_eng['ap_hi'] <= 250)]
        df_eng = df_eng[(df_eng['ap_lo'] >= 30) & (df_eng['ap_lo'] <= 150)]
        df_eng = df_eng[(df_eng['bp_ratio'] >= 0.5) & (df_eng['bp_ratio'] <= 3)]
        df_eng = df_eng[(df_eng['bmi'] >= 15) & (df_eng['bmi'] <= 50)]

        # 选择最终特征
        self.feature_names = [
            'age_years', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
            'cholesterol', 'gluc', 'smoke', 'alco', 'active',
            'bmi', 'bp_ratio', 'pulse_pressure', 'map', 'age_bmi', 'bp_bmi', 'risk_score'
        ]

        X = df_eng[self.feature_names]
        y = df_eng['cardio']

        print(f"特征工程后数据形状: {X.shape}")
        print(f"使用的特征数量: {len(self.feature_names)}")
        print("特征列表:", self.feature_names)

        return X, y

    def prepare_data(self, X, y):
        """准备训练、验证和测试数据"""
        print("\n=== 数据划分 ===")

        # 划分数据集：训练集60%，验证集20%，测试集20%
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

        print(f"训练集大小: {X_train.shape}")
        print(f"验证集大小: {X_val.shape}")
        print(f"测试集大小: {X_test.shape}")

        print(f"\n训练集目标分布:\n{y_train.value_counts()}")
        print(f"验证集目标分布:\n{y_val.value_counts()}")
        print(f"测试集目标分布:\n{y_test.value_counts()}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def handle_imbalance(self, X_train, y_train):
        """处理数据不平衡"""
        print("\n=== 处理数据不平衡 ===")

        print("过采样前类别分布:")
        print(y_train.value_counts())

        # 使用SMOTE过采样
        smote = SMOTE(random_state=42)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        print("\n过采样后类别分布:")
        print(pd.Series(y_train_resampled).value_counts())

        # 可视化过采样效果
        plt.figure(figsize=(10, 5))

        plt.subplot(1, 2, 1)
        before_counts = y_train.value_counts()
        plt.bar(['无疾病', '有疾病'], before_counts.values,
                color=['lightblue', 'lightcoral'])
        plt.title('过采样前类别分布')
        plt.ylabel('样本数')

        plt.subplot(1, 2, 2)
        after_counts = pd.Series(y_train_resampled).value_counts()
        plt.bar(['无疾病', '有疾病'], after_counts.values,
                color=['lightblue', 'lightcoral'])
        plt.title('过采样后类别分布')
        plt.ylabel('样本数')

        plt.tight_layout()
        plt.show()

        return X_train_resampled, y_train_resampled

    def hyperparameter_tuning(self, X_train, y_train, X_val, y_val):
        """超参数调优"""
        print("\n=== 超参数调优 ===")

        # 数据标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        # 定义参数网格
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga'],
            'max_iter': [1000, 2000]
        }

        # 创建逻辑回归模型
        lr = LogisticRegression(random_state=42)

        # 网格搜索
        print("开始网格搜索...")
        start_time = time.time()

        grid_search = GridSearchCV(
            estimator=lr,
            param_grid=param_grid,
            cv=5,
            scoring='roc_auc',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train_scaled, y_train)

        end_time = time.time()
        print(f"网格搜索完成，耗时: {end_time - start_time:.2f}秒")

        # 输出最佳参数
        print(f"\n最佳参数: {grid_search.best_params_}")
        print(f"最佳交叉验证分数: {grid_search.best_score_:.4f}")

        # 在验证集上评估最佳模型
        best_model = grid_search.best_estimator_
        y_val_pred = best_model.predict(X_val_scaled)
        y_val_prob = best_model.predict_proba(X_val_scaled)[:, 1]

        val_accuracy = accuracy_score(y_val, y_val_pred)
        val_auc = roc_auc_score(y_val, y_val_prob)

        print(f"验证集准确率: {val_accuracy:.4f}")
        print(f"验证集AUC: {val_auc:.4f}")

        self.best_params = grid_search.best_params_

        return best_model

    def train_final_model(self, X_train, y_train, X_val, y_val):
        """训练最终模型"""
        print("\n=== 训练最终模型 ===")

        # 合并训练集和验证集用于最终训练
        X_final_train = pd.concat([X_train, X_val])
        y_final_train = pd.concat([y_train, y_val])

        print(f"最终训练集大小: {X_final_train.shape}")

        # 处理不平衡
        X_final_resampled, y_final_resampled = self.handle_imbalance(X_final_train, y_final_train)

        # 数据标准化
        X_final_scaled = self.scaler.fit_transform(X_final_resampled)

        # 使用最佳参数训练最终模型
        final_model = LogisticRegression(
            C=self.best_params['C'],
            penalty=self.best_params['penalty'],
            solver=self.best_params['solver'],
            max_iter=self.best_params['max_iter'],
            random_state=42
        )

        print("训练最终模型...")
        final_model.fit(X_final_scaled, y_final_resampled)

        print("最终模型训练完成!")
        self.model = final_model

        return final_model

    def evaluate_model(self, model, X_test, y_test):
        """评估模型性能"""
        print("\n=== 模型评估 ===")

        # 数据标准化
        X_test_scaled = self.scaler.transform(X_test)

        # 预测
        y_pred = model.predict(X_test_scaled)
        y_prob = model.predict_proba(X_test_scaled)[:, 1]

        # 计算指标
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_prob)
        ap_score = average_precision_score(y_test, y_prob)

        print(f"测试集准确率: {accuracy:.4f}")
        print(f"测试集AUC: {auc_score:.4f}")
        print(f"测试集平均精确率: {ap_score:.4f}")

        # 分类报告
        print("\n分类报告:")
        print(classification_report(y_test, y_pred))

        # 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)

        # 存储结果
        self.results = {
            'accuracy': accuracy,
            'auc': auc_score,
            'average_precision': ap_score,
            'predictions': y_pred,
            'probabilities': y_prob,
            'confusion_matrix': cm
        }

        return accuracy, auc_score, ap_score, y_pred, y_prob, cm

    def visualize_results(self, X_test, y_test, y_pred, y_prob):
        """可视化结果"""
        print("\n=== 结果可视化 ===")

        # 创建子图
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('逻辑回归模型性能评估', fontsize=16, fontweight='bold')

        # 1. 混淆矩阵
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title(f'混淆矩阵\n准确率: {self.results["accuracy"]:.4f}')
        axes[0, 0].set_xlabel('预测标签')
        axes[0, 0].set_ylabel('真实标签')

        # 2. ROC曲线
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        axes[0, 1].plot(fpr, tpr, color='darkorange', lw=2,
                        label=f'ROC曲线 (AUC = {roc_auc:.4f})')
        axes[0, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('假正率')
        axes[0, 1].set_ylabel('真正率')
        axes[0, 1].set_title('ROC曲线')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True)

        # 3. PR曲线
        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        average_precision = average_precision_score(y_test, y_prob)

        axes[0, 2].plot(recall, precision, color='blue', lw=2,
                        label=f'PR曲线 (AP = {average_precision:.4f})')
        axes[0, 2].set_xlim([0.0, 1.0])
        axes[0, 2].set_ylim([0.0, 1.05])
        axes[0, 2].set_xlabel('召回率')
        axes[0, 2].set_ylabel('精确率')
        axes[0, 2].set_title('精确率-召回率曲线')
        axes[0, 2].legend(loc="upper right")
        axes[0, 2].grid(True)

        # 4. 预测概率分布
        axes[1, 0].hist(y_prob[y_test == 0], bins=30, alpha=0.7,
                        label='无疾病', color='skyblue')
        axes[1, 0].hist(y_prob[y_test == 1], bins=30, alpha=0.7,
                        label='有疾病', color='lightcoral')
        axes[1, 0].set_xlabel('预测概率')
        axes[1, 0].set_ylabel('频数')
        axes[1, 0].set_title('预测概率分布')
        axes[1, 0].legend()

        # 5. 特征重要性
        if hasattr(self.model, 'coef_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': abs(self.model.coef_[0])
            }).sort_values('importance', ascending=True)

            axes[1, 1].barh(feature_importance['feature'],
                            feature_importance['importance'],
                            color='steelblue')
            axes[1, 1].set_xlabel('特征重要性 (系数绝对值)')
            axes[1, 1].set_title('逻辑回归特征重要性')

        # 6. 性能指标总结
        metrics = ['准确率', 'AUC', '平均精确率']
        values = [self.results['accuracy'], self.results['auc'], self.results['average_precision']]

        axes[1, 2].bar(metrics, values, color=['lightgreen', 'lightblue', 'lightcoral'])
        axes[1, 2].set_ylabel('分数')
        axes[1, 2].set_title('性能指标总结')
        axes[1, 2].set_ylim(0, 1)

        # 在柱状图上添加数值
        for i, v in enumerate(values):
            axes[1, 2].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom')

        plt.tight_layout()
        plt.show()

        # 显示特征重要性表格
        if hasattr(self.model, 'coef_'):
            print("\n特征重要性排序:")
            importance_df = pd.DataFrame({
                '特征': self.feature_names,
                '系数': self.model.coef_[0],
                '重要性(绝对值)': abs(self.model.coef_[0])
            }).sort_values('重要性(绝对值)', ascending=False)

            print(importance_df.to_string(index=False))

    def permutation_importance_analysis(self, X_test, y_test):
        """排列重要性分析"""
        print("\n=== 排列重要性分析 ===")

        X_test_scaled = self.scaler.transform(X_test)

        def score_function(estimator, X, y):
            return roc_auc_score(y, estimator.predict_proba(X)[:, 1])

        perm_importance = permutation_importance(
            self.model, X_test_scaled, y_test,
            scoring=score_function,
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )

        # 创建重要性DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': perm_importance.importances_mean,
            'std': perm_importance.importances_std
        }).sort_values('importance', ascending=True)

        # 可视化排列重要性
        plt.figure(figsize=(12, 8))
        plt.barh(importance_df['feature'], importance_df['importance'],
                 xerr=importance_df['std'], color='lightcoral')
        plt.xlabel('排列重要性')
        plt.title('排列重要性分析')
        plt.tight_layout()
        plt.show()

        print("排列重要性排序:")
        print(importance_df.sort_values('importance', ascending=False).to_string(index=False))

        return importance_df

    def cross_validation(self, X, y, cv=5):
        """交叉验证"""
        print("\n=== 交叉验证 ===")

        # 合并标准化和模型
        from sklearn.pipeline import Pipeline

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(
                C=self.best_params['C'],
                penalty=self.best_params['penalty'],
                solver=self.best_params['solver'],
                max_iter=self.best_params['max_iter'],
                random_state=42
            ))
        ])

        # 执行交叉验证
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc')

        print(f"交叉验证AUC分数: {cv_scores}")
        print(f"平均AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

        # 可视化交叉验证结果
        plt.figure(figsize=(10, 6))
        plt.bar(range(1, cv + 1), cv_scores, color='lightblue', alpha=0.7)
        plt.axhline(y=cv_scores.mean(), color='red', linestyle='--',
                    label=f'平均AUC: {cv_scores.mean():.4f}')
        plt.xlabel('交叉验证折数')
        plt.ylabel('AUC分数')
        plt.title('交叉验证结果')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        return cv_scores

    def save_model(self, filepath='logistic_regression_model.pkl'):
        """保存模型"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'best_params': self.best_params,
            'results': self.results
        }

        joblib.dump(model_data, filepath)
        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath='logistic_regression_model.pkl'):
        """加载模型"""
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.best_params = model_data['best_params']
        self.results = model_data['results']

        print(f"模型已从 {filepath} 加载")

    def predict_new_data(self, new_data):
        """预测新数据"""
        if self.model is None:
            raise ValueError("模型未训练，请先训练模型或加载已训练的模型")

        # 确保新数据包含所有特征
        missing_features = set(self.feature_names) - set(new_data.columns)
        if missing_features:
            raise ValueError(f"缺少特征: {missing_features}")

        # 选择特征并标准化
        X_new = new_data[self.feature_names]
        X_new_scaled = self.scaler.transform(X_new)

        # 预测
        predictions = self.model.predict(X_new_scaled)
        probabilities = self.model.predict_proba(X_new_scaled)[:, 1]

        result = pd.DataFrame({
            '预测类别': predictions,
            '患病概率': probabilities
        })

        return result


def main():
    """主函数"""
    print("=== 心血管疾病预测 - 逻辑回归模型 ===")

    # 初始化分类器
    lr_classifier = CardiovascularLRClassifier()

    try:
        # 1. 加载和预处理数据
        df = lr_classifier.load_and_preprocess_data('5_心血管疾病预测.csv')

        # 2. 数据可视化
        df = lr_classifier.visualize_data_distribution(df)

        # 3. 特征工程
        X, y = lr_classifier.feature_engineering(df)

        # 4. 数据划分
        X_train, X_val, X_test, y_train, y_val, y_test = lr_classifier.prepare_data(X, y)

        # 5. 处理不平衡数据
        X_train_resampled, y_train_resampled = lr_classifier.handle_imbalance(X_train, y_train)

        # 6. 超参数调优
        best_model = lr_classifier.hyperparameter_tuning(
            X_train_resampled, y_train_resampled, X_val, y_val)

        # 7. 训练最终模型
        final_model = lr_classifier.train_final_model(
            X_train_resampled, y_train_resampled, X_val, y_val)

        # 8. 模型评估
        accuracy, auc_score, ap_score, y_pred, y_prob, cm = lr_classifier.evaluate_model(
            final_model, X_test, y_test)

        # 9. 结果可视化
        lr_classifier.visualize_results(X_test, y_test, y_pred, y_prob)

        # 10. 排列重要性分析
        perm_importance_df = lr_classifier.permutation_importance_analysis(X_test, y_test)

        # 11. 交叉验证
        cv_scores = lr_classifier.cross_validation(
            pd.concat([X_train, X_val]), pd.concat([y_train, y_val]))

        # 12. 保存模型
        lr_classifier.save_model('cardiovascular_lr_model.pkl')

        # 13. 实验总结
        print("\n" + "=" * 50)
        print("实验总结")
        print("=" * 50)

        print("1. 数据预处理:")
        print("   - 年龄从天数转换为年")
        print("   - 计算BMI指数")
        print("   - 处理血压异常值")

        print("\n2. 特征工程:")
        print("   - 创建了血压比、脉压、平均动脉压等新特征")
        print("   - 创建了年龄-BMI、血压-BMI等交互特征")
        print("   - 创建了综合风险评分")

        print("\n3. 数据准备:")
        print("   - 使用SMOTE处理类别不平衡")
        print("   - 特征标准化")
        print("   - 训练集/验证集/测试集划分")

        print("\n4. 模型训练:")
        print(f"   - 最佳参数: {lr_classifier.best_params}")
        print(f"   - 测试集准确率: {accuracy:.4f}")
        print(f"   - 测试集AUC: {auc_score:.4f}")
        print(f"   - 测试集平均精确率: {ap_score:.4f}")

        print("\n5. 最重要的特征:")
        if hasattr(final_model, 'coef_'):
            importance_df = pd.DataFrame({
                '特征': lr_classifier.feature_names,
                '重要性': abs(final_model.coef_[0])
            }).sort_values('重要性', ascending=False)

            for i, row in importance_df.head(5).iterrows():
                print(f"   - {row['特征']}: {row['重要性']:.4f}")

        print("\n6. 模型优势:")
        print("   - 可解释性强，特征重要性明确")
        print("   - 训练速度快")
        print("   - 对线性关系建模效果好")
        print("   - 概率输出便于风险评估")

        # 示例预测
        print("\n" + "=" * 50)
        print("示例预测")
        print("=" * 50)

        # 创建示例数据
        example_data = pd.DataFrame({
            'age_years': [45, 60, 35],
            'gender': [1, 2, 1],
            'height': [170, 165, 175],
            'weight': [75, 80, 70],
            'ap_hi': [130, 140, 120],
            'ap_lo': [85, 90, 80],
            'cholesterol': [2, 3, 1],
            'gluc': [1, 2, 1],
            'smoke': [0, 1, 0],
            'alco': [0, 0, 0],
            'active': [1, 0, 1],
            'bmi': [25.95, 29.38, 22.86],
            'bp_ratio': [1.53, 1.56, 1.50],
            'pulse_pressure': [45, 50, 40],
            'map': [100.0, 106.7, 93.3],
            'age_bmi': [1167.75, 1762.80, 800.10],
            'bp_bmi': [3373.50, 4113.20, 2743.20],
            'risk_score': [3.0, 6.5, 2.0]
        })

        try:
            predictions = lr_classifier.predict_new_data(example_data)
            print("预测结果:")
            print(predictions)
        except Exception as e:
            print(f"预测错误: {e}")

        return lr_classifier

    except Exception as e:
        print(f"运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    classifier = main()