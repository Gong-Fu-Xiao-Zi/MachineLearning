from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_iris_dataset():
    """
    加载鸢尾花数据集
    """
    iris = datasets.load_iris()
    return iris


def load_wine_dataset():
    """
    加载葡萄酒数据集
    """
    wine = datasets.load_wine()
    return wine


def look_dataset(dataset, dataset_name):
    """
    查看数据集基本信息
    """
    print("=" * 50)
    print(f"{dataset_name}数据集基本信息")
    print("=" * 50)

    # 数据集基本信息
    print(f"数据集包含的键: {list(dataset.keys())}")
    print(f"数据形状: {dataset.data.shape}")
    print(f"目标变量形状: {dataset.target.shape}")
    print(f"特征名称: {dataset.feature_names}")
    print(f"目标类别名称: {dataset.target_names}")

    # 显示前5个样本
    print("\n前5个样本数据:")
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['target'] = dataset.target
    df['class'] = df['target'].apply(lambda x: dataset.target_names[x])
    print(df.head())

    return df


def analyze_features(dataset, dataset_name):
    """
    分析特征分布及特征与类别关系
    """
    print("\n" + "=" * 50)
    print("特征分析")
    print("=" * 50)

    # 创建DataFrame
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['class'] = [dataset.target_names[x] for x in dataset.target]

    # 1. 特征统计信息
    print("各特征的统计信息:")
    print(df.describe())

    # 2. 各类别样本数量
    print("\n各类别样本数量:")
    print(df['class'].value_counts())

    # 3. 各特征与类别的相关性
    numeric_df = df.drop('class', axis=1)  # 移除字符串列
    numeric_df['target'] = dataset.target  # 使用数值目标变量
    print("\n特征与目标变量的相关性:")
    correlation = numeric_df.corr()['target'].sort_values(ascending=False)
    print(correlation)

    return df


def visualize_pairplot(dataset, dataset_name):
    """
    观察特征两两关系 - 散点图矩阵
    """
    print("\n" + "=" * 50)
    print("特征两两关系图")
    print("=" * 50)

    # 创建DataFrame
    df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    df['species'] = [dataset.target_names[x] for x in dataset.target]

    print("正在生成特征两两关系图...")

    # 使用seaborn的pairplot函数
    # 注意：这里直接使用sns.pairplot，不要用plt.figure包装
    pair_plot = sns.pairplot(df, hue='species', palette='viridis', diag_kind='hist')
    pair_plot.fig.suptitle(f'{dataset_name}数据集特征两两关系图', y=1.02)

    # 显示图形
    plt.show()

    return df


def visualize_features(dataset_df, dataset_name):
    """
    特征可视化 - 只保留多特征分布情况图（平行坐标图）
    """
    print("\n" + "=" * 50)
    print("特征可视化 - 多特征分布情况")
    print("=" * 50)

    # 平行坐标图（使用所有特征）
    print("正在生成平行坐标图...")
    plt.figure(figsize=(14, 8))
    pd.plotting.parallel_coordinates(dataset_df, 'class', color=('#FF6B6B', '#4ECDC4', '#45B7D1'))
    plt.title(f'{dataset_name}数据集多特征分布情况 - 平行坐标图')
    plt.xticks(rotation=45)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def knn_experiment(dataset, dataset_name, test_size=0.3, k=1, random_state=10, scale_data=True):
    """
    KNN分类实验
    """
    print("\n" + "=" * 50)
    print("KNN分类实验")
    print("=" * 50)

    X = dataset.data
    y = dataset.target

    # 数据标准化（对于葡萄酒数据集特征量纲差异较大，建议标准化）
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print("已对数据进行标准化处理")

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"K值: {k}")

    # 创建并训练KNN模型
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    # 预测
    y_pred = knn.predict(X_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n模型准确率: {accuracy:.4f}")

    # 详细分类报告
    print("\n详细分类报告:")
    print(classification_report(y_test, y_pred, target_names=dataset.target_names))

    return knn, X_train, X_test, y_train, y_test, y_pred, accuracy


def compare_k_values(dataset, dataset_name, scale_data=False):
    """
    比较不同K值对分类效果的影响 - 只显示测试集准确率
    """
    print("\n" + "=" * 50)
    print("不同K值对比实验")
    print("=" * 50)

    X = dataset.data
    y = dataset.target

    # 数据标准化
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # 测试不同的K值
    k_values = range(1, 21)
    test_scores = []

    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        test_score = knn.score(X_test, y_test)
        test_scores.append(test_score)

        if k in [1, 3, 5, 10, 15, 20]:
            print(f"K={k:2d} | 测试集准确率: {test_score:.4f}")

    # 绘制K值对比图
    plt.figure(figsize=(12, 6))
    plt.plot(k_values, test_scores, 's-', label='测试集准确率', color='red')
    plt.xlabel('K值')
    plt.ylabel('准确率')
    plt.title(f'不同K值对KNN分类准确率的影响（{dataset_name}数据集）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(k_values)
    plt.show()

    # 找到最佳K值
    best_k = k_values[np.argmax(test_scores)]
    best_score = max(test_scores)
    print(f"\n最佳K值: {best_k}, 对应的测试集准确率: {best_score:.4f}")

    return best_k


def compare_test_sizes(dataset, dataset_name, k=5, scale_data=False):
    """
    比较不同测试集比例对分类效果的影响
    """
    print("\n" + "=" * 50)
    print("不同测试集比例对比实验")
    print("=" * 50)

    X = dataset.data
    y = dataset.target

    # 数据标准化
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    test_sizes = [0.1, 0.2, 0.3, 0.4, 0.5]
    scores = []

    for test_size in test_sizes:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        score = knn.score(X_test, y_test)
        scores.append(score)

        print(
            f"测试集比例: {test_size:.1f} | 训练集大小: {X_train.shape[0]} | 测试集大小: {X_test.shape[0]} | 准确率: {score:.4f}")

    # 绘制测试集比例对比图
    plt.figure(figsize=(10, 6))
    plt.plot(test_sizes, scores, 'o-', color='green', linewidth=2, markersize=8)
    plt.xlabel('测试集比例')
    plt.ylabel('准确率')
    plt.title(f'不同测试集比例对KNN分类准确率的影响（{dataset_name}数据集）')
    plt.grid(True, alpha=0.3)
    plt.xticks(test_sizes)
    plt.show()


def feature_importance_analysis(dataset, dataset_name, scale_data=False):
    """
    特征重要性分析
    """
    print("\n" + "=" * 50)
    print("特征重要性分析")
    print("=" * 50)

    X = dataset.data
    y = dataset.target

    # 数据标准化
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    # 使用所有特征
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    knn_full = KNeighborsClassifier(n_neighbors=5)
    knn_full.fit(X_train, y_train)
    full_score = knn_full.score(X_test, y_test)
    print(f"使用所有特征的准确率: {full_score:.4f}")

    # 测试单个特征
    feature_names = dataset.feature_names
    single_scores = []

    for i, feature in enumerate(feature_names):
        X_single = X[:, i].reshape(-1, 1)
        X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
            X_single, y, test_size=0.3, random_state=42, stratify=y
        )

        knn_single = KNeighborsClassifier(n_neighbors=5)
        knn_single.fit(X_train_s, y_train_s)
        single_score = knn_single.score(X_test_s, y_test_s)
        single_scores.append(single_score)

        print(f"仅使用 {feature:25} 的准确率: {single_score:.4f}")

    # 绘制特征重要性图
    plt.figure(figsize=(14, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(feature_names)))
    bars = plt.bar(range(len(feature_names)), single_scores, color=colors)
    plt.axhline(y=full_score, color='red', linestyle='--', label=f'全特征准确率: {full_score:.4f}')
    plt.ylabel('准确率')
    plt.title(f'单特征分类准确率对比（{dataset_name}数据集）')
    plt.legend()
    plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha='right')
    plt.ylim(0, 1)

    # 在柱子上添加数值标签
    for bar, score in zip(bars, single_scores):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f'{score:.3f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.show()

    # 显示最佳单特征
    best_feature_idx = np.argmax(single_scores)
    print(f"\n最佳单特征: {feature_names[best_feature_idx]}, 准确率: {single_scores[best_feature_idx]:.4f}")


def visualize_predictions(dataset, dataset_name, y_test, y_pred):
    """
    可视化预测结果 - 使用二维图，横轴为测试集样本序号，纵轴为类别
    参考所给图片样式
    """
    # 创建测试集样本序号
    sample_indices = np.arange(1, len(y_test) + 1)

    # 创建图形
    plt.figure(figsize=(12, 6))

    # 绘制真实类别
    plt.scatter(sample_indices, y_test, color='blue', marker='o', alpha=0.7, label='真实类别')

    # 绘制预测类别
    plt.scatter(sample_indices, y_pred, color='red', marker='x', alpha=0.7, label='预测类别')

    # 设置图形属性
    plt.title(f'KNN分类结果 - {dataset_name}数据集')
    plt.xlabel('测试集样本序号')
    plt.ylabel('类别')

    # 设置纵坐标刻度
    plt.yticks([0, 1, 2], dataset.target_names)

    # 设置横坐标刻度 - 显示奇数序号，与参考图片一致
    if len(y_test) <= 50:
        plt.xticks(range(1, len(y_test) + 1, 2))
    else:
        plt.xticks(range(1, len(y_test) + 1, max(1, len(y_test) // 20)))

    # 添加图例
    plt.legend(loc='upper right')

    # 添加网格
    plt.grid(True, alpha=0.3)

    # 标记错误分类的样本
    errors = np.where(y_test != y_pred)[0]
    if len(errors) > 0:
        for error_idx in errors:
            plt.axvline(x=error_idx + 1, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

    # 输出错误分类的样本信息
    print(f"\n错误分类的样本数量: {len(errors)}")
    if len(errors) > 0:
        print("错误分类的样本详情:")
        error_df = pd.DataFrame({
            '样本序号': errors + 1,
            '真实类别': [dataset.target_names[y_test[i]] for i in errors],
            '预测类别': [dataset.target_names[y_pred[i]] for i in errors]
        })
        print(error_df)


def run_iris_experiment():
    """
    执行鸢尾花实验
    """
    print("KNN鸢尾花分类实验")
    print("=" * 50)

    # 1. 加载数据集
    iris = load_iris_dataset()
    dataset_name = "鸢尾花"

    # 2. 查看数据集基本信息
    iris_df = look_dataset(iris, dataset_name)

    # 3. 分析特征
    iris_df = analyze_features(iris, dataset_name)

    # 4. 特征两两关系图（新增）
    visualize_pairplot(iris, dataset_name)

    # 5. 特征可视化 - 只保留多特征分布情况图
    visualize_features(iris_df, dataset_name)

    # 6. 基础KNN实验
    knn_model, X_train, X_test, y_train, y_test, y_pred, accuracy = knn_experiment(iris, dataset_name)

    # 7. 不同K值对比
    best_k = compare_k_values(iris, dataset_name)

    # 8. 不同测试集比例对比
    compare_test_sizes(iris, dataset_name, k=best_k)

    # 9. 特征重要性分析
    feature_importance_analysis(iris, dataset_name)

    # 10. 使用最佳参数重新训练并可视化结果
    print("\n" + "=" * 50)
    print("使用最佳参数进行最终训练")
    print("=" * 50)

    final_knn, X_train_final, X_test_final, y_train_final, y_test_final, y_pred_final, final_accuracy = \
        knn_experiment(iris, dataset_name, k=best_k)

    # 11. 可视化预测结果
    visualize_predictions(iris, dataset_name, y_test_final, y_pred_final)



def run_wine_experiment():
    """
    执行葡萄酒实验
    """
    print("KNN葡萄酒分类实验")
    print("=" * 50)

    # 1. 加载数据集
    wine = load_wine_dataset()
    dataset_name = "葡萄酒"

    # 2. 查看数据集基本信息
    wine_df = look_dataset(wine, dataset_name)

    # 3. 分析特征
    wine_df = analyze_features(wine, dataset_name)

    # 4. 特征可视化 - 只保留多特征分布情况图
    visualize_features(wine_df, dataset_name)

    # 5. 基础KNN实验
    knn_model, X_train, X_test, y_train, y_test, y_pred, accuracy = knn_experiment(wine, dataset_name, scale_data=True)

    # 6. 不同K值对比
    best_k = compare_k_values(wine, dataset_name, scale_data=True)

    # 7. 不同测试集比例对比
    compare_test_sizes(wine, dataset_name, k=best_k, scale_data=True)

    # 8. 特征重要性分析
    feature_importance_analysis(wine, dataset_name, scale_data=True)

    # 9. 使用最佳参数重新训练并可视化结果
    print("\n" + "=" * 50)
    print("使用最佳参数进行最终训练")
    print("=" * 50)

    final_knn, X_train_final, X_test_final, y_train_final, y_test_final, y_pred_final, final_accuracy = \
        knn_experiment(wine, dataset_name, k=best_k, scale_data=True)

    # 10. 可视化预测结果
    visualize_predictions(wine, dataset_name, y_test_final, y_pred_final)

def main():
    """
    主函数 - 用户选择实验，可以重复选择
    """
    while True:
        print("\n" + "=" * 40)
        print("KNN分类实验选择")
        print("=" * 40)
        print("1. 鸢尾花分类实验")
        print("2. 葡萄酒分类实验")
        print("3. 退出程序")
        print("=" * 40)

        choice = input("请选择实验（输入1、2或3）: ")

        if choice == '1':
            run_iris_experiment()
            print("\n实验完成！")
            continue_choice = input("是否继续其他实验？(y/n): ")
            if continue_choice.lower() != 'y':
                print("感谢使用，再见！")
                break
        elif choice == '2':
            run_wine_experiment()
            print("\n实验完成！")
            continue_choice = input("是否继续其他实验？(y/n): ")
            if continue_choice.lower() != 'y':
                print("感谢使用，再见！")
                break
        elif choice == '3':
            print("感谢使用，再见！")
            break
        else:
            print("输入无效，请重新输入1、2或3")


if __name__ == "__main__":
    main()