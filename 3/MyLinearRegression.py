from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import metrics
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 字体设置
try:
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False
except:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False


def lookfeature(data, target, dataset_name="Boston"):
    """观察数据集特征与目标变量的关系 - 先绘制散点图，再绘制热力图"""
    if dataset_name == "Boston":
        feature_names = (
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PIRATIO", "B", "LSTAT")
    elif dataset_name == "California":
        feature_names = (
        "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "Latitude", "Longitude")

    # 合成字典
    data_dict = {}
    column = 0
    for feature_name in feature_names:
        data_dict[feature_name] = data[:, column]
        column += 1
    data_dict["price"] = target

    # 合成dataFrame
    pd_data = pd.DataFrame(data_dict)
    print(f"\n=== {dataset_name}数据集基本信息 ===")
    print(f"数据集形状: {data.shape}")
    print(f"目标变量范围: {target.min():.2f} - {target.max():.2f}")

    # 计算相关性
    correlation_matrix = pd_data.corr()
    target_correlation = correlation_matrix['price'].abs().sort_values(ascending=False)
    print(f"\n与房价相关性最高的特征:")
    print(target_correlation[1:4])  # 排除自身相关性

    # 绘制各个特征与房价的关系图 - 先绘制散点图
    plt.figure(figsize=(15, 12))
    column = 1
    for feature_name in feature_names:
        plt.subplot(4, 4, column)
        plt.xlabel(feature_name)
        plt.ylabel('price')
        plt.scatter(pd_data[feature_name], pd_data["price"], s=0.6)
        column += 1
    plt.tight_layout()
    plt.show()

    # 绘制相关性热力图 - 在散点图之后显示
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
    plt.title(f'{dataset_name}数据集特征相关性热力图')
    plt.tight_layout()
    plt.show()

    return pd_data, feature_names


def compare_models_plots(y_test, y_pre_lr, y_pre_ridge, dataset_name="Boston"):
    """在同一图窗中对比线性回归和岭回归的预测效果"""

    # 图1: 真实值 vs 预测值对比
    plt.figure(figsize=(15, 6))

    # 线性回归
    plt.subplot(1, 2, 1)
    plt.grid(True, alpha=0.3)
    r2_lr = metrics.r2_score(y_test, y_pre_lr)
    plt.title(f'LinearRegression - {dataset_name} (R2 = {r2_lr:.4f})')
    plt.xlabel('price_real')
    plt.ylabel('price_predict')
    plt.scatter(y_test, y_pre_lr, color='black', alpha=0.7, s=20)
    min_val = min(y_test.min(), y_pre_lr.min())
    max_val = max(y_test.max(), y_pre_lr.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='blue', linewidth=3)

    # 岭回归
    plt.subplot(1, 2, 2)
    plt.grid(True, alpha=0.3)
    r2_ridge = metrics.r2_score(y_test, y_pre_ridge)
    plt.title(f'RidgeRegression - {dataset_name} (R2 = {r2_ridge:.4f})')
    plt.xlabel('price_real')
    plt.ylabel('price_predict')
    plt.scatter(y_test, y_pre_ridge, color='black', alpha=0.7, s=20)
    min_val = min(y_test.min(), y_pre_ridge.min())
    max_val = max(y_test.max(), y_pre_ridge.max())
    plt.plot([min_val, max_val], [min_val, max_val], color='blue', linewidth=3)

    plt.tight_layout()
    plt.show()

    # 图2: 房价预测曲线对比
    plt.figure(figsize=(15, 6))

    # 线性回归 - 保持PPT中的绘制方式
    plt.subplot(1, 2, 1)
    plt.grid(True)
    plt.title(f'LinearRegression {dataset_name}')
    plt.xlabel('test order')
    plt.ylabel('price')
    plt.plot(range(len(y_test)), y_pre_lr, color='blue', linewidth=1)
    plt.plot(range(len(y_test)), y_test, color='red', linewidth=1)
    plt.legend(['price_predict', 'price_real'], loc='lower right')

    # 岭回归 - 保持PPT中的绘制方式
    plt.subplot(1, 2, 2)
    plt.grid(True)
    plt.title(f'RidgeRegression {dataset_name}')
    plt.xlabel('test order')
    plt.ylabel('price')
    plt.plot(range(len(y_test)), y_pre_ridge, color='blue', linewidth=1)
    plt.plot(range(len(y_test)), y_test, color='red', linewidth=1)
    plt.legend(['price_predict', 'price_real'], loc='lower right')

    plt.tight_layout()
    plt.show()

    return r2_lr, r2_ridge


def parameter_comparison(data, target, dataset_name="Boston"):
    """参数对比实验 - 使用训练集、验证集和测试集"""
    print(f"\n=== {dataset_name} - 参数对比实验 ===")

    # 将数据划分为训练集、验证集和测试集
    # 首先划分训练集和测试集 (60% 训练+验证, 40% 测试)
    x_temp, x_test, y_temp, y_test = train_test_split(data, target, test_size=0.4, random_state=42)

    # 然后将训练集划分为训练集和验证集 (75% 训练, 25% 验证)
    x_train, x_val, y_train, y_val = train_test_split(x_temp, y_temp, test_size=0.25, random_state=42)

    print(f"训练集大小: {x_train.shape[0]}")
    print(f"验证集大小: {x_val.shape[0]}")
    print(f"测试集大小: {x_test.shape[0]}")

    results = []

    # 1. 不同正则化参数对比（岭回归） - 在验证集上评估
    print("\n--- 岭回归不同alpha参数对比 (在验证集上评估) ---")

    # 标准化数据
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)

    alphas = [0.001, 0.01, 0.1, 1, 10, 100]

    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(x_train_scaled, y_train)
        y_pred_ridge = ridge.predict(x_val_scaled)

        r2 = metrics.r2_score(y_val, y_pred_ridge)
        rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred_ridge))

        results.append({
            '参数类型': '岭回归alpha',
            '参数值': alpha,
            '模型': '岭回归',
            'r2': r2,
            'rmse': rmse
        })

        print(f"alpha = {alpha:8.3f}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")

    # 2. 不同特征数量对比 - 在验证集上评估
    print("\n--- 不同特征数量对比 (在验证集上评估) ---")

    # 根据数据集确定特征数量范围
    if dataset_name == "Boston":
        feature_counts = [3, 5, 8, 10, 13]
    else:  # California
        feature_counts = [3, 5, 8]

    for k in feature_counts:
        # 使用SelectKBest选择最佳特征
        selector = SelectKBest(f_regression, k=k)
        x_train_selected = selector.fit_transform(x_train, y_train)
        x_val_selected = selector.transform(x_val)

        lr = LinearRegression()
        lr.fit(x_train_selected, y_train)
        y_pred_selected = lr.predict(x_val_selected)

        r2 = metrics.r2_score(y_val, y_pred_selected)
        rmse = np.sqrt(metrics.mean_squared_error(y_val, y_pred_selected))

        results.append({
            '参数类型': '特征数量',
            '参数值': k,
            '模型': '线性回归',
            'r2': r2,
            'rmse': rmse
        })

        print(f"特征数量 {k:2d}: R2 = {r2:.4f}, RMSE = {rmse:.4f}")

    # 转换为DataFrame并返回
    results_df = pd.DataFrame(results)

    # 可视化参数对比结果
    plt.figure(figsize=(15, 5))

    # 岭回归alpha对比
    plt.subplot(1, 3, 1)
    ridge_results = results_df[results_df['参数类型'] == '岭回归alpha']
    plt.semilogx(ridge_results['参数值'], ridge_results['r2'], 'o-', linewidth=2, markersize=6, color='red')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('R2 Score')
    plt.title(f'{dataset_name} - 岭回归Alpha参数对R2的影响')
    plt.grid(True, alpha=0.3)

    # 特征数量对比
    plt.subplot(1, 3, 2)
    feature_results = results_df[results_df['参数类型'] == '特征数量']
    plt.plot(feature_results['参数值'], feature_results['r2'], 'o-', linewidth=2, markersize=6, color='green')
    plt.xlabel('特征数量')
    plt.ylabel('R2 Score')
    plt.title(f'{dataset_name} - 特征数量对R2的影响')
    plt.grid(True, alpha=0.3)

    # 最佳参数性能对比
    plt.subplot(1, 3, 3)
    best_ridge = ridge_results.loc[ridge_results['r2'].idxmax()]
    best_features = feature_results.loc[feature_results['r2'].idxmax()]

    models = ['岭回归', '线性回归']
    r2_scores = [best_ridge['r2'], best_features['r2']]

    plt.bar(models, r2_scores, color=['red', 'green'], alpha=0.7)
    plt.xlabel('模型')
    plt.ylabel('最佳R2 Score')
    plt.title(f'{dataset_name} - 最佳参数性能对比')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    return results_df, x_train, x_val, x_test, y_train, y_val, y_test


def run_boston_experiment():
    """运行波士顿房价数据集实验"""
    print("=" * 60)
    print("波士顿房价数据集线性回归实验")
    print("=" * 60)

    # 加载波士顿数据集
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    # 剔除异常值
    data = data[target != 50]
    target = target[target != 50]

    # 观察数据 - 先绘制散点图，再绘制热力图
    pd_data, feature_names = lookfeature(data, target, "Boston")

    # 参数对比实验 - 返回划分好的数据集
    param_results, x_train, x_val, x_test, y_train, y_val, y_test = parameter_comparison(data, target, "Boston")

    # 使用最佳参数进行最终训练和评估
    print("\n=== 使用最佳参数进行最终评估 ===")

    # 找到最佳参数组合
    ridge_results = param_results[param_results['模型'] == '岭回归']
    feature_results = param_results[param_results['模型'] == '线性回归']

    if not ridge_results.empty:
        best_ridge = ridge_results.loc[ridge_results['r2'].idxmax()]
        print(f"岭回归最佳alpha: {best_ridge['参数值']} (验证集R2 = {best_ridge['r2']:.4f})")

    if not feature_results.empty:
        best_features = feature_results.loc[feature_results['r2'].idxmax()]
        print(f"最佳特征数量: {best_features['参数值']} (验证集R2 = {best_features['r2']:.4f})")

    # 使用最佳参数在测试集上评估模型
    print("\n=== 在测试集上评估模型性能 ===")

    # 1. 线性回归（所有特征）
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pre_lr = lr.predict(x_test)

    print(u'线性回归系数:', lr.coef_)
    print(u'线性回归截距:', lr.intercept_)
    print("使用LinearRegression模型的均方误差为:", metrics.mean_squared_error(y_test, y_pre_lr))
    print("使用LinearRegression模型的均方根误差为:", np.sqrt(metrics.mean_squared_error(y_test, y_pre_lr)))
    print("使用LinearRegression模型的R2为:", metrics.r2_score(y_test, y_pre_lr))

    # 2. 岭回归（使用最佳alpha）
    if not ridge_results.empty:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        ridge = Ridge(alpha=best_ridge['参数值'])
        ridge.fit(x_train_scaled, y_train)
        y_pre_ridge = ridge.predict(x_test_scaled)

        print(u'岭回归系数:', ridge.coef_)
        print(u'岭回归截距:', ridge.intercept_)
        print("使用RidgeRegression模型的均方误差为:", metrics.mean_squared_error(y_test, y_pre_ridge))
        print("使用RidgeRegression模型的均方根误差为:", np.sqrt(metrics.mean_squared_error(y_test, y_pre_ridge)))
        print("使用RidgeRegression模型的R2为:", metrics.r2_score(y_test, y_pre_ridge))

    # 3. 特征选择（使用最佳特征数量）
    if not feature_results.empty:
        selector = SelectKBest(f_regression, k=int(best_features['参数值']))
        x_train_selected = selector.fit_transform(x_train, y_train)
        x_test_selected = selector.transform(x_test)

        lr_selected = LinearRegression()
        lr_selected.fit(x_train_selected, y_train)
        y_pre_selected = lr_selected.predict(x_test_selected)

        print(u'特征选择线性回归系数:', lr_selected.coef_)
        print(u'特征选择线性回归截距:', lr_selected.intercept_)
        print("使用特征选择LinearRegression模型的均方误差为:", metrics.mean_squared_error(y_test, y_pre_selected))
        print("使用特征选择LinearRegression模型的均方根误差为:",
              np.sqrt(metrics.mean_squared_error(y_test, y_pre_selected)))
        print("使用特征选择LinearRegression模型的R2为:", metrics.r2_score(y_test, y_pre_selected))

    # 交叉验证
    print("\n=== 交叉验证结果 ===")
    y_preCV5 = cross_val_predict(LinearRegression(), data, target, cv=5)
    print("使用交叉验证的均方误差为:", metrics.mean_squared_error(target, y_preCV5))
    print("使用交叉验证的均方根误差为:", np.sqrt(metrics.mean_squared_error(target, y_preCV5)))
    print("使用交叉验证的R2为:", metrics.r2_score(target, y_preCV5))

    # 在同一图窗中对比线性回归和岭回归的预测效果
    if not ridge_results.empty:
        r2_lr, r2_ridge = compare_models_plots(y_test, y_pre_lr, y_pre_ridge, "Boston")

        # 输出模型性能对比
        print(f"\n=== 模型性能对比 (测试集) ===")
        print(f"线性回归 R2: {r2_lr:.4f}")
        print(f"岭回归 R2: {r2_ridge:.4f}")
        if r2_ridge > r2_lr:
            print("岭回归模型性能优于线性回归")
        elif r2_ridge < r2_lr:
            print("线性回归模型性能优于岭回归")
        else:
            print("两个模型性能相同")

    return param_results


def run_california_experiment():
    """运行加州房价数据集实验"""
    print("\n" + "=" * 60)
    print("加州房价数据集线性回归实验")
    print("=" * 60)

    # 加载加州房价数据集
    california = fetch_california_housing()
    data = california.data
    target = california.target

    # 观察数据 - 先绘制散点图，再绘制热力图
    pd_data, feature_names = lookfeature(data, target, "California")

    # 参数对比实验 - 返回划分好的数据集
    param_results, x_train, x_val, x_test, y_train, y_val, y_test = parameter_comparison(data, target, "California")

    # 使用最佳参数进行最终训练和评估
    print("\n=== 使用最佳参数进行最终评估 ===")

    # 找到最佳参数组合
    ridge_results = param_results[param_results['模型'] == '岭回归']
    feature_results = param_results[param_results['模型'] == '线性回归']

    if not ridge_results.empty:
        best_ridge = ridge_results.loc[ridge_results['r2'].idxmax()]
        print(f"岭回归最佳alpha: {best_ridge['参数值']} (验证集R2 = {best_ridge['r2']:.4f})")

    if not feature_results.empty:
        best_features = feature_results.loc[feature_results['r2'].idxmax()]
        print(f"最佳特征数量: {best_features['参数值']} (验证集R2 = {best_features['r2']:.4f})")

    # 使用最佳参数在测试集上评估模型
    print("\n=== 在测试集上评估模型性能 ===")

    # 1. 线性回归（所有特征）
    lr = LinearRegression()
    lr.fit(x_train, y_train)
    y_pre_lr = lr.predict(x_test)

    print(u'线性回归系数:', lr.coef_)
    print(u'线性回归截距:', lr.intercept_)
    print("使用LinearRegression模型的均方误差为:", metrics.mean_squared_error(y_test, y_pre_lr))
    print("使用LinearRegression模型的均方根误差为:", np.sqrt(metrics.mean_squared_error(y_test, y_pre_lr)))
    print("使用LinearRegression模型的R2为:", metrics.r2_score(y_test, y_pre_lr))

    # 2. 岭回归（使用最佳alpha）
    if not ridge_results.empty:
        scaler = StandardScaler()
        x_train_scaled = scaler.fit_transform(x_train)
        x_test_scaled = scaler.transform(x_test)

        ridge = Ridge(alpha=best_ridge['参数值'])
        ridge.fit(x_train_scaled, y_train)
        y_pre_ridge = ridge.predict(x_test_scaled)

        print(u'岭回归系数:', ridge.coef_)
        print(u'岭回归截距:', ridge.intercept_)
        print("使用RidgeRegression模型的均方误差为:", metrics.mean_squared_error(y_test, y_pre_ridge))
        print("使用RidgeRegression模型的均方根误差为:", np.sqrt(metrics.mean_squared_error(y_test, y_pre_ridge)))
        print("使用RidgeRegression模型的R2为:", metrics.r2_score(y_test, y_pre_ridge))

    # 3. 特征选择（使用最佳特征数量）
    if not feature_results.empty:
        selector = SelectKBest(f_regression, k=int(best_features['参数值']))
        x_train_selected = selector.fit_transform(x_train, y_train)
        x_test_selected = selector.transform(x_test)

        lr_selected = LinearRegression()
        lr_selected.fit(x_train_selected, y_train)
        y_pre_selected = lr_selected.predict(x_test_selected)

        print(u'特征选择线性回归系数:', lr_selected.coef_)
        print(u'特征选择线性回归截距:', lr_selected.intercept_)
        print("使用特征选择LinearRegression模型的均方误差为:", metrics.mean_squared_error(y_test, y_pre_selected))
        print("使用特征选择LinearRegression模型的均方根误差为:",
              np.sqrt(metrics.mean_squared_error(y_test, y_pre_selected)))
        print("使用特征选择LinearRegression模型的R2为:", metrics.r2_score(y_test, y_pre_selected))

    # 交叉验证
    print("\n=== 交叉验证结果 ===")
    y_preCV5 = cross_val_predict(LinearRegression(), data, target, cv=5)
    print("使用交叉验证的均方误差为:", metrics.mean_squared_error(target, y_preCV5))
    print("使用交叉验证的均方根误差为:", np.sqrt(metrics.mean_squared_error(target, y_preCV5)))
    print("使用交叉验证的R2为:", metrics.r2_score(target, y_preCV5))

    # 在同一图窗中对比线性回归和岭回归的预测效果
    if not ridge_results.empty:
        r2_lr, r2_ridge = compare_models_plots(y_test, y_pre_lr, y_pre_ridge, "California")

        # 输出模型性能对比
        print(f"\n=== 模型性能对比 (测试集) ===")
        print(f"线性回归 R2: {r2_lr:.4f}")
        print(f"岭回归 R2: {r2_ridge:.4f}")
        if r2_ridge > r2_lr:
            print("岭回归模型性能优于线性回归")
        elif r2_ridge < r2_lr:
            print("线性回归模型性能优于岭回归")
        else:
            print("两个模型性能相同")

    return param_results


if __name__ == "__main__":
    """
    主函数 - 用户选择实验，可以重复选择
    """
    while True:
        print("\n" + "=" * 40)
        print("线性回归实验选择")
        print("=" * 40)
        print("1. 波士顿房价回归实验")
        print("2. 加州房价回归实验")
        print("3. 退出程序")
        print("=" * 40)

        choice = input("请选择实验（输入1、2或3）: ")

        if choice == '1':
            run_boston_experiment()
            print("\n实验完成！")
            continue_choice = input("是否继续其他实验？(y/n): ")
            if continue_choice.lower() != 'y':
                print("感谢使用，再见！")
                break
        elif choice == '2':
            run_california_experiment()
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