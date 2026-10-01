# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def lookfeature(data,target):
    #data = dataset["data"]
    #feature_names = dataset["feature_names"]
    feature_names = ("CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PIRATIO","B","LSTAT")
    #target = dataset["target"]
    # 合成字典
    boston_dict = {}
    column = 0
    for feature_name in feature_names:    # 把各个特征作为各个栏目加入字典
        boston_dict[feature_name] = data[:, column]
        column += 1
    boston_dict["price"] = target     # 把房价（标签）加入字典
    # 合成dataFrame
    pd_boston = pd.DataFrame(boston_dict)     # 将字典转为dataframe
    print(pd_boston)
    column = 1
    for feature_name in feature_names:   # 画房价根据单个特征变化的散点图，观察相关性
        plt.subplot(4, 4, column)
        plt.xlabel(feature_name)
        plt.ylabel('price')
        plt.scatter(pd_boston[feature_name], pd_boston["price"], s=0.6)
        column += 1
    plt.tight_layout()
    plt.show()


def get_price_plot1(y_test, y_pre):   # 绘制真实房价和预测房价的关系图，样本分布越接近45°直线则预测越准确
    # price-price plot
    plt.figure()
    plt.grid()
    plt.title('LinearRegression Boston')
    plt.xlabel('price_real')
    plt.ylabel('price_predict')
    plt.scatter(y_test, y_pre, color='black')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='blue', linewidth=3)
    plt.show()


def get_price_plot2(y_test, y_pre):    # 房价曲线
    # price-test_set_order plot
    plt.figure(figsize=(10, 5))
    plt.grid(True)
    plt.title('LinearRegression Boston')
    plt.xlabel('test order')
    plt.ylabel('price')
    plt.plot(range(len(y_test)), y_pre, color='blue', linewidth=1)
    plt.plot(range(len(y_test)), y_test, color='red', linewidth=1)
    plt.legend(['price_predict', 'price_real'], loc='lower right')
    plt.show()


if __name__ == "__main__":
    # 加载数据集、观察数据集
    # boston = load_boston()
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    print(raw_df.keys())
    # lookfeature(raw_df)
    # print(boston.keys())
    # lookfeature(boston)

    # 划分数据集
    # data = boston["data"]
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    # feature_names = boston["feature_names"]
    # target = boston["target"]
    target = raw_df.values[1::2, 2]
    data = data[target != 50]    # 剔除异常值（认为房价为50的是异常数据）
    target = target[target != 50]
    lookfeature(data,target)
    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.20, random_state=0)

    # 训练及测试
    clf1 = LinearRegression()        # 获取线性回归模型，默认最小二乘法
    clf1.fit(x_train, y_train)       # 训练
    y_pre = clf1.predict(x_test)     # 对测试数据进行房价预测
    print(u'系数', clf1.coef_)
    print(u'截距', clf1.intercept_)
    print("使用LinearRegression模型的均方误差为:", metrics.mean_squared_error(y_test, y_pre))
    print("使用LinearRegression模型的均方根误差为:", np.sqrt(metrics.mean_squared_error(y_test, y_pre)))
    y_preCV5 = cross_val_predict(clf1, data, target, cv=5)    # CV5交叉验证
    print("使用交叉验证的均方误差为:", metrics.mean_squared_error(target, y_preCV5))
    print("使用交叉验证的均方根误差为:", np.sqrt(metrics.mean_squared_error(target, y_preCV5)))

    # 绘图观察结果
    get_price_plot1(y_test, y_pre)   # 获取真实房价和预测房价关系图
    get_price_plot2(y_test, y_pre)   # 获取房价曲线

