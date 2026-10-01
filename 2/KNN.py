from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def lookdataset(iris):
    """
    lookdataset is a function to look some basic information of the dataset.
    parameter : the dataset you load.
    returns : some "prints" you can see below, containing some shape information and 5 samples.

    """
    print('\nHere is the output of function lookdataset: \n')
    print(iris.keys())  # 观察数据集构成
    print('The shape of iris_target is :', iris.target.shape)  # 样本类别的维度 150*1
    print('The 3 types of flowers are :', iris.target_names)  # 三个类别的名称
    print('The shape of iris_data is :', iris.data.shape)  # 样本特征的为度 150*4
    print('The 4 features are :', iris.feature_names)  # 罗列4个特征
    print('some of features data :')
    print(iris.data[:5])  # 展示部分特征数据


def lookfeatures(iris):
    """
    lookfeatures is a function to look the relationships between every two features.
    parameter : the dataset you load.
    returns : 16 plots of the single-single feature relationship;
              to help understand, here also provide 5 samples of data after merging the
              features and labels.
    """
    print('\nHere is the output of function lookdataset: \n')
    sns.set(style="white", color_codes=True)
    iris_frame = pd.DataFrame(iris['data'], columns=iris['feature_names'])   # 把鸢尾花特征数据结构变为dataframe，可该行下面加print观察
    iris_frame = pd.merge(iris_frame, pd.DataFrame(iris['target'], columns=['species']), left_index=True, right_index=True)     # 在dataframe最右加多1列标签
    labels = dict(zip([0, 1, 2], iris['target_names']))     # 生成字典，建立0，1，2和三个类别的名字的映射
    iris_frame['species'] = iris_frame['species'].apply(lambda x: labels[x])  # 利用上述映射把标签列的内容（012）对应为类别名
    print(iris_frame[:5])          # 查看设计好的dataframe
    sns.pairplot(iris_frame, hue='species', size=3, diag_kind='hist')   # 绘制两两特征图，参数的数据结构为dataframe
    plt.show()


def getparallel(data_origin):     # 获取平行坐标图，同样要先生成dataframe
    data = data_origin["data"]
    target = data_origin["target"]
    target_names = data_origin["target_names"]  # 相当于查找表，与前面的字典作用相同
    target_labels = []
    for class_num in target:     # 利用查找表把标签列的内容（012）对应为类别名
        target_labels.append(target_names[class_num])

    feature_names = data_origin["feature_names"]
    # 合成字典
    data_dict = {}   # 可以直接把一个字典变为dataframe的结构，所以把特征、标签等内容先合成字典
    column = 0
    for feature_name in feature_names:  # feature_name="sepal length", sepal width, .. , ..
        data_dict[feature_name] = data[:, column]  # dict={"sepal length":data[:,0] , ....    }
        column += 1
    data_dict["target_labels"] = target_labels
    # 合成dataFrame
    pd_data = pd.DataFrame(data_dict)    # 字典构建好后，直接转成dataframe结构，字典各栏目即为dataframe各列
    # 画图
    plt.figure()
    pd.plotting.parallel_coordinates(pd_data, "target_labels")     # 绘制平行坐标
    plt.show()


iris = datasets.load_iris()  # 读入鸢尾花数据集
lookdataset(iris)  # 获得数据集相关信息
lookfeatures(iris)  # 查看特征两两之间分布情况

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33, random_state=0)
print('The shape of x_train is :', x_train.shape)  # 100个训练样本，每个样本4个特征，1个标签
print('The shape of y_train is :', y_train.shape)
print('The shape of x_test is :', x_test.shape)    # 50个测试样本，每个样本4个特征，1个标签
print('The shape of y_test is :', y_test.shape)

# 训练、测试
knn = KNeighborsClassifier(n_neighbors=1)   # 获得kNN模型
knn.fit(x_train, y_train)           # 用训练集数据对模型进行训练
y_pre = knn.predict(x_test)
score = knn.score(x_test, y_test)   # 比较测试集的模型输出和真实标签，获得分数（一定程度表征正确率）
print('The score of this classifier is :', score)
getparallel(iris)            # 获取平行坐标图，查看各特征分布

# 绘图观察哪个样本被错判
plt.figure()
plt.title("KNN Classification")
data_order = np.arange(1, 50+1)   # 生成1-50序列，是测试集样本序号，作为绘图的横轴
plt.scatter(data_order, y_test, color='b')  # 训练数据打点
plt.scatter(data_order, y_pre, color='r')  # 测试数据打点
plt.legend(["real", "predict"], loc='right', bbox_to_anchor=(1, 0.25))
plt.xticks(np.arange(1, 50+1, 2))
plt.yticks([0, 1, 2])
plt.grid()  # 加网格
plt.show()
