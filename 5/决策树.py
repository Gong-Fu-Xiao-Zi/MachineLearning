import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn import tree
#import pydotplus
import os


os.environ["PATH"]+=os.pathsep + r'C:\Program Files\Graphviz/bin'

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',None)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False


data=pd.read_csv('heart_improve.csv')
data=pd.DataFrame(data)
print(data)

# data=data.dropna()
# print(data)
data['glucose'].fillna(value=data['glucose'].mean(),inplace=True)
data['heartRate'].fillna(value=data['heartRate'].mean(),inplace=True)
data['education'].fillna(value=data['education'].mean(),inplace=True)
data['cigsPerDay'].fillna(value=data['cigsPerDay'].mean(),inplace=True)
data['BPMeds'].fillna(value=data['BPMeds'].mean(),inplace=True)
data['totChol'].fillna(value=data['totChol'].mean(),inplace=True)
data['BMI'].fillna(value=data['BMI'].mean(),inplace=True)
# print(data)

x=data.iloc[:,0:-1]
y=data['TenYearCHD']

color_map={0:'yellow',1:'red'}
color1=list(map(lambda x:color_map[x],data['TenYearCHD']))



fig,axs=plt.subplots(4,4,figsize=(20,10.8),dpi=70)

axs[0,0].scatter(np.linspace(0,4237,4238),x['male'],color=color1,s=3)
# sns.stripplot(x=np.linspace(0,4237,4238),y=x['male'],jitter=True,ax=axs[0,0])
axs[0,0].set_title('male')
# axs[0,1].scatter(x['age'],y,color=color1)
axs[0,1].scatter(np.linspace(0,4237,4238),x['age'],color=color1,s=3)
axs[0,1].set_title('age')
# axs[0,2].scatter(x['education'],y,color=color1)
axs[0,2].scatter(np.linspace(0,4237,4238),x['education'],color=color1,s=6)
axs[0,2].set_title('education')
# axs[0,3].scatter(x['currentSmoker'],y,color=color1)
axs[0,3].scatter(np.linspace(0,4237,4238),x['currentSmoker'],color=color1,s=4)
axs[0,3].set_title('currentSmoker')
# axs[1,0].scatter(x['cigsPerDay'],y,color=color1)
axs[1,0].scatter(np.linspace(0,4237,4238),x['cigsPerDay'],color=color1,s=4)
axs[1,0].set_title('cigsPerDay')
# axs[1,1].scatter(x['BPMeds'],y,color=color1)
axs[1,1].scatter(np.linspace(0,4237,4238),x['BPMeds'],color=color1,s=5)
axs[1,1].set_title('Blood Pressure Medicine服药史')
# axs[1,2].scatter(x['prevalentStroke'],y,color=color1)
axs[1,2].scatter(np.linspace(0,4237,4238),x['prevalentStroke'],color=color1,s=5)
axs[1,2].set_title('prevalentStroke中风史')
# axs[1,3].scatter(x['prevalentHyp'],y,color=color1)
axs[1,3].scatter(np.linspace(0,4237,4238),x['prevalentHyp'],color=color1,s=5)
axs[1,3].set_title('prevalentHyp高血压')
# axs[2,0].scatter(x['diabetes'],y,color=color1)
axs[2,0].scatter(np.linspace(0,4237,4238),x['diabetes'],color=color1,s=5)
axs[2,0].set_title('diabetes糖尿病')
# axs[2,1].scatter(x['totChol'],y,color=color1)
axs[2,1].scatter(np.linspace(0,4237,4238),x['totChol'],color=color1,s=3)
axs[2,1].set_title('totChol总胆固醇')
# axs[2,2].scatter(x['sysBP'],y,color=color1)
axs[2,2].scatter(np.linspace(0,4237,4238),x['sysBP'],color=color1,s=3)
axs[2,2].set_title('sysBP收缩压')
# axs[2,3].scatter(x['diaBP'],y,color=color1)
axs[2,3].scatter(np.linspace(0,4237,4238),x['diaBP'],color=color1,s=3)
axs[2,3].set_title('diaBP扩张压')
# axs[3,0].scatter(x['BMI'],y,color=color1)
axs[3,0].scatter(np.linspace(0,4237,4238),x['BMI'],color=color1,s=3)
axs[3,0].set_title('BMI')
# axs[3,1].scatter(x['heartRate'],y,color=color1)
axs[3,1].scatter(np.linspace(0,4237,4238),x['heartRate'],color=color1,s=3)
axs[3,1].set_title('heartRate心率')
# # axs[3,2].scatter(x['glucose'],y,color=color1)
axs[3,2].scatter(np.linspace(0,4237,4238),x['glucose'],color=color1,s=3)
axs[3,2].set_title('glucose血糖')
axs[3,3].scatter(y,y,color=color1)
plt.tight_layout()
#plt.savefig('D:\Python pro\exercise\pythonProject1/5. DecisionTree/feature.png')
plt.show()

# 数据分割
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=22,test_size=0.25)
# 实例化
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.fit_transform(x_test)

# 机器学习——实例化估计器
estimator=DecisionTreeClassifier(max_depth=8)
estimator.fit(x_train,y_train)
dot_data=tree.export_graphviz(estimator,feature_names=['male','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose'])
#graph=pydotplus.graph_from_dot_data(dot_data)
#graph.write_png('tree1.png')
# 模型评估
y_pre=estimator.predict(x_test)
print('预测值是：\n',y_pre)
score=estimator.score(x_test,y_test)
print('准确率是：\n',score)
ret=classification_report(y_test,y_pre,labels=(0,1),target_names=("健康","亚健康"))
print(ret)

# x_onehot=pd.get_dummies(x,columns=['male','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose'])
# print(x_onehot)
print(estimator.feature_importances_)

fig,axs=plt.subplots(1,2,figsize=(10,5.4),dpi=80)
axs[0].scatter(x=np.linspace(0,1059,1060),y=y_test,s=3,c='dodgerblue',label='True')
axs[0].scatter(x=np.linspace(0,1059,1060),y=y_pre,s=3,c='crimson',label='Predict',alpha=0.6)
axs[0].legend()
axs[0].set_title('Output')
axs[1].barh(range(15),estimator.feature_importances_,align='center',height=0.2,alpha=0.6)
axs[1].set_title('Factors')
axs[1].set_yticks(range(15),['male','age','education','currentSmoker','cigsPerDay','BPMeds','prevalentStroke','prevalentHyp','diabetes','totChol','sysBP','diaBP','BMI','heartRate','glucose'])
plt.tight_layout()
#plt.savefig('D:\Python pro\exercise\pythonProject1/5. DecisionTree/feature_importances.png')
plt.show()



# estimator=RandomForestClassifier()
# param_grid={"n_estimators":[120,200,300,500,800,1200],'max_depth':[5,8,15,25,30]}
# estimator=GridSearchCV(estimator,param_grid=param_grid,cv=5)
# estimator.fit(x_train,y_train)
# print(estimator)
# print(estimator.best_estimator_)
# 根据随机森林的结果分析，最优深度是8，n_estimators=120

# x=x.to_dict(orient='records')
# x_train,x_test=train_test_split(x,random_state=22,test_size=0.25)
# # print(x_train)
# # print(x)
# transfer=DictVectorizer()
# x_train=transfer.fit_transform(x_train)
# x_test=transfer.fit_transform(x_test)


