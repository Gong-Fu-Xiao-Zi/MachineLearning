import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score

pd.set_option('display.width',1000)
pd.set_option('display.max_columns',None)

plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False

# 数据读取
data=pd.read_csv('heart_improve.csv')
data=pd.DataFrame(data)
print(data)
# data=data.dropna()
# print(data)
# 空白数据处理
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


# 数据特征展示
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
#plt.savefig('D:\Python pro\exercise\pythonProject1/4.Logic Regerssion/figure1')
plt.show()



# 数据划分
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=22)
# 实例化转换器
transfer=StandardScaler()
x_train=transfer.fit_transform(x_train)
x_test=transfer.fit_transform(x_test)
# 机器学习——逻辑回归
estimator=LogisticRegression()
estimator.fit(x_train,y_train)
# 模型评估
y_pre=estimator.predict(x_test)
print("预测值是：\n",y_pre)

score=estimator.score(x_test,y_test)
print("准确率是：\n",score)
fig,axs=plt.subplots(1,2,figsize=(10,5.4),dpi=120)
# axs[0].plot(y_test,color='lime',label='Real',linestyle='-.')
# axs[0].plot(y_pre,color='crimson',label='Predict',linestyle='--')
# sns.stripplot(x=np.linspace(0,1059,1060),y=y_test,c='crimson',s=6,ax=axs[0])
axs[0].scatter(np.linspace(0,1059,1060),y_test,c='orange',s=0.5,label='real',alpha=0.75)
# axs[0].set_title('Real Situation')
axs[0].scatter(np.linspace(0,1059,1060),y_pre,c='blue',s=2,label='predict',alpha=0.75)
axs[0].set_title('Real and Predict')
axs[0].legend()

plt.tight_layout()
#plt.savefig('D:\Python pro\exercise\pythonProject1/4.Logic Regerssion/figure2')
plt.show()



ret=classification_report(y_test,y_pre,labels=(0,1),target_names=("健康","亚健康"))
print(ret)

# ROC-AUC评估方法
score1=roc_auc_score(y_true=y_test,y_score=y_pre)
print(score1)

