# bagging集成过程:采样，学习，集成
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
#import pydotplus
import os
import matplotlib.pyplot as plt

os.environ["PATH"]+=os.pathsep + 'C:\Program Files\Graphviz/bin'

# 数据以全部列展示
pd.set_option('display.width',1000)
pd.set_option('display.max_columns',None)

# 获取数据
data=pd.read_csv('titanic_data.csv')
data=pd.DataFrame(data)
data.replace(to_replace='male',value=0,inplace=True)
data.replace(to_replace='female',value=1,inplace=True)
print(data)

# 多因子决定是否存活
x=data[['Pclass','Age','Sex']]

x['Age'].fillna(value=data['Age'].mean(),inplace=True)


print(pd.DataFrame(x))
y=data['Survived']

x_onehot=pd.get_dummies(x,columns=['Pclass','Age','Sex'])
# print(pd.DataFrame(x_onehot))
# 数据划分
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=22,test_size=0.25)
# x=x.to_dict(orient='records')
# x_train,x_test=train_test_split(x,random_state=22,test_size=0.25)
print(pd.DataFrame(x_test))

# 特征划分
# transfer=DictVectorizer()
# transfer=StandardScaler()
# x_train=transfer.fit_transform(x_train)
# x_test=transfer.fit_transform(x_test)
# print(x_test)


# 模型训练
estimator=RandomForestClassifier()
estimator.fit(x_train,y_train)
# print(estimator.max_features)

# estimators=estimator.estimators_
# print(estimators)
# for index, model in enumerate(estimators):
#     filename='tree_'+str(index)+'.png'
#     dot_data=tree.export_graphviz(model,feature_names=['Pclass','Age','Sex'])
#     graph = pydotplus.graph_from_dot_data(dot_data)
#     graph.write_png(filename)
#     print(index)
#     print(model)


param_grid={"n_estimators":[120,200,300,500,800,1200],'max_depth':[3,4,5,6,8]}
estimator=GridSearchCV(estimator,param_grid=param_grid,cv=5,n_jobs=-1)
estimator.fit(x_train,y_train)
score=estimator.score(x_test,y_test)
print('随机森林准确率是：\n',score)
y_pre=estimator.predict(x_test)
fig,ax=plt.subplots(figsize=(10,5.4),dpi=80)
ax.scatter(x=np.linspace(0,327,328),y=y_test,s=5,c='blue',alpha=0.7,label='real')
ax.scatter(x=np.linspace(0,327,328),y=y_pre,s=5,c='orange',alpha=0.7, label='predict')
# ax.plot(np.linspace(0,327,328),np.linspace(0,327,328),color='red',linestyle=':',alpha=0.7)
ax.set_title('Output')
ax.legend()
plt.show()

print(estimator)
print(estimator.best_estimator_)



# y_importances=estimator.feature_importances_
# x_importances=transfer.feature_names_
# y_pos=np.arange(len(x_importances))
# plt.barh(y_pos,y_importances,align='center')
# plt.show()



