import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

data1=np.random.randn(150)
print(data1)
fig, axs = plt.subplots(1, 3, figsize=(5, 2.7), layout='constrained');  '''绘制多个表格时，ax要变成axs'''
xdata = np.arange(len(data1))  # make an ordinal for this
data = 10**data1;   '''乘方运算'''
data0=10**(data1*2)




axs[0].plot(xdata, data,color='red',linewidth=0.8)


axs[1].set_yscale('log')
axs[1].plot(xdata, data,color='black');

axs[2].set_yscale('log')
axs[2].plot(xdata,data0,color='blue')
plt.show()