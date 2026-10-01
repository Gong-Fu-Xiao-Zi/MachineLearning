import numpy as np
import matplotlib.pyplot as plt

data1=np.random.randn(150)
xdata=np.arange(len(data1))
fig, axs = plt.subplots(2, 1, layout='constrained');    '''绘制图表，2个竖向分布的表'''
axs[0].plot(xdata, data1);            '''第一个表的绘制，数值对应，线条颜色等均是默认'''
axs[0].set_title('Automatic ticks');     '''表一题目'''


axs[1].plot(xdata, data1)
axs[1].set_xticks(np.arange(0, 100, 30), ['zero', '30', 'sixty', '90']);    '''自定义坐标刻度标尺'''
axs[1].set_yticks([-1.5, 0, 1.5])  # note that we don't need to specify labels
axs[1].set_title('Manual ticks');

plt.show()