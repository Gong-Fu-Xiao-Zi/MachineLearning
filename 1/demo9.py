import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

# 数据准备
X, Y = np.meshgrid(np.linspace(-3, 3, 128), np.linspace(-3, 3, 128))
print(X)
print(Y)
Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2)
data1,data2,data3=np.random.randn(3,150)
print(data1)

# 创建画布
fig, axs = plt.subplots(2, 2,figsize=(10,6), dpi=80);'''绘制2*2四个图'''

# 图像绘制，注意多图时子图索引顺序
pc = axs[0, 0].pcolormesh(X, Y, Z, vmin=-1, vmax=1, cmap='RdBu_r');        '''第一个图，热力图'''
fig.colorbar(pc, ax=axs[0, 0])
axs[0, 0].set_title('pcolormesh()')

co = axs[0, 1].contourf(X, Y, Z, levels=np.linspace(-1.25, 1.25, 11));         '''第二个图，等高线图，添加颜色依旧是用cmap=rainbow'''
fig.colorbar(co, ax=axs[0, 1])
axs[0, 1].set_title('contourf()')

pc = axs[1, 0].imshow(Z**2 * 100, cmap='plasma',
                          norm=mpl.colors.LogNorm(vmin=0.01, vmax=100));     '''第三个图，热图'''
fig.colorbar(pc, ax=axs[1, 0], extend='both')
axs[1, 0].set_title('imshow() with LogNorm()')

pc = axs[1, 1].scatter(data1, data2, c=data3, cmap='RdBu_r');                '''散点图'''
fig.colorbar(pc, ax=axs[1, 1], extend='both')
axs[1, 1].set_title('scatter()');


#plt.savefig('D:\Python pro\exercise\pythonProject1/0.Mat---Picture/figure3.png')
plt.savefig('G:\picture/figure3.png')
plt.show()