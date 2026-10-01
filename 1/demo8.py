import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(8, 2.7), layout='constrained')
t=np.arange(0,5,0.01)
s=np.sin(2*np.pi*t)
l1, = ax1.plot(t, s)
ax2 = ax1.twinx()
l2, = ax2.plot(t, range(len(t)), color='orange')
ax12=ax1.twinx()
l3, =ax12.plot(t,t**2,color='#87CEFA')
print(range(len(t)))
ax2.legend([l1, l2, l3], ['Sine (left)', 'Straight (right)','HHHHH'])
ax3.plot(t, s)
ax3.set_xlabel('Angle [°]')
ax4 = ax3.secondary_xaxis('top', functions=(np.rad2deg, np.deg2rad))
ax4.set_xlabel('Angle [rad]')
plt.show()