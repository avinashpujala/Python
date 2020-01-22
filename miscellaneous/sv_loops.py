
#%% Testing some things about sklearn regression model

import numpy as np

dt = 0.0001
t = np.arange(-2*np.pi, 2*np.pi,dt)

f = lambda x:x**2/(x**2)

y = f(t)

plt.plot(t,y,'.-')
plt.xlim(np.min(t),np.max(t))
plt.xlim(-0.01, 0.01)