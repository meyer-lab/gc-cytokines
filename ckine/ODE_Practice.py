# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 13:57:45 2017

@author: alifa
"""

import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def dy_dt(y,t):
    dydt = -y + 1
    return dydt

y0 = 0
t = np.linspace(0,5)
y = odeint(dy_dt, y0, t)

plt.plot(t,y)
plt.xlabel('time')
plt.ylabel('y(t)')
plt.show()
