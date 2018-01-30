#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 18:54:58 2018

@author: joe
"""

import numpy as np
import matplotlib.pyplot as plt

number_sample = 100000
inner_area,outer_area = 0,0
inner_X = []
inner_Y = []
outer_X = []
outer_Y = []
for i in range(number_sample):
    x = np.random.uniform(0,1)
    y = np.random.uniform(0,1)
    #print(x)
    
    if (x**2 + y**2) < 1 :
        inner_area += 1
        inner_X = np.append(inner_X, x)
        inner_Y = np.append(inner_Y, y)
    else:
        outer_X = np.append(outer_X, x)
        outer_Y = np.append(outer_Y, y)
    outer_area += 1
    
    
print("The computed value of Pi:",4*(inner_area/float(outer_area))) 
#print(X)
#print(Y)
plt.plot(inner_X,inner_Y, 'bo')
plt.plot(outer_X, outer_Y, 'ro')
plt.show()
