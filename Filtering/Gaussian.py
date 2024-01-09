# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 14:18:03 2022

@author: Jon
"""
import numpy as np

def Gaussian(x,y,z,x_center,y_center, z_center,sigma):
    
    x_0 = x - (x[x_center][y_center][z_center])
    y_0 = y - (y[y_center][y_center][z_center])
    z_0 = z - (z[z_center][y_center][z_center])
    
    gauss = (1/sigma*(2*np.pi)**(1/2))*np.exp(-(x_0**2 + y_0**2 + z_0**2)/(2*sigma**2))
    
    return gauss