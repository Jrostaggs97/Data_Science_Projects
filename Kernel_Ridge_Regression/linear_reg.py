# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 12:06:58 2022

@author: Jon
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
plt.ioff()
import sklearn as sk
from sklearn import model_selection
#from sklearn.model_selection import Standard_Scalar
from sklearn import kernel_ridge
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, Ridge
import pandas as pd
import time

#training data
train_data = pd.read_csv("wine_training.csv")
train_data = np.array(train_data)

#Scale to mean 0 and unit variance
X_train =train_data[:,0:10]
X_train_shape = X_train.shape[0]
X_train_mean = np.mean(X_train,axis=0) 
X_train_std = np.std(X_train,axis=0)
X_train = (X_train-np.matlib.repmat(X_train_mean,X_train_shape,1))/np.matlib.repmat(X_train_std,X_train_shape,1)

Y_train = train_data[:,11]
Y_train_shape = Y_train.shape[0]
Y_train_mean = np.mean(Y_train,axis =0)
Y_train_std = np.std(Y_train,axis =0)
Y_train = (Y_train-Y_train_mean)/Y_train_std



#testing data
test_data =pd.read_csv("wine_test.csv")
test_data = np.array(test_data)

#Scale to mean 0 and unit variance
X_test = test_data[0:,0:10]
X_test_shape = X_test.shape[0]
X_test_mean = np.mean(X_test,axis=0) 
X_test_std = np.std(X_test,axis=0)
X_test = (X_test-np.matlib.repmat(X_test_mean,X_test_shape,1))/np.matlib.repmat(X_test_std,X_test_shape,1)


Y_test = test_data[:,11]
Y_test_shape = Y_test.shape[0]
Y_test_mean = np.mean(Y_test,axis =0)
Y_test_std = np.std(Y_test,axis =0)
Y_test = (Y_test-Y_test_mean)/Y_test_std


#New data
new_data = pd.read_csv("wine_new_batch.csv")
new_data = np.array(new_data)
X_new =new_data[:,0:10]
X_new_shape = X_new.shape[0]
X_new_mean = np.mean(X_new,axis=0) 
X_new_std = np.std(X_new,axis=0)
X_new = (X_new-np.matlib.repmat(X_new_mean,X_new_shape,1))/np.matlib.repmat(X_new_std,X_new_shape,1)



lin_reg = LinearRegression().fit(X_train,Y_train)

train_pred = lin_reg.predict(X_train)
train_pred = np.round(train_pred*Y_train_std + Y_train_mean)
Y_train = Y_train*Y_train_std + Y_train_mean
train_mse = mean_squared_error(Y_train,train_pred)

test_pred = lin_reg.predict(X_test)
test_pred = np.round(test_pred*Y_train_std + Y_train_mean)
Y_test = Y_test*Y_test_std + Y_test_mean
test_mse = mean_squared_error(Y_test,test_pred)

new_pred = lin_reg.predict(X_new)
new_pred = np.round(new_pred*Y_train_std + Y_train_mean)