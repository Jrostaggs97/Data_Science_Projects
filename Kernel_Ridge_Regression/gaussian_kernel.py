# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 10:41:16 2022

@author: Jon

We train a Kernel Regression model with a Gaussian Kernel on a wine quality data set and use the model to predict the quality of a new batch. 
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
import pandas as pd
import time

#training data
train_data = pd.read_csv("wine_training.csv")
train_data = np.array(train_data)

#Scale to mean 0 and unit variance
X_train =train_data[:,0:10]
X_train_shape = X_train.shape[0]
X_train_mean = np.mean(X_train,axis=0) #on axis 0 or axis 1? I'm thinking it's actaully 1?
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


K_sigma_bound = 1
K_sigma_step = 5 #15

K_alpha_bound = 3 
K_alpha_step = 5 #15


sigmas_gauss = np.linspace(0,3, K_sigma_step)
alphas_gauss = np.linspace(-2,0, K_alpha_step)

scores_gauss = np.zeros((K_sigma_step, K_alpha_step)) 
scores_std_gauss = np.zeros((K_sigma_step, K_alpha_step)) 

print("sigma shape",str(sigmas_gauss.shape))
print("alpha/lambda shape",str(alphas_gauss.shape))


print("score shape ",str(scores_gauss.shape))
print("score std shape", str(scores_std_gauss.shape))




#Gaussian kernel cross validation analysis

start_time = time.time()
Kernel_ridge_reg_gauss_CV = sk.kernel_ridge.KernelRidge(kernel = "rbf")



for i in range(K_sigma_step):

    Kernel_ridge_reg_gauss_CV.gamma = 1/(2*(2**sigmas_gauss[i])**2) 

    for j in range(K_alpha_step): 

        Kernel_ridge_reg_gauss_CV.alpha = (2**alphas_gauss[j])
        this_score = sk.model_selection.cross_val_score(Kernel_ridge_reg_gauss_CV, X_train, Y_train, scoring= 'neg_mean_squared_error', cv=5)

        scores_gauss[i,j] = (np.mean(this_score))
        scores_std_gauss[i,j] = (np.std(this_score))


print("CV execuation time: ",(time.time()-start_time))


avg, svg = np.meshgrid(alphas_gauss, sigmas_gauss)

#fig = plt.figure(figsize = (10, 7))
#ax = plt.axes(projection ="3d")

#ax.scatter(sv, lv, np.log(scores))
#ax.set_xlabel('$\sigma$', fontsize=20)
#ax.set_ylabel('$\lambda$', fontsize=20)

fig, ax = plt.subplots(1,2, figsize=(20,10))


cm0 = ax[0].contourf( avg, svg, np.log2(np.abs(scores_gauss)))
ax[0].set_xlabel('$\log_2\lambda$',fontsize =20)
ax[0].set_ylabel('$\log_2\sigma$',fontsize = 20)
ax[0].set_title('Score',fontsize =20)
fig.colorbar(cm0, ax=ax[0])


cm1 = ax[1].contourf(avg, svg, np.log2(np.abs(scores_std_gauss)))
ax[1].set_xlabel('$\log_2\lambda$',fontsize=20)
ax[1].set_ylabel('$\log_2\sigma$',fontsize =20)
ax[1].set_title('Score std',fontsize=20)
fig.colorbar(cm1, ax=ax[1])

fig.suptitle("Gaussian Kernel Cross Validation Score (MSE)",fontsize=30)
plt.show()


# =============================================================================
alpha_gauss = 2**(-1) #These are optimal "alpha and sigma" which we get from the cross validation score
sigma_gauss = 2**(1.5)
gamma_gauss = 1/(2*(2**sigma_gauss)**2)
Kernel_ridge_reg_gauss = sk.kernel_ridge.KernelRidge(kernel ="rbf",alpha = alpha_gauss,gamma=gamma_gauss)

Kernel_ridge_reg_gauss.fit(X_train,Y_train)

#Training errors
Y_pred_gauss_train = Kernel_ridge_reg_gauss.predict(X_train)
Y_pred_gauss_train = np.round((Y_pred_gauss_train*Y_train_std) + Y_train_mean)
Y_train = Y_train*Y_train_std + Y_train_mean
mse_train_err = mean_squared_error(Y_train,Y_pred_gauss_train)
print("Train MSE: ",mse_train_err) 
inf_train_err = max(abs(Y_train - Y_pred_gauss_train))
print("Train inf error:",inf_train_err)


Y_pred_gauss_test = Kernel_ridge_reg_gauss.predict(X_test)
mse_test_err = mean_squared_error(Y_test,Y_pred_gauss_test)
Y_pred_gauss_test = np.round((Y_pred_gauss_test*Y_train_std) + Y_train_mean)
Y_test = Y_test*Y_test_std + Y_test_mean
print("Test MSE: ",mse_test_err) 
inf_test_err = max(abs(Y_test - Y_pred_gauss_test))
print("Test inf error:",inf_test_err)


Y_pred_gauss = Kernel_ridge_reg_gauss.predict(X_new)
Y_pred_gauss = np.round((Y_pred_gauss*Y_train_std + Y_train_mean))