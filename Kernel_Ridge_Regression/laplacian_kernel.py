# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 10:35:54 2022

@author: Jon

We train a Kernel Regression model with a Laplacian Kernel on a wine quality data set and use the model to predict the quality of a new batch. 
"""

import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
plt.ioff()
import sklearn as sk
from sklearn import model_selection
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


#parameters for cross validation
lapl_sigma_bound = 1
lapl_sigma_step = 10

lapl_alpha_bound = 3 
lapl_alpha_step = 10


sigmas_lapl = np.linspace(5,8, lapl_sigma_step)
alphas_lapl = np.linspace(-8,-5, lapl_alpha_step)

scores_lapl = np.zeros((lapl_sigma_step, lapl_alpha_step)) 
scores_std_lapl = np.zeros((lapl_sigma_step, lapl_alpha_step)) 

print("sigma shape",str(sigmas_lapl.shape))
print("alpha/lambda shape",str(alphas_lapl.shape))


print("score shape ",str(scores_lapl.shape))
print("score std shape", str(scores_std_lapl.shape))


start_time = time.time()
Kernel_ridge_reg_lapl_CV = sk.kernel_ridge.KernelRidge(kernel = "laplacian")


#Cross Validation across (what is sigma?) and (what is alpha?)
for i in range(lapl_sigma_step):

    Kernel_ridge_reg_lapl_CV.gamma = 1/((2**sigmas_lapl[i])) 

    for j in range(lapl_alpha_step): 

        Kernel_ridge_reg_lapl_CV.alpha = (2**alphas_lapl[j])
        lapl_score = sk.model_selection.cross_val_score(Kernel_ridge_reg_lapl_CV, X_train, Y_train, scoring= 'neg_mean_squared_error', cv=10)

        scores_lapl[i,j] = (np.mean(lapl_score))
        scores_std_lapl[i,j] = (np.std(lapl_score))

print("CV Executation time: ", time.time()-start_time)

avl, svl = np.meshgrid(alphas_lapl, sigmas_lapl)

fig_lap, ax_lap = plt.subplots(1,2, figsize=(20,10))

cm0_lapl = ax_lap[0].contourf( avl, svl, np.log2(np.abs(scores_lapl)))
ax_lap[0].set_xlabel('$\log_2\lambda$',fontsize=20)
ax_lap[0].set_ylabel('$\log_2\sigma$',fontsize = 20)
ax_lap[0].set_title('Score',fontsize=20)
fig_lap.colorbar(cm0_lapl, ax=ax_lap[0])

cm1_lapl = ax_lap[1].contourf(avl, svl, np.log2(np.abs(scores_std_lapl)))
ax_lap[1].set_xlabel('$\log_2\lambda$',fontsize = 20)
ax_lap[1].set_ylabel('$\log_2\sigma$',fontsize = 20)
ax_lap[1].set_title('Score std',fontsize=20)
fig_lap.colorbar(cm1_lapl, ax=ax_lap[1])
fig_lap.suptitle("Laplacian Kernel Hyperparamter Cross Validation Score (MSE)",fontsize=30)
#fig_lap.tight_layout()
plt.show()

alpha_lapl = 2**(-6) #Optimal sigma and alpha from cross validation (eyeball norm)
sigma_lapl = 2**(6.5)
gamma_lapl = 1/(sigma_lapl**2)
Kernel_ridge_reg_lapl = sk.kernel_ridge.KernelRidge(kernel ="laplacian",alpha = alpha_lapl,gamma=gamma_lapl)

Kernel_ridge_reg_lapl.fit(X_train,Y_train)

Y_pred_train_lapl = Kernel_ridge_reg_lapl.predict(X_train)
Y_pred_train_lapl = np.round((Y_pred_train_lapl*Y_train_std))+Y_train_mean
Y_train = Y_train*Y_train_std + Y_train_mean
mse_train_err = mean_squared_error(Y_train,Y_pred_train_lapl)
print("Train MSE:",mse_train_err) 
inf_train_err = max(abs(Y_train - Y_pred_train_lapl))
print("Train inf error:",inf_train_err)

Y_pred_test_lapl = Kernel_ridge_reg_lapl.predict(X_test)
Y_pred_test_lapl = np.round((Y_pred_test_lapl*Y_train_std))+Y_train_mean
Y_test = Y_test*Y_test_std + Y_test_mean
mse_test_err = mean_squared_error(Y_test,Y_pred_test_lapl)
print("Test mse:",mse_test_err) 
inf_test_err = max(abs(Y_test - Y_pred_test_lapl))
print("Test inf error:",inf_test_err)

Y_pred_lapl = Kernel_ridge_reg_lapl.predict(X_new)
Y_pred_lapl = np.round((Y_pred_lapl*Y_train_std) + Y_train_mean)
