# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 12:34:41 2022

@author: Jon

We use PCA for dimensionality reduction and linear regression for classifying digits from the MNIST dataset. 
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import decomposition
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error

data_train = np.load("MNIST_training_set.npy",allow_pickle=True)

data_test = np.load("MNIST_test_set.npy",allow_pickle=True)

X_train = data_train.item().get("features") #each feature is 16x16 
Y_train = data_train.item().get("labels")


X_test = data_test.item().get("features")
Y_test = data_test.item().get("labels")




#Use PCA to investigate the dimensionality of X-train
X_train = X_train - np.mean(X_train,axis=0) #centered train data
X_test = X_test - np.mean(X_train,axis=0) #centered test
U,S,V_t = np.linalg.svd(X_train)


#plot singular values
# =============================================================================
# plt.plot(S)
# plt.xlabel("index i")
# plt.ylabel("singular value")
# plt.title("singular value vs. index")
# plt.show()
# 
#plt.plot(np.log(S))
#plt.xlabel("index i")
#plt.ylabel("log(singular value)")
#plt.title("log(singular value) vs. index")
#plt.show()
# =============================================================================


N = 256 #number of principle components desired


prince_comp = decomposition.PCA(n_components=N).fit(X_train)

#Plot first 16 PCA modes

def plot_digits(XX, N, title):
    """Small helper function to plot N**2 digits."""
    fig, ax = plt.subplots(N, N, figsize=(8, 8))
    for i in range(N):
      for j in range(N):
        ax[i,j].imshow(XX[(N)*i+j,:].reshape((16, 16)), cmap="Greys")
        ax[i,j].axis("off")
    fig.suptitle(title, fontsize=24)
    

# =============================================================================
# plot_digits(prince_comp.components_.reshape((N,16,16)), 4, "First 16 PCA Modes" )
# 
# #this is just to show labels and plot each mode individually
# pca_modes = prince_comp.components_.reshape((N,16,16))
# for i in range(0,16):
#      plt.pcolormesh(pca_modes[i])
#      plt.xlabel("x")
#      plt.ylabel("y")
#      plt.title("PCA mode i="+str(i+1))
#      plt.show()
# =============================================================================


X_train_proj_trans = prince_comp.fit_transform(X_train) #projects input data onto the new basis

X_train_proj = prince_comp.inverse_transform(X_train_proj_trans) #brings data into original space with reduced dimension



Frob_norm = np.sqrt(sum((prince_comp.singular_values_)**2))

approx_diff = 1-np.sqrt(sum(prince_comp.singular_values_[5:]**2))/Frob_norm #16 gives 60.56%, 38 gives 80.27%, 65 gives 90.018%


Frob_trunc = np.sqrt(sum((prince_comp.singular_values_[:14]**2)))


approx_var = Frob_trunc/Frob_norm


#Reconstructed digits with specified pca modes

# =============================================================================
# prince_comp_check = decomposition.PCA(n_components=14).fit(X_train)
# X_train_proj_trans_check = prince_comp_check.fit_transform(X_train)
# X_train_proj_check = prince_comp_check.inverse_transform(X_train_proj_trans_check)
# 
# 
# plot_digits(X_train_proj_check, 4, "First 16 digits reconstructed w/ 14 modes")
# 
# 
# 
# =============================================================================

#classifying digits with linear regression


#Get indices and labels from training set 
mode_num=3
num1 = 1
num2 = 8
digit_training_list = []
for i in range(0,len(Y_train)):
    if Y_train[i] ==num1:
        digit_training_list.append(i)
    elif Y_train[i] ==num2:
        digit_training_list.append(i)
    else:
        pass

#Y_train_digit = Y_train[digit_training_list]  #no reassign labels  
#update training labels
Y_train_digit = np.zeros(len(digit_training_list))
for i in range(0,len(digit_training_list)):
    if Y_train[digit_training_list[i]] ==num1:
        Y_train_digit[i] = -1
    else:
        Y_train_digit[i] = 1

#separated training features
X_train_digit = X_train[digit_training_list]


#plt.pcolormesh(np.reshape(X_train_digit[4],(16,16)))
#plt.title("should be 8")

#Get indices and labels from test set
digit_test_list = []
for i in range(0,len(Y_test)):
    if Y_test[i] ==num1:
        digit_test_list.append(i)
    elif Y_test[i] ==num2:
        digit_test_list.append(i)
    else:
        pass

#Y_test_digit = Y_test[digit_test_list] #no reassign labels
#update test labels
Y_test_digit = np.zeros(len(digit_test_list))
for i in range(0,len(digit_test_list)):
    if Y_test[digit_test_list[i]] ==num1:
        Y_test_digit[i] = -1
    else:
        Y_test_digit[i] = 1

#separated test features
X_test_digit = X_test[digit_test_list]



#Project this onto first 16 PCA basis computed above 
prince_comp_digit_train = decomposition.PCA(n_components = mode_num).fit(X_train)
X_train_digit_proj_trans = prince_comp_digit_train.fit_transform(X_train_digit) #projection onto basis- this is A matrix in hw
#X_train_digit_proj = prince_comp_digit_train.inverse_transform(X_train_digit_proj_trans) #bring back into og basis with reduced dimension

reg = LinearRegression().fit(X_train_digit_proj_trans,Y_train_digit) #training linear regression

#Ridge Regression
#ridge_train= Ridge(alpha =1)
#ridge_train = ridge_train.fit(X_train_digit_proj_trans,Y_train_digit) #trainign ridge regression

#Y_train_pred_ridge= np.dot(X_train_digit_proj_trans,ridge_train.coef_) + ridge_train.intercept_ #Training Y with Ridge
#MSE_training_ridge = sk.metrics.mean_squared_error(Y_train_digit,Y_train_pred_ridge) #Ridge Regression Mean Square Error

Y_train_pred = np.matmul(X_train_digit_proj_trans,reg.coef_) + reg.intercept_ #Training Y with linear regression
for i in range(0,len(Y_train_pred)):
    if Y_train_pred[i]>0:
        Y_train_pred[i] = 1
    else:
        Y_train_pred[i] = -1

np.set_printoptions(precision=16)
MSE_training_linreg = np.float(sk.metrics.mean_squared_error(Y_train_digit,Y_train_pred)) #Linear regression Mean square Error


#prince_comp_digit_test = decomposition.PCA(n_components = 16)
X_test_digit_proj_trans = prince_comp_digit_train.transform(X_test_digit)

print(MSE_training_linreg)


Y_test_pred = np.matmul(X_test_digit_proj_trans,reg.coef_) + reg.intercept_
for i in range(0,len(Y_test_pred)):
    if Y_test_pred[i]>0:
        Y_test_pred[i] = 1
    else:
        Y_test_pred[i] = -1

MSE_test_linreg = sk.metrics.mean_squared_error(Y_test_digit,Y_test_pred)
    
    


















