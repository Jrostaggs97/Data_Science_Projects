# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 12:34:41 2022

@author: Jon
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk
from sklearn import decomposition
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler

data_train = np.load("MNIST_training_set.npy",allow_pickle=True)

data_test = np.load("MNIST_test_set.npy",allow_pickle=True)

X_train = data_train.item().get("features") #each feature is 16x16 
Y_train = data_train.item().get("labels")

#print(X_train.shape)
#print(Y_train.shape)

X_test = data_test.item().get("features")
Y_test = data_test.item().get("labels")

#print(X_test.shape)
#print(Y_test.shape)

#Plotting original image example
#plt.pcolormesh(np.reshape(X_train[0],(16,16)))

#Use PCA to investigate the dimensionality of X-train


#plot singular values
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
U,S,V_t = np.linalg.svd(X_train)
plt.plot(S)
plt.xlabel("index i")
plt.ylabel("s_i")
plt.title("s_i vs. i")
plt.show()

plt.plot(np.log(S))
plt.xlabel("index i")
plt.ylabel("log(s_i)")
plt.title("log(s_i) vs. i")
plt.show()


#finding and plotting the first 16 principal components/pca modes
prince_comp = decomposition.PCA(n_components=16).fit(X_train) 

pca_modes = prince_comp.components_.reshape((16,16,16))
#for i in range(0,16):
    #plt.pcolormesh(pca_modes[i])
    #plt.xlabel("x")
    #plt.ylabel("y")
    #plt.title("PCA mode i="+str(i+1))
    #plt.show()

X_train_proj_trans = prince_comp.fit_transform(X_train) #projects input data onto the new basis

X_train_proj = prince_comp.inverse_transform(X_train_proj_trans) #brings data into original space

check = np.reshape(X_train_proj[4],(16,16))


plt.pcolormesh(np.reshape(X_train[4],(16,16)))
plt.title("original")
plt.show()

plt.pcolormesh(check)
plt.title("reduced?")
plt.show()

# =============================================================================
# approx_rank = 5
# 
# sum = 0 #initialize sum
# 
# for i in range(0,approx_rank+1):
#     #grab column of u 
#     u_column = U[:,i]
#     
#     #grab column of v and conjugate transpose
#     v_star_column = V_t[:,i]
# 
#     #singular value
#     singular_value = S[i]
# 
#     sum = sum +singular_value*(np.dot(u_column,v_star_column))
# =============================================================================






































