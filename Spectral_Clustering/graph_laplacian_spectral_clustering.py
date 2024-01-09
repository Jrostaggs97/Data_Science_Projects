# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 11:55:21 2022

@author: Jon

Using Graph Laplacian to determine clusters within our data set and predict the political parties of congress-people based on previous voting data.
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
import sklearn as sk
from sklearn import decomposition
from sklearn.decomposition import pca
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import scipy as sp
import scipy.spatial

#read in data and give header row with labels, pandas data frame
XYdf = pd.read_csv('house-votes-84.data', sep=',', header=None)
XYdf = XYdf.replace({0:{"republican":-1,"democrat":1}})
XYdf = XYdf.replace("y",1)
XYdf = XYdf.replace("n",-1)
XYdf = XYdf.replace("?",0)

XYdata = XYdf.to_numpy()
Y = XYdata[:,0]
X = XYdata[0:,1:17]

#next we move to spectral clustering where we work with X and use Y for validation


#construct the unnormalized graph Laplacian matrix on X using the weight function
#eta(t) = exp(-t^2/(2sigma^2)) with variance sigma and compute its second eigenvec
#ie the Fiedler vector which we denote as q_1

#distance matrix
distance = sp.spatial.distance_matrix(X,X) #Minkowski p-norm
sigma = .05*distance.mean() #variance


#weight function
def eta(t,sigma):
    
    val = np.exp((-t**2)*(1/(2*(sigma**2))))
    
    return val


#some parameters for building graph laplacian
m=1
N = 10**m


# =============================================================================
sig_list = np.linspace(0,4,4*N+1)
cluster_acc_list = np.zeros((len(sig_list)-1))
mis_calc =np.zeros((len(Y),1))
for i in sig_list[1:len(sig_list)]:
    W = eta(distance,np.round(i,m)) #weight matrix, is this supposed to be just i or i*distance_mean()
    deg_vec = np.sum(W,axis=1) #degree vector
    D = np.diag(deg_vec) #Diagonal Degree Matrix
    L = D-W #Unnormalized Graph Laplacian Matrix
    eig_vals,eig_vecs = np.linalg.eigh(L)
    fiedler_vec = eig_vecs[:,1]
    
    Y_pred = np.sign(fiedler_vec)
    
    mis_calc_sum = sum(Y_pred!=Y)            
    cluster_acc = 1-(1/len(Y))*mis_calc_sum
    cluster_acc_list[int(np.round(i*N,m)-1)] = cluster_acc
    #print(int(np.round(i*N,m)-1))


opt_sig_acc = np.max(cluster_acc_list[np.nonzero(cluster_acc_list)])
opt_sig_loc = np.min(np.where(cluster_acc_list == opt_sig_acc))+1
opt_sig = sig_list[opt_sig_loc]    #opt_sig = 1.159


#Graph Laplacian Computation with optimal sigma    
# =============================================================================
opt_sig = 1.159 #Why are we not getting this value from opt_sig above?
W = eta(distance,opt_sig) #weight matrix, is this supposed to be just i or i*distance_mean()
deg_vec = np.sum(W,axis=1) #degree vector
D = np.diag(deg_vec) #Diagonal Degree Matrix
L = D-W #Unnormalized Graph Laplacian Matrix
eig_vals,eig_vecs = np.linalg.eigh(L)
fiedler_vec = eig_vecs[:,1]
individs = np.linspace(1,435,435)
Y_sc = np.sign(fiedler_vec)
indx_neg_1 = np.argwhere(Y==-1)
indx_1 = np.argwhere(Y==1)

indx = np.append(indx_neg_1,indx_1,axis=0).flatten()

X_spec_cluster_plot = X[indx,:]
Y_spec_cluster_plot_neg = Y[indx_neg_1]
Y_spec_cluster_plot_pos = Y[indx_1]

repub_count = np.linspace(1,len(Y_spec_cluster_plot_neg),len(Y_spec_cluster_plot_neg))
dem_count = np.linspace(len(repub_count),len(repub_count)+len(Y_spec_cluster_plot_pos),len(Y_spec_cluster_plot_pos))
zero_append = np.zeros((1,len(repub_count)))

repub_called_dem_sc = []
dem_called_repub_sc =[]
for k in range(1,len(Y)):
    if Y_sc[k] == -1 and Y[k] == 1: #predicted as repub when really dem
        dem_called_repub_sc.append(1)
    elif Y_sc[k] == 1 and Y[k] ==-1: #predicted as dem when really repub
        repub_called_dem_sc.append(1)
    else:
        continue
        
misident_repub_sc = sum(repub_called_dem_sc)
midident_dem_sc = sum(dem_called_repub_sc)


#Y_spec_cluster_plot_pos = np.append(zero_append,Y_spec_cluster_plot_pos)

#Y_spec_cluster_pred_plot = np.sign(fiedler_vec[indx])
#r_scat = plt.scatter(repub_count,Y_spec_cluster_plot_neg,s=35,c="r",marker="o")
#d_scat = plt.scatter(dem_count,Y_spec_cluster_plot_pos,s=35,c="b",marker="o")
#spec_clust_scat = plt.scatter(individs,Y_spec_cluster_pred_plot,s=10,c="k",marker="x")
#plt.xlabel("Sorted Individual Index")
#plt.ylabel("Party")
#plt.title("Predicted Party - Spectral Clustering")

#plt.legend([r_scat,d_scat,spec_clust_scat],["Republicans","Democrats","Predicted Party"])
#plt.show()

# =============================================================================

#Plotting for sigma vs cluster accuracy
# =============================================================================
#plt.plot(sig_list[1:len(sig_list)],cluster_acc_list)
#plt.xlabel("$\sigma$")
#plt.ylabel("Cluster Accuracy")
#plt.title("Cluster Accuracy Vs. $\sigma$")
#plt.show()
#
# =============================================================================
#Ds = np.diag(1/np.sqrt(deg_vec))
#L_N= np.dot(Ds,np.dot((D-W),Ds)) #normalized graph laplacian

#eig_vals_N,eig_vecs_N = np.linalg.eigh(L_N)
#fiedler_vec_N = eig_vecs[:,1]
#Y_pred_N = np.sign(fiedler_vec_N)


#Plotting unnormalized and normalized eigenvalues (log scale)
# =============================================================================
# fig,ax = plt.subplots(1,2, figsize=(16,8))
# 
# ax[0].plot(np.log(eig_vals))
# ax[0].set_title("Unnormalized eigenvalue, $\lambda$, vs. diagonal index (i,i)")
# ax[0].set_xlabel("index, i")
# ax[0].set_ylabel("$\lambda$")
# 
# ax[1].plot(np.log(eig_vals_N))
# ax[1].set_xlabel("index, i")
# ax[1].set_ylabel("$\lambda$")
# ax[1].set_title("Normalized eigenvalue, $\lambda_N$, vs. diagonal index (i,i)")
# 
# plt.show()
# 
# =============================================================================

#Plotting sign(Fiedler Vector) for normalized and unnormalized Laplacian (bad graphs though)
# =============================================================================
# fig,ax = plt.subplots(1,2,figsize=(16,8))
# ax[0].scatter(X[:,0],X[:,1],c = np.sign(eig_vecs[:,1]),cmap="jet")
# ax[0].set_xlabel("x_1")
# ax[0].set_ylabel("x_2")
# ax[0].set_title("sign(Fiedler Vector) of Unnormalized Laplacian")
# 
# ax[1].scatter(X[:,0],X[:,1],c = np.sign(eig_vecs_N[:,1]),cmap="jet")
# ax[1].set_xlabel("x_1")
# ax[1].set_ylabel("x_2")
# ax[1].set_title("sign(Fiedler Vector) of Normalized Laplacian")
# plt.show()
# =============================================================================




#number of eigenvectors to use to approximate the classifier

MJ_acc_list = [];
for M in [2,3,4,5,6]:
    for J in [5,10,20,40,60]:
        #M= observed features..
        F_embed  =eig_vecs[:,0:M] #Laplacian embedding with only M observables
        #J = low rank approximation
        F_sub_mat = F_embed[0:J,:]
        #eig_vecs_cut = eig_vecs[0:M, 0:low_rank] #[0:M,0:K]
        
        Y_cut = Y[:J]
        
        lap_lin_reg = sk.linear_model.LinearRegression(fit_intercept = False).fit(F_sub_mat,Y_cut)
        beta_hat = lap_lin_reg.coef_
        ycheck = np.dot(F_embed,beta_hat)
        y_hat = np.sign(np.matmul(F_embed,beta_hat))
        mis_calc_sum =sum(y_hat!=Y)
        ssl_acc = 1-(mis_calc_sum/len(Y))
        tup = (M,J,mis_calc_sum,ssl_acc)
        MJ_acc_list.append(tup)
        
        
        
        
F_embed_opt = eig_vecs[:,0:2]
F_sub_mat_opt = F_embed_opt[0:5,:]

Y_cut_opt = Y[:5]
lap_lin_reg_opt = sk.linear_model.LinearRegression(fit_intercept = False).fit(F_sub_mat_opt,Y_cut_opt)
beta_hat_opt = lap_lin_reg_opt.coef_
ycheck_opt = np.dot(F_embed_opt,beta_hat_opt)
y_hat_opt = np.sign(np.matmul(F_embed_opt,beta_hat_opt))

repub_called_dem_ssl = []
dem_called_repub_ssl =[]
for k in range(1,len(Y)):
    if y_hat_opt[k] == -1 and Y[k] == 1: #predicted as repub when really dem
        dem_called_repub_ssl.append(1)
    elif y_hat_opt[k] == 1 and Y[k] ==-1: #predicted as dem when really repub
        repub_called_dem_ssl.append(1)
    else:
        continue
        
misident_repub_ssl = sum(repub_called_dem_ssl)
midident_dem_ssl = sum(dem_called_repub_ssl)

y_hat_ssl_plot = y_hat_opt[indx]

r_scat = plt.scatter(repub_count,Y_spec_cluster_plot_neg,s=35,c="r",marker="o")
d_scat = plt.scatter(dem_count,Y_spec_cluster_plot_pos,s=35,c="b",marker="o")
y_hat_scat = plt.scatter(individs,y_hat_ssl_plot,s  =10,c="k",marker = "x")
plt.xlabel("Sorted Individual Index")
plt.ylabel("Party")
plt.title("Predicted Party - Semi-Supervised Learning")
plt.legend([r_scat,d_scat,y_hat_scat],["Republicans","Democrats","Predicted Party"])
plt.show()