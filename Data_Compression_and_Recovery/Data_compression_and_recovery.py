# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 19:20:03 2022

@author: Jon

We compress/corupt image data by varying levels and use sparse regression to reconstruct the image. 
"""

import numpy as np
import matplotlib.pyplot as plt
plt.ioff()
import sklearn as sk
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
import pandas as pd
import scipy as sp
import scipy.spatial
import cvxpy as cp
import skimage as ski
import skimage.io
import skimage.transform
import scipy.fftpack as spfft #for discrete cosine transform
from numpy import random

img_path = "SonOfMan.png" #Image Path
img_og = ski.io.imread(img_path)
#convert to grayscale
img_g = ski.color.rgb2gray(img_og)



img = ski.transform.rescale(img_g,.18,anti_aliasing=False)
plt.imshow(img,cmap="gray")
plt.title("Downsized Gray Scale Version of 'Son of Man' Image")
plt.xticks([])
plt.yticks([])
plt.show()

def construct_DCT_Mat(Nx,Ny):
    
    #input: Nx number of columns of image
    #       Ny number of rows of image
    
    #output: (Discrete Cosine Transform) DCT matrix, D,  mappind image.flatten to DCT(image).flatten()
    
    Dx = spfft.dct(np.eye(Nx), axis=0,norm="ortho")
    Dy = spfft.dct(np.eye(Ny),axis=0,norm="ortho")
    D = np.kron(Dy,Dx)
    
    return D

#construct inverse DCT matrix
def construct_iDCT_Mat(Nx,Ny):
    #input: Nx number of columns of image
    #       Ny number of rows of image
    
    #output: iD, iDCT matrix mapping DCT(image).flatten() to image.flatten()

    Dx = spfft.idct(np.eye(Nx), axis =0, norm="ortho")
    Dy = spfft.idct(np.eye(Ny), axis = 0, norm="ortho")
    D = np.kron(Dy,Dx)
    
    return D

Nx = img.shape[0]
Ny = img.shape[1]
N = Nx*Ny

F= img
vec_F = F.flatten()

D = construct_DCT_Mat(Nx,Ny) #forward transform

dct_f = np.matmul(D,vec_F) #dct of vector

#computing amount of coefficients that are large in magnitude to explore compressability
# =============================================================================
# dct_mean = np.mean(abs(dct_f)) #mean of dct coefficients
# dct_std = np.std(abs(dct_f)) #standard dev of dct coefficients
# large_lb = -2*dct_std+dct_mean #2 std devs below mean
# large_ub = 2*dct_std+dct_mean #2 std devs above mean
# bigger_coefs_idx= np.where(abs(dct_f)>=large_ub) #indices of
# lower_coefs_idx = np.where(abs(dct_f)<=large_lb) #indices of | |
# big_coefs = dct_f[bigger_coefs_idx] #value of coefficients below 2 std devs
# low_coefs = dct_f[lower_coefs_idx] #value of coefficients above 2 std devs
# 
# #A historgram of the coefficients would be cool or log plot
# n,bins,patches = plt.hist(np.sort(abs(dct_f)),100)
# plt.title("Histogram of Coefficient Magnitudes for DCT of Image Vector")
# plt.ylabel("Amount of Coeffs.")
# plt.xlabel("Coeff. Magnitudes")
# plt.xlim([0,5])
# plt.show()
# =============================================================================



#grab top P percent of DCT coefficients
imgs_holder = [np.zeros((53,41))]
coeff_kept_count = [(0,0)]
for P in [95,90,80,60]:
    percentile = np.percentile(abs(dct_f),P)
    dct_f_percentile_vals_idx = np.where(abs(dct_f)>=percentile)
    coeff_kept_count.append((100-P,np.shape(dct_f_percentile_vals_idx)[1]))
    dct_f_percentile_vals = dct_f[dct_f_percentile_vals_idx]
    
    #reconstruct image with top P percent of DCT coeffs
    dct_f_percentile = np.zeros((len(dct_f),))
    dct_f_percentile[dct_f_percentile_vals_idx] = dct_f_percentile_vals
    
    #inverse transform
    iD = construct_iDCT_Mat(Nx,Ny)
    
    #plot the reconstructed images, need do to fig thin
    reconstruct_img_vec = np.matmul(iD,dct_f_percentile)
    reconstruct_img = np.reshape(reconstruct_img_vec,(Nx,Ny))
    imgs_holder.append(reconstruct_img)

#Plotting reconstructed images from top P percentile of DCT coeffs
# =============================================================================
#fig,ax = plt.subplots(2,2,figsize=(9,9))
#P_list = [5,10,20,40]
#for i in [0,1]:
#    for j in [0,1]:
#       ax[i,j].imshow(imgs_holder[2*i+j+1],cmap="gray")
#       ax[i,j].set_title("P = " + str(P_list[2*i+j]),fontsize=15)
#       ax[i,j].set_xticks([])
#       ax[i,j].set_yticks([]) 
#
#fig.suptitle("Reconstructed Images from top P percentile of DCT Coeffs.",fontsize=20)
#plt.show()
# =============================================================================
    
#Mystery Image
# =============================================================================
#mystery_y = np.load("y.npy")
#mystery_B = np.load("B.npy")
#m_Nx = mystery_B.shape[0]
#m_Ny= mystery_B.shape[1]
#m_N = m_Nx*m_Ny
#mystery_iD = construct_iDCT_Mat(int(np.sqrt(m_Ny)),int(np.sqrt(m_Ny)))#what should the column input sizes be?

#m_A = np.matmul(mystery_B,mystery_iD) #these values are not lining up?
#m_x = cp.Variable(m_Ny)
#m_objective = cp.Minimize(cp.norm(m_x,1))
#m_constraint = [np.matmul(m_A,m_x)==mystery_y]
#m_prob = cp.Problem(m_objective,m_constraint)
#
#m_opt_val = m_prob.solve(verbose=True,solver="CVXOPT",max_iter=1000,reltol=1e-2,featol=1e-2)
#m_opt_x = m_x.value
#
#m_sparse_reconstruct_img_vec = np.matmul(mystery_iD,m_opt_x)
#m_sparse_reconstruct_img =np.reshape(m_sparse_reconstruct_img_vec,(int(np.sqrt(m_Ny)),int(np.sqrt(m_Ny))))
#plt.imshow(m_sparse_reconstruct_img,cmap = "gray")
#plt.title("Compressed Image Recovery of Nyan Cat")
#plt.xticks([])
#plt.yticks([])
#plt.show()
# 
# =============================================================================

imgs_sparse_holder = [np.zeros(img.shape)]
r_list = [.2,.4,.6]
sparse_idct = construct_iDCT_Mat(Nx,Ny)
D = construct_DCT_Mat(Nx,Ny)
iD = construct_iDCT_Mat(Nx, Ny)

for r in [.2,.4,.6]:
    M = round(r*N)
    for iter_count in [0,1,2]:
        print(r)
        
        #want m random number pulled from 0 to N
        
        B_rows = random.randint(N,size=(M))
        big_I = np.eye(N) #identity
        B = big_I[B_rows,:]

        #vector of measurements
        y = np.matmul(B,vec_F)

        
        A = np.matmul(B,iD) 

        x = cp.Variable(N) 
        objective = cp.Minimize(cp.norm((x),1)) #minimize l1 norm of x
        constraint = [np.matmul(A,x)==y] #subject to Ax = y
        prob = cp.Problem(objective,constraint)
        
        opt_val = prob.solve(verbose=True,solver="CVXOPT",max_iter=1000,reltol=1e-2,featol=1e-2)
        opt_x = x.value #this is the DCT of an image F* that hopefully resembles original F

        sparse_reconstruct_img_vec = np.matmul(sparse_idct,opt_x)
        
        sparse_reconstruct_img =np.reshape(sparse_reconstruct_img_vec,(Nx,Ny))
        imgs_sparse_holder.append(sparse_reconstruct_img)





fig,ax = plt.subplots(3,3,figsize=(10,10))
for i in [0,1,2]:
    for j in [0,1,2]:
        ax[i,j].imshow(imgs_sparse_holder[3*i+j+1],cmap="gray")
        ax[i,j].set_title("r = " + str(r_list[i])+", trial "+str(j+1),fontsize=15)
        ax[i,j].set_xticks([])
        ax[i,j].set_yticks([])
fig.suptitle("Image Recovery with varying Compression, r",fontsize=20)
plt.show()
    

