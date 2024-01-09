# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 10:14:18 2022

@author: Jon

"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D

data = np.load("subdata.npy")


#import libraries for plotting isosurfaces
import plotly 
import plotly.graph_objs as go



from Gaussian import Gaussian #custom function in separate file

#plot data in time

L = 10; #lenght of spatial domain (cube of sides L=10)
N_grid = 64; #number of grid points/Fourier modes in each direction
xx = np.linspace(-L,L,N_grid+1) #spatial grid in x direction
x = xx[0:N_grid]
y = x 
z = x

K_grid = ((2*np.pi)/(2*L))*np.linspace(-N_grid/2,N_grid/2-1,N_grid) #frequency grid for one coordinate

xv, yv, zv = np.meshgrid(x,y,z) #3d mesh for plotting

kvx,kvy,kvz = np.meshgrid(K_grid,K_grid,K_grid)

#Computes 2D Frequencies Averaged in time
freq_2 = np.zeros((N_grid,N_grid,N_grid))
for j in range(0,49): #49 is number of time steps
    signal = np.reshape(data[:,j],(N_grid,N_grid,N_grid)) #reshape to 64x64x64

    freq = np.fft.fftn(signal) #take 3d fourier transform
    freq_shift = np.fft.fftshift(freq) #fourier shift
    freq_1 = freq_shift
    freq_th = freq_1 + freq_2 #add frequencies together for average
    freq_1 = freq_2 
    freq_2 = freq_th
    
avg_freq = abs((1/49)*freq_2) #averaging the frequencies across time

real_avg_freq = np.array(avg_freq).real #grab real part of the frequency 



#plotting 2D Time average frequency slice
# =============================================================================
# sig_ticks = [0,6.4] #this is to relabel axes
# sig_ticks_append = 6.4
# for k in range(1,10):
#    sig_ticks_append += 6.4
#    sig_ticks.append(sig_ticks_append)
# fig,ax = plt.subplots()    
# im =ax.pcolormesh(np.max(real_avg_freq,axis=2)) 
# fig.colorbar(ax.imshow(np.max(real_avg_freq,axis=2),origin="l"), ax=ax)
# plt.xticks(sig_ticks,[-10,-8,-6,-4,-2,0,2,4,6,8,10])
# plt.yticks(sig_ticks,[-10,-8,-6,-4,-2,0,2,4,6,8,10])
# plt.xlabel("k_y")
# plt.ylabel("k_x")
# plt.title("|time averaged frequencies| at k_z=")
# plt.show()
# =============================================================================



#Plotting 2d slices where dominant frequency can be seen 
# =============================================================================
# center_freq_1=real_avg_freq[:,:,10]
# 
# center_freq_2 = real_avg_freq[:,:,54]
#  
# im =plt.pcolormesh(center_freq_1) 
# cbar = plt.colorbar()
# cbar.ax.set_title("Hz")
# plt.xticks(sig_ticks,[-10,-8,-6,-4,-2,0,2,4,6,8,10])
# plt.yticks(sig_ticks,[-10,-8,-6,-4,-2,0,2,4,6,8,10])
# plt.xlabel("k_y")
# plt.ylabel("k_x")
# plt.title("|Time Avg. Frequencies|, k_z ~ -6.911")
# plt.show()
# 
# 
# =============================================================================


#plotting raw signal at fixed t and fixed z
# =============================================================================
# signal_test = np.reshape(data[:,0],(N_grid,N_grid,N_grid))
# sig_ticks = [0,3.0476]
# sig_ticks_append = 3.0476
# for k in range(1,20):
#     sig_ticks_append += 3.0476
#     sig_ticks.append(sig_ticks_append)
# 
# fig,ax = plt.subplots()    
# im =ax.pcolormesh(signal_test[::,0]) 
# fig.colorbar(ax.imshow(np.max(real_avg_freq,axis=2),origin="l"), ax=ax)
# plt.xticks(sig_ticks,[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10])
# plt.yticks(sig_ticks,[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10])
# plt.xlabel("y")
# plt.ylabel("x")
# plt.title("OG sig t=10,z=10")
# plt.show()
# 
# =============================================================================


center1_coord1,center1_coord2,center1_coord3 = np.unravel_index(real_avg_freq.argmax(),real_avg_freq.shape) #indices of dominant frequency, not labeling x,y,z because it confused me when plotting because order of coordinates is not x,y,z 

center2_coord1,center2_coord2,center2_coord3 = np.unravel_index(real_avg_freq[0:32,0:32,32:63].argmax(),real_avg_freq[0:32,0:32,32:63].shape) #using symmetry we can figure out what index the other gaussian should be
                                                                                                                                #but better safe to check anyway
center2_coord3 = center2_coord3+32 #output index is weird because it started at 0 for index value 32

#for sigma in range(1,30,2): #for loop for sigma if you want
gaussian_filter = Gaussian(kvx,kvy,kvz,center1_coord1,center1_coord2,center1_coord3,7) + Gaussian(kvx,kvy,kvz,center2_coord1,center2_coord2,center2_coord3,7) #Gaussian centered at dominant frequency

coord1 = []
coord2 = []
coord3 = []
for j in range(0,49):
    signal = np.reshape(data[:,j],(N_grid,N_grid,N_grid))
    freq = np.fft.fftn(signal)
    filtered_freq = np.fft.fftshift(freq*gaussian_filter) #shift everything in fourier/freq domain
    filtered_signal = np.fft.ifftn(np.fft.ifftshift(filtered_freq)).real
    coord1_append,coord2_append,coord3_append = np.unravel_index(filtered_signal.argmax(),filtered_signal.shape) #indices of dominant frequencies in each direction
    
    coord1.append(coord1_append)
    coord2.append(coord2_append)
    coord3.append(coord3_append)
    
    
    
#Plotting the 3D trajectory 
# =============================================================================
fig = plt.figure()
ax= Axes3D(fig)

ax.scatter(y[coord2],x[coord1],z[coord3],c="purple",s=12)
plt.plot(y[coord2],x[coord1],z[coord3],"k")
ax.view_init(elev=90,azim=90)
ax.set_xlabel("x")
ax.set(xlim=(-8,8))
ax.set_ylabel("y")
ax.set(ylim=(-8,8))
ax.set_zlabel("z")
plt.title("Submarine Trajectory")

plt.show()
# =============================================================================



#Plotting filtered signal for fixed t and fixed z
# =============================================================================
# sig_plot = np.reshape(data[:,0],(N_grid,N_grid,N_grid))
# freq_test = np.fft.fftn(sig_plot)
# filter_freq_test = np.fft.fftshift(freq_test*gaussian_filter)
# filter_sig_test= np.fft.ifftn(np.fft.ifftshift(filter_freq_test)).real
#         
# fig,ax = plt.subplots()    
# im =ax.pcolormesh(filter_sig_test[::,0]) 
# fig.colorbar(ax.imshow(np.max(real_avg_freq,axis=2),origin="l"), ax=ax)
# plt.xticks(sig_ticks,[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10])
# plt.yticks(sig_ticks,[-10,-9,-8,-7,-6,-5,-4,-3,-2,-1,0,1,2,3,4,5,6,7,8,9,10])
# plt.xlabel("y")
# plt.ylabel("x")
# plt.title("filter sig t=10,z=10")
# plt.show()
# =============================================================================

    
#Plotting cleaned signal in 3D through time
# =============================================================================
#     for j in range(0,49,3):
#     
#       signal = np.reshape(data[:, j], (N_grid, N_grid, N_grid))
#       freq = np.fft.fftn(signal)
#       #freq_shift = np.fft.fftshift(freq)
#       filtered_freq = np.fft.fftshift(freq*gaussian_filter)
#       filtered_signal = np.fft.ifftn(np.fft.ifftshift(filtered_freq)).real
#       filtered_sig_norm = np.abs(filtered_signal)/np.abs(filtered_signal).max()
#     
#       # generate data for isosurface of the 3D data 
#       fig_data = go.Isosurface( x = xv.flatten(), y = yv.flatten(), z = zv.flatten(),
#                                 value = filtered_sig_norm.flatten(), isomin=0.7, isomax=0.7)
#       
#       # generate plots
#       clear_output(wait=True) # need this to discard previous figs
#       fig = go.Figure( data = fig_data )
#       fig.show()
# =============================================================================
    
    
#Plotting each coordinate as a function of time
# =============================================================================
# 
# time = np.linspace(1,49,49)/2
# 
# 
# plt.scatter(time,y[coord2],c="blue",s=11)
# #plt.plot(time,y[coord2],"k")
# plt.title("x(t)")
# plt.xlabel("time (hours)")
# plt.ylabel("x-coordinate")
# plt.show()
# 
# plt.scatter(time,x[coord1],c="red",s=11)
# #plt.plot(time,x[coord1],"k")
# plt.title("y(t)")
# plt.xlabel("time (hours)")
# plt.ylabel("y-coordinate")
# plt.show()
# 
# plt.scatter(time,z[coord3],c="green",s=11)
# #plt.plot(time,z[coord3],"k")
# plt.title("z(t)")
# plt.xlabel("time (hours)")
# plt.ylabel("z-coordinate")
# plt.show()
# =============================================================================


#aircraft trajectory x vs y plot
# =============================================================================
# 
# plt.scatter(y[coord2],x[coord1],c="orange")
# plt.title("y(t) vs x(t)")
# plt.xlabel("x-coordinate")
# plt.ylabel("y-coordinate")
# plt.show()
# 
# =============================================================================





