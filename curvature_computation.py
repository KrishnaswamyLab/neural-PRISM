from os import listdir

import numpy as np

import phate

import h5py
import scipy
import scipy.io as sio
import time

from collections import Counter
from collections import defaultdict
from numpy import linalg as LA
from tqdm import tqdm
import matplotlib.pyplot as plt



from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from sklearn.metrics import pairwise_distances
import networkx as nx
import tphate
import scprep
from gtda.diagrams import BettiCurve
from numpy import savetxt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D 


def fit_circle_2d(x, y, w=[]):
    
    A = np.array([x, y, np.ones(len(x))]).T
    b = x**2 + y**2
    
    # Modify A,b for weighted least squares
    if len(w) == len(x):
        W = np.diag(w)
        A = np.dot(W,A)
        b = np.dot(W,b)
    
    # Solve by method of least squares
    c = np.linalg.lstsq(A,b,rcond=None)[0]
    
    # Get circle parameters from solution c
    xc = c[0]/2
    yc = c[1]/2
    r = np.sqrt(c[2] + xc**2 + yc**2)
    return xc, yc, r

def rodrigues_rot(P, n0, n1):
    
    # If P is only 1d array (coords of single point), fix it to be matrix
    if P.ndim == 1:
        P = P[newaxis,:]
    
    # Get vector of rotation k and angle theta
    n0 = n0/np.linalg.norm(n0)
    n1 = n1/np.linalg.norm(n1)
    k = np.cross(n0,n1)
    k = k/np.linalg.norm(k)
    theta = np.arccos(np.dot(n0,n1))
    
    # Compute rotated points
    P_rot = np.zeros((len(P),3))
    for i in range(len(P)):
        P_rot[i] = P[i]*np.cos(theta) + np.cross(k,P[i])*np.sin(theta) + k*np.dot(k,P[i])*(1-np.cos(theta))

    return P_rot

def angle_between(u, v, n=None):
    
    if n is None:
        return np.arctan2(np.linalg.norm(np.cross(u,v)), np.dot(u,v))
    else:
        return np.arctan2(np.dot(n,np.cross(u,v)), np.dot(u,v))
    
def compute_curvature(nbd, traj, num_pts):

    kappa = []

    for pt_idx in range(0, num_pts):

        P = traj[max(0, pt_idx-nbd):min(num_pts, pt_idx+nbd),:]
        P_mean = P.mean(axis=0)
        P_centered = P - P_mean
        U,s,V = np.linalg.svd(P_centered)
        normal = V[2,:]
        d = -np.dot(P_mean, normal) 
        P_xy = rodrigues_rot(P_centered, normal, [0,0,1])
        xc, yc, r = fit_circle_2d(P_xy[:,0], P_xy[:,1])
        kappa.append(1.0/r)
    
    return np.array(kappa)

def curvature_nd(traj, dt=1e-3):
    # traj is an (m, n) array where m is the number of points and n is the dimension
    m, n = traj.shape
    
    # First and second derivatives with respect to time
    X_dot = np.gradient(traj, dt, axis=0)         # First derivative
    X_ddot = np.gradient(X_dot, dt, axis=0)       # Second derivative
    
    # Calculate norms of the first derivatives (speeds)
    X_dot_norm = np.linalg.norm(X_dot, axis=1)
    X_ddot_norm = np.linalg.norm(X_ddot, axis=1)
    
    # Compute cross-product magnitude for higher dimensions as a Gramian determinant
    dot = np.diag(np.dot(X_dot, X_ddot.T) ** 2)
    
    
    curvature_num = np.sqrt(X_dot_norm**2 * X_ddot_norm**2 - dot)
       
    kappa = curvature_num / X_dot_norm[i]**3

    return kappa


group = 1  #change for group 1
sublist = [1,2,3, 4,5, 6, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21] # group 1
condlist = [1,2,3, 4]
blocklist = [1,2,3, 4, 5, 6]
#sublist = [1,2,4,5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]   # group 2

band = 'alpha'

data = np.load(f'data/G{group}_tPhate_{band}.npz')

cond1 = []
cond2 = []
cond3 = []
cond4 = []
for i in range(0, len(sublist)):
    for j in range(0, len(condlist)):
        for k in range(0, len(blocklist)):
            sub = sublist[i]
            cond = condlist[j]
            block = blocklist[k]
            key = f'{band}_sub{sub}_cond{cond}_block{block}'
            if key in data:
                if cond == 1:
                    cond1.append(data[key])
                elif cond == 2:
                    cond2.append(data[key])
                elif cond == 3:
                    cond3.append(data[key])
                elif cond == 4:
                    cond4.append(data[key])
             
            else:
                print(f'Warning: {key} not found in the data.')
                
group = 1  #change for group 1
sublist = [1,2,3, 4,5, 6, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 21] # group 1
condlist = [1,2,3, 4]
blocklist = [1,2,3, 4, 5, 6]
#sublist = [1,2,4,5, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17]   # group 2

nbhd_size = int(data['alpha_sub1_cond1_block1'].shape[0] * 0.05)
band = 'alpha'

data = np.load(f'data/G{group}_tPhate_{band}.npz')

cond1 = []
cond2 = []
cond3 = []
cond4 = []
for i in tqdm(range(0, len(sublist))):
    for j in range(0, len(condlist)):
        for k in range(0, len(blocklist)):
            sub = sublist[i]
            cond = condlist[j]
            block = blocklist[k]
            key = f'{band}_sub{sub}_cond{cond}_block{block}'
            if key not in data:
                continue
            if cond == 1:
                cur = compute_curvature(nbhd_size, data[key], len(data[key]))
                cond1.append(cur)
            elif cond == 2:
                cur = compute_curvature(nbhd_size, data[key], len(data[key]))
                cond2.append(cur)
            elif cond == 3:
                cur = compute_curvature(nbhd_size, data[key], len(data[key]))
                cond3.append(cur)
            elif cond == 4:
                cur = compute_curvature(nbhd_size, data[key], len(data[key]))
                cond4.append(cur)
                
                
G1_cond1_all = np.array(cond1)
G1_cond2_all = np.array(cond2)
G1_cond3_all = np.array(cond3)
G1_cond4_all = np.array(cond4)

G1_cond_stack_all_cur = np.concatenate((G1_cond1_all, G1_cond2_all, G1_cond3_all, G1_cond4_all), axis = 0)
f = 'data/Curvature/alpha_G1_cond_stack_all_cur.npy'
np.save(f, G1_cond_stack_all_cur)

