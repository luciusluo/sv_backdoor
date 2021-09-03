# -*- coding: utf-8 -*-
######### Required Modules ##########
import numpy as np
import pandas as pd
from scipy.sparse.linalg import eigsh
from numpy.linalg import norm, matrix_rank
#from rpy2.robjects import r
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
#####################################


# Minimum Trace Factor Analysis (MTFA)
# Input: pandas.DataFrame
# 1. Helper Functions
def shrink_eig(X, tau):
    d, v = eigsh(X, k = min(5, X.shape[0]), which="LM")
    if d[0] <= tau:
        return np.dot(v*np.maximum(d-tau,0), v.T)
    else:
        return np.dot(v*np.maximum(d-tau,0), v.T)+shrink_eig(X-np.dot(v*d, v.T), tau)

def objective_func(O, A, tau):
    D = np.diag(np.maximum(np.diag(O-A), 0))
    return norm(A, 'nuc')*tau + 1/2*norm(O-A-D)**2

def func_A(O, A, tau):
    X = O-np.diag(np.maximum(np.diag(O-A),0))
    return shrink_eig(X, tau)

# 2. MTFA procedure functions
def fix_pt_iteration(k, O, A, tau):
    print(objective_func(O, A, tau))
    for i in range(k):
        A = func_A(O, A, tau)
        print(objective_func(O, A, tau))
    return A

def fp2(O, A, tau, epsilon = 1e-12,max_iter = 5000):
    a = np.inf,
    for _ in range(max_iter):
        A = func_A(O, A, tau)
        b = objective_func(O, A, tau)
        if(abs(a-b)<epsilon): break
        a = b
    return A

def match_rank0(O, A, tau_range, r=1,max_iter = 100, steps = 10000, epsilon = 1e-8, smallest = True):
    # r: target rank
    # O: sample covariance
    tau = (tau_range[0]+tau_range[1])/2
    A = fp2(O, A, tau, epsilon, max_iter)
    mr = matrix_rank(A)
    for _ in range(steps):
        if mr == r: break
        tau_range[1*(mr<r)] = tau
        tau = (tau_range[0]+tau_range[1])/2
        A = fp2(O, A, tau, epsilon, max_iter)    
        mr = matrix_rank(A)
    out_tau = tau_range.copy()
    if mr == r:
        index = 1 - smallest
        tau_range[index] = tau
        mtau = tau
        for tau in np.geomspace(tau_range[index], tau_range[1-index], num = 50):
            A = fp2(O, A, tau, epsilon, max_iter)
            if matrix_rank(A) != r: 
                break
            mtau = tau 
        A = fp2(O, A, mtau, epsilon, max_iter)
        tau_range[index] = mtau
    return A, tau_range, (mr == r)

# MTFA portfolio analytics methods #
# minimum variance portfolio weights
def ls_minvar(B, V_inv, svar):
    M = B.T / svar
    A = V_inv + M @ B
    b = np.sum(M, 1)
    theta = np.linalg.solve(A,b)
    w = (1.0 - B @ theta) / svar
    return w/np.sum(w)