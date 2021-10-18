#!/usr/bin/env python3
"""! @brief PCA model. """

##
# @file PCA.py
#
# @brief Principal Component Analysis (PCA)
#
# @section description_doxygen_example Description
# Estimates PCA models of the form (Pi, Psi, Omega) 
# <am_standards link>
# Routines:
# - pca (main routine)
# - james_stein
#
# @section libraries_main Libraries/Modules
# - Numpy
# - Scipy
#
# @section notes_doxygen_example Notes
# - Comments are Doxygen compatible.
#
# @section todo_doxygen_example TODO
# - None.
#
# @section author_doxygen_example Author(s)
# - Created by Alex Shkolnik on 10/15/2021.
#
# Copyright (c) 2021 Prossimo Tech Inc.  All rights reserved.



import numpy as np
from scipy.sparse.linalg import eigsh
from helpers import *

def pca_model(data, options):
    
    # TODO : add options defaults

    # load the data p x n matrix
    returns = struct(data)
    n = returns.n
    p = returns.p
    Y = returns.data.T

    # load the pca options
    opt = struct(options)

    if (opt.debug):
        check_input(Y, n, p, opt)

    q = opt.number_factors
 
    #####################################
    # estimate low dimensional subspace #
    #####################################
    
    # estimate low dimensional subspace #

    if (p > n):  # use dual sample covariance for speed

        # TODO : check if correlation matrix is needed by 
        # any of the options to compute once only

        L = Y.T @ Y / p
        S_trace = np.trace(L) * p / n
        
        if q <= 0:
            q = get_number_factors(L, n, p, q, dual=True)

        vals, vecs = eigsh(L, q, which='LA')
        
        # Model Y = Pi @ diag(sqrt(n*v)) @ X + Z
        X = np.fliplr(vecs)         # factor returns
        v = vals[::-1] * p / n      # factor variances
        B = Y @ X / np.sqrt(n * v)  # factor exposures

    #else:  # use sample covariance for speed
       
        S = Y @ Y.T / n
        S_trace = np.trace(S)
        
        if q <= 0:
            q = get_number_factors(S, n, p, q, dual=False)
        
        vals, vecs = eigsh(S, q, which='LA')
        
        # Model Y = Pi @ diag(sqrt(n*v)) @ X + Z
        v = vals[::-1]
        B = np.fliplr(vecs)
        X = Y.T @ B / np.sqrt(n * v)       


    # TODO : define spectrum class to use for adjustment
    # q top eigenvalues, trace, n, p, q

    #########################################
    # apply adjustments to factor exposures #
    #########################################
    for adj in opt.exposure_adjustments:
        i = adj['factor_id']
        B[:,i-1] = adjust_exposure(n,p,q,i,v,
            adj['type'],B[:, i-1]) 
    
    #########################################
    # apply adjustments to factor variances #
    #########################################
    for adj in opt.variance_adjustments:
        i = adj['factor_id']
        v[i-1] = adjust_variance(n,p,q,i,v,adj['type'],S_trace) 

    # follow am_standards to normalize exposures/variances
    Pi, psi = standardize_factors(B, v)
        
    Z = Y - (Pi * np.sqrt(n*psi)) @ X.T 
    specifics = np.diag(Z @ Z.T) / n

    #########################################
    # apply adjustments to specific risks   #
    #########################################
    for adj in opt.variance_adjustments:
        specifics = adjust_specifics(n,p,q,v,specifics,
            adj['type'],S_trace) 
  

    model = dict()
    model.update(p = p)
    model.update(n = n)
    model.update(q = q)
    model.update(method = 'PCA')
    model.update(code = 'PCA version 1.0')
    model.update(options = options)
    model.update(exposures = Pi)
    model.update(variances = psi)
    model.update(specifics = specifics)


    return model

# temporary import
import inspect

def check_input(Y, n, p, options):
    fn = inspect.currentframe().f_code.co_name
    print("Unimplemented : " + fn)
    return True


def adjust_exposure(n,p,q,i,v,type,vec):
    fn = inspect.currentframe().f_code.co_name
    print("<fail> : unimplemented : " + fn)
    return(vec)

def adjust_variance(n,p,q,i,v,type,S_trace):
    fn = inspect.currentframe().f_code.co_name
    print("<fail> : unimplemented : " + fn)
    return(v[i-1])

def adjust_specifics(n,p,q,v,specifics,type,S_trace): 
    fn = inspect.currentframe().f_code.co_name
    print("<fail> : unimplemented : " + fn)
    return(specifics)

def get_number_factors(M, n, p, q, dual):
    fn = inspect.currentframe().f_code.co_name
    print("<fail> : unimplemented : " + fn)
    if q <= 0:
        q = 1
    return(q)

def standardize_factors(B, v):
    fn = inspect.currentframe().f_code.co_name
    print("<fail> : unimplemented : " + fn)
    return (B, v)

