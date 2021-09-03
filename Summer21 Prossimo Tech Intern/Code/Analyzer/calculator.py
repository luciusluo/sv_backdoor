"""
Created: Mon Aug  16 10:30:00 2021
@author: Yuxiao
"""
# -*- coding: utf-8 -*-
######### Required Modules ##########
import numpy as np
from scipy.sparse.linalg import eigsh
import scipy
import matplotlib.pyplot as plt
from multiprocessing import Pool as CPU
sqrt = np.sqrt


################## General methods ##################
# Collect all financial indicators methods from sample_scatter.py

# Input:
#   R: p-n index return matrix in numpy.ndarray
#   centered: 0 or 1. Indicate whether to normalize mean to 0
# Output:
#   S: n-n sample covariance matrix
def cov_matrix(R, centered = False):
    Z = R - centered * np.ones((R.shape[0], 1)) @ R.mean(axis = 0).reshape(1, R.shape[1])
    S = Z.T @ Z / Z.shape[0] 
    return S

# Compute correlation matrix from covariance matrix
# Input: n-n covariance matrix
# Output: n-n correlation matrix
def cor_matrix(Sigma):
    stdevs = np.sqrt(np.diag(Sigma))
    Corma = (np.diag(1/stdevs) @ Sigma) @ np.diag(1/stdevs)
    return(Corma)

# Compute average correlation matrix from covariance matrix
# Input: n-n covariance matrix
def ave_corr(Sigma):
    N = Sigma.shape[0]
    Corma = cor_matrix(Sigma)
    acorr = np.sum(Corma - np.eye(N))/(N**2-N)
    return(acorr)

# Compute Dispersin of portfolio return
# Input: n-1 vector
def dispersion(vec):
    if (np.sum(vec) < 0):
        vec = -1.0 * vec
    vave = np.mean(vec)
    vvar = np.mean((vec - vave)**2)
    return(sqrt(vvar)/vave)

# Compute melt_up
def melt_up(vec):
    if (np.sum(vec) < 0):
        vec = -1.0 * vec
    up = np.sum(vec > 0)
    down = np.sum(vec <= 0)
    return(np.sum(up - down)/len(vec))

# Compute skewness
def skewness(vec):
    if (np.sum(vec) < 0):
        vec = -1.0 * vec
    vave = np.mean(vec)
    vvar = np.mean((vec - vave)**2)
    vskw = np.mean((vec - vave)**3)
    return(np.cbrt(vskw/(vave*vvar)))

# Compute entropy
def entropy(vec):
    if (np.sum(vec) < 0):
        vec = -1.0 * vec
    vave = np.mean(vec)
    vvar = np.mean((vec - vave)**2)
    phis = scipy.stats.norm(vave, np.sqrt(vvar)).pdf(vec)
    return(-1.0*np.sum(phis*np.log(phis)))

# A SINGLE SAMPLE FUNCTION FOR LAER SIMULATION
# INPUT:
#   N: stocks
#   T: observations
#   K: factors
#   fmg_seed: factor model seed
#   mark_vol: market volatility (% ann.)
#   svol_min, svol_max:  min/max specific volatility
#   min_fvol, max_fvol:  min/max nonbeta factor volatility
#   tau_beta:    beta dispersion
def sample(seed, N=128, T=256, K=1, fmg_seed=0, mark_vol=16, 
    svol_min=10, svol_max = 50, min_fvol = 1, max_fvol = 9, tau_beta = 0.4):

    # Initializations
    eucn = np.linalg.norm
    srng = np.random.RandomState 

    # EDIT only the "ADD" section below
    # CONSTUCT FACTOR MODEL
    ones = np.ones(N)
    z = ones / sqrt(N)

    fmrng = srng(fmg_seed)
    beta = fmrng.normal (1.0, tau_beta, N)  # n
    # normalize to have unit mean and disersion tau_beta
    # first normalize ...
    beta = beta / np.mean(beta)
    # now it follows that |beta|^2 = N (tau_beta^2 + 1)
    # taking c beta + (1-c) does not change the mean but 
    # changes the dispersion to c tau_beta. therefore,
    c = tau_beta / np.sqrt(np.mean((beta-1.0)**2))

    # final beta vector with mean one and dispersion tau_beta
    beta = c * beta + (1-c)

    # market: 16% ann. vol. (256 trading days)
    fvol = sqrt((mark_vol / 100)**2 / 256)
    # specific: ann. vol.
    svol = fmrng.uniform(svol_min, svol_max, N)
    svol = sqrt((svol / 100)**2 / 256)
    # specific variance
    svar = svol**2

    #renormalize betas and variances
    beuc = beta / eucn(beta)
    # variance of factor beta (with mean entry 1)
    fvar = fvol**2
    # variance of factor beuc of Euclidian length 1
    flam = fvar * eucn(beta)**2

    Sigma = np.diag(svar)

    # add remaining (non beta) factors
    for k in range(K):
        if k: # add a style factor
            zeta = fmrng.normal (0.0, 1.0, N)
            zvol = np.random.uniform(min_fvol,max_fvol)
            zvar = (zvol / 100)**2 / 256
            Sigma = Sigma + zvar * np.outer(zeta, zeta)
        else: # add market factor (betas)
            Sigma = Sigma + fvar * np.outer(beta, beta)


    rng = srng(seed) 
    
    # specific and factor returns
    sret = rng.normal (0, svol * ones, (T,N))
    fret = rng.normal (0, fvol, T) # p

    # security returns
    R = np.outer(fret, beta) + sret # p-n
    
    # compute the sample covariance
    S = R.T.dot(R) / T  # n-n
    C = cor_matrix(S)
    
    # extract the first eigenvector (1st PC)
    cval, cvec = eigsh (C, 1, which='LA')
    hval, hvec = eigsh (S, 1, which='LA')
    hvar = hval[0]
    cvar = cval[0]
    hvec = hvec.flatten()
    if (sum(hvec) < 0):
        hvec = -hvec
    if (sum(cvec) < 0):
        cvec = -cvec

    svar_est = np.diag(S - hvar*np.outer(hvec,hvec))
    cov_est = hvar*np.outer(hvec,hvec) + np.diag(svar_est)
 
    ## BEGIN ADD ANY METRIC TO COMPUTE HERE ##

    data = dict()
    data.update(ave_pwcorr=ave_corr(cov_est))
    data.update(cov_beta_dipsersion=dispersion(hvec))
    data.update(cov_beta_entropy=entropy(hvec))
    data.update(cov_beta_melt_up=melt_up(hvec))
    data.update(cov_beta_projection=np.mean(hvec))
    data.update(cov_beta_skewness=skewness(hvec))
    data.update(cor_beta_dipsersion=dispersion(cvec))
    data.update(cor_beta_melt_up=melt_up(cvec))
    data.update(cor_beta_entropy=entropy(cvec))
    data.update(cor_beta_projection=np.mean(cvec))
    data.update(cor_beta_skewness=skewness(cvec))
    data.update(variance=hvar*np.mean(hvec))
    data.update(cor_eigenvalue=cvar)
    data.update(cov_eigenvalue=hvar)
    
    ## END ADD ANY METRIC TO COMPUTE HERE ##
    
    return data


# Input: 
#   N,T, K, tau_beta, mark_vol, svol_min, svol_max: same as sample()
#   num_sims, num_cpus: number simulations and cpus
def simulate(N,T, K, tau_beta, mark_vol, svol_min, svol_max, num_sims = 2048, num_cpus = 8):
    pool = CPU (processes = num_cpus)
    seeds = range(1, 1 + num_sims)
    X = zip(seeds)
    Y = pool.starmap (sample, X)

    pool.close()
    pool.join()

    # PLOT
    deluge = "#7C71AD"
    keys = Y[0].keys()

    def custom_plot(s1,s2,v1,v2):
        tfig = v1 + '_vs_' + v2 + '_'
        tfig = tfig + f"N{N}T{T}K{K}"
        tfig = tfig + f"disp{tau_beta}"
        tfig = tfig + f"fvol{mark_vol}"
        tfig = tfig + f"minsvol{svol_min}"
        tfig = tfig + f"maxsvol{svol_max}"

        plt.figure (tfig, figsize=(8,8))
        plt.scatter(s1,s2, c=deluge)
        plt.xlabel(v1)
        plt.ylabel(v2)
        plt.tight_layout()

        plt.savefig("img/" + tfig + ".pdf", 
            transparent=True, format="pdf")
        plt.close()

    for i,v1 in enumerate(keys):
        for j,v2 in enumerate(keys):
            if (i > j):
                # extract the series for variables v1 and v2
                s1 = list(map(lambda x : x[v1], Y))
                s2 = list(map(lambda x : x[v2], Y))
                custom_plot(s1,s2,v1,v2)


def getA(L):
    vals, vecs = eigsh(L, 1, which = "LA")
    return vecs.dot(np.diag(vals).dot(vecs.T))
    
def getD(A, O):
    return np.diag(np.maximum(np.diag(O-A), 0))




