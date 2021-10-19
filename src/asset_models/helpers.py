#!/usr/bin/env python3

# Add numpydoc header 

import numpy as np
import time

sqrt = np.sqrt
eucn = np.linalg.norm
srng = np.random.RandomState


class return_data():
    """ Specification for data input to PCA 

    Attributes
    ----------
    source : str
        Data source path and time stamp.
    n : int
        The number of observations.
    p : int
        The number of assets.
    data : numpy.array
        n x p matrix of returns to assets.
    freq : str
        Data observation frequency in {'day', 'week', 'month', 
        'quarter', 'year'}.
    start : time
        Start of observation period.
    end  : time
        End of observation period.
    """

class pca_options(return_data):
    """ Options for PCA analysis for the input data
	
    Attributes
    ----------
    number_factors : int
        Number of factors if greater than 0; estimate otherwise.
    exposure_adjustments : list
        List of adjustements for exposures to factors (e.g. the
        James-Stein or correlation matrix based methods).
    variance_adjustments : list
        List of adjustements for factors variances (e.g. shrinkage
        estimators, random matrix theory based corrections.)
    specific_adjustments : list
        List of adjustments to specific risk estimates.
	
    """


class exposure_adjustment():
    """ Intructions to adjust factor exposures 

    Attributes
    ----------
    factor_id : int
    type : str
        Possible types are {'JS', 'COR'}
    """


class variance_adjustment():
    """ Intructions to adjust factor variances

    Attributes
    ----------
    factor_id : int
    type : str
        Possible types are {'RMT', 'MP'}
    """

class factor_model():
    """ An estimated asset model

    Attributes
    ----------
    p : int
        The number of assets.
    n : int
        The number of observations.
    q : int
        The number of factors. 
    method : str
        The method used to construct the model in {'PCA', ...}
    code : str
        Code version used to estimate the model
    options : dict
        Options passed to the method in {'pca_options', ...}
    exposures : numpy.array
        The p x q matrix of factor exposures <am_standards>
    variances : numpy.array
        The q vector of factor variances <am_standards>
    specific : numpy.array
        Either a p vector of specific risks for each asset or a
        p x p covariance matrix of the specific returns.
    """

def pca_model(data, options):
	""" Main routing for generating a PCA model 

    Parameters
    ----------
    data : return_data
        The returns data dictionary
    options : pca_options
        Specification for a particular models (default?)
	
    Returns
    -------
    model : dict
        Estimated model.

    Notes
    -----
	
    References
    ----------
    Path to documents <am_standards>.

    Examples
    --------
    """


class struct(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


def fancy_wait(k=5, dt=1):
    for i in range(k):
        print("* ", end='', flush=True)
        time.sleep(dt)
    print("<pass>")
    return(time.sleep(2*dt))

def sample_ave(x, w = None):
    if (w is not None):
        return( np.mean(w) )
    else:
        return( x @ w )

def sample_var(x, w = None):
    if (w is not None):
        return(np.var(x))


# to be outsources to simulation engine ...
def sample_basic_return(n, p, q, mod_seed=0, sim_seed=0):

    #########################
    # Generate factor model #
    #########################
    
    fmrng = srng(mod_seed)

    # market volatility 
    market_vol = 15 
    market_var = market_vol ** 2

    # factor volatilities and variances
    fvol = fmrng.uniform(1, 9, q-1) 

    # factor volatilities and variances
    fvol = np.append(market_vol, fvol)
    fvar = fvol**2

    # specific variance
    svol = fmrng.uniform(25, 75, p) 
    svar = svol ** 2

    # true dispersion level for beta
    beta_disp = 1/5

    # construct beta vector with disperion disp
    beta = fmrng.normal(1.0, beta_disp, p)
    beta = beta / np.mean(beta)
    draw = sqrt(np.mean((beta - 1.0)**2))
    c = beta_disp / draw
    # unit mean vector of p betas with dispersion = disp
    beta = c * beta + (1 - c) 

    # construct the factor matrix B
    B = np.zeros((p, q))
    # variances normalized for exposures on unit sphere
    for k in range(q):
        if k:  # add a style factor (mean zero, unit var.)
            zeta = fmrng.normal(0.0, 1.0, p)
            zeta = zeta - np.mean(zeta)
            # TODO: standardize to exact unit variance
            B[:, k] = zeta
        else:  # include market factor (betas)
            B[:, k] = beta

    # covariance matrix (not used)
    # Sigma = (B * fvar) @ B.T + np.diag(svar)

    #########################
    # Sample normal returns #
    #########################
    
    smrng = srng(sim_seed)

    X = smrng.normal(0, fvol, (n, q))
    Z = smrng.normal(0, svol * np.ones(p), (n, p)).T

    # realized market variance
    mvar_realized = (X[:,0].T @ X[:,0]) / n

    # generate security returns
    Y = B @ X.T + Z

    return(Y.T)


