# PCA method
# Input: pandas.DataFrame
from scipy.linalg import eigh
import numpy as np
from math import sqrt
from sys import path
from pathlib import Path
path.append(Path(__file__).parent.absolute())
from Analyzer import calculator


# Input:
#   Y: p-n asset return matrix
#   q: num of factors. aka rank?  
#   js: flag to apply "James-Stein" correction. Default=0. When JS=1 applies the JS bias correction
#   sped: flag(0 or 1) to speed up PCA. Default=0.
#   corrad: correlation-based adjustment. flag(0 or 1)

def pca(Y, q, js=0, sped=0, corrad=0):
  #============= Step 1: Form covariance matrix =============#
  p = Y.shape[0]
  n = Y.shape[1]
  S = Y @ Y.T / n  # Covariance matrix
  L = Y.T @ Y / p  # used for speed up when p>=2n

  #============= Step 2 =============#
  eigval = None
  eigvec= None
  if sped == 0:
    # Standardized PCA
    # Extract original eigenvalues, eigenvectors of sample covariance matrix
    eigval_S, eigvec_S = eigh(S)  # eigval_S: 1d array. eigvec_S: 2d array
    eigval = eigval_S[:-(q+1):-1] # last 'rank' elements in reversed order (large to small) 
    eigvec = (eigvec_S.T)[:-(q+1):-1].T # corresponding eigenvectors to make them in column form
  elif sped == 1:
    # Sped up PCA
    eigval_L, eigvec_L = eigh(L)
    eigval = eigval_L[:-(q+1):-1] # last 'rank' elements in reversed order (large to small) 
    eigvec = (eigvec_L.T)[:-(q+1):-1].T # corresponding eigenvectors
    eigvec = Y @ eigvec / sqrt(p)
    for i in range(q):
      eigvec[:,i] = eigvec[:,i]/eigval[i]
    eigval = np.square(eigval) * p / n

  #============= Step 3 =============#
  # Helper func1: James Stein
  def js_correct(mh):
    # variance of first vector of largest eigenvalue
    varh = np.var(eigvec[:, 0])  
    # eigval, eigvec equals those of S or L regardless whether sped = 0 or 1
    c = 1 - (np.trace(S)- np.sum(np.square(eigval)))/((min(n,p)-q) * p * eigval[0] * varh) if sped == 0 \
      else 1 - (np.trace(L)- np.sum(np.square(eigval)))/((min(n,p)-q) * p * eigval[0] * varh)
    # compute the corrected vector
    rho = (mh + c * (eigvec[:,0] - mh))/mh
    return rho

  # Helper func2: correlation bias adjust
  def corr_correct(mh):
    # D = np.diag(S)
    # Let R be the correlation matrix for S. ??(should we use R=D^-1SD^-1 or cor_matrix)
    R = calculator.cor_matrix(S)
    eigval_R, eigvec_R = eigh(R)
    eigval_R = eigval_R[:-(q+1):-1] # last 'rank' elements in reversed order (large to small) 
    eigvec_R = (eigvec_R.T)[:-(q+1):-1].T # corresponding eigenvectors
    # Construct B
    # 1. rho = v(1)/m(v(1)). v(1) is 1st column of eigvec_R
    rho = eigvec_R[:,0]/ np.mean(eigvec_R[:, 0])  
    # 2. kth column of B for 1<k<=q: gamma(k) = v(k)/mh
    B = eigvec_R / mh
    B[:,0] = rho
    # Construct V
    V = np.diag(np.square(mh) * eigval_R)
    delta = np.diag(S - np.dot(B@V, B.T))
    return B, V, delta

  # start step 3
  B = None
  V = None
  delta = None
  if js == 0:
    if corrad == 0:
      # Standardized PCA
      print(eigvec[:,0])
      mh = np.nanmean(eigvec[:,0])  # m(h(1)). use numpy.nanmean to avoid nan output
      B = eigvec/mh # normalized p-q
      V = np.diag(np.square(mh) * eigval) # q-q diagonal
      delta = np.diag(S - np.dot(B@V, B.T)) # 1-p
    elif corrad == 1:
      # Extract first eigvec h of S with the largest eigenvalue
      mh = np.mean(eigvec[:,0])
      B, V, delta = corr_correct(mh)
  else:
    # No correlation bias correction. Simply perform JS.
    if corrad == 0:
      # mean of first vector of largest eigenvalue
      mh = np.mean(eigvec[:,0])  
      # Construct B using js corrected rho
      B = eigvec/mh
      rho = js_correct(mh)
      B[:,0] = rho
      # Normal way of computing B, V, delta
      V = np.diag(np.square(mh) * eigval)
      delta = np.diag(S - np.dot(B@V, B.T))
    else:
      mh = np.mean(eigvec[:,0]) 
      # apply correlation correction only to factors i>=2
      B, V, delta = corr_correct(mh)
      # compute james stein corrected rho
      rho = js_correct(mh)
      # update to corr corrected B
      B[:,0] = rho

  # Step 4
  return {"B":B, "V":V, "delta":delta}


