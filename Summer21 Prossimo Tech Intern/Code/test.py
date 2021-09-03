from scipy.linalg import eigh, eig
import numpy as np
from scipy.sparse.linalg import eigsh
import time

a = np.array([[-2, -4, 2],[-2, 1, 2],[4, 2, 5]])
#a = np.array([[2, 2], [5, -1]])
"""
eigval, eigvec = eigh(a)
print(eigval)
print(eigvec)

eigh_start = time.time()
val, vec = eigh(a)
eigh_end = time.time()

eigsh_start = time.time()
val2, vec2 = eigsh(a)
eigsh_end = time.time()

print("eigh run time is: ", eigh_end-eigh_start)
print("eigsh run time is: ", eigsh_end-eigsh_start)

val, vec = eigh(a)
val_q = val[:-3:-1]

print("val: ", val)
print("vec: ", vec)

vec[:, 2]=vec[:,1]
print(vec[:,2])
"""

from sys import path
from pathlib import Path

path.append(Path(__file__).parent.absolute())

print(Path(__file__).parent.absolute())