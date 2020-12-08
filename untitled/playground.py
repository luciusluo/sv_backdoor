import numpy as np
import cvxopt

from cvxopt import solvers
from cvxopt import matrix

from numpy.linalg import eigh

cvxopt.solvers.options['show_progress'] = True