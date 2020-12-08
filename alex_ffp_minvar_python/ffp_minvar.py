import numpy as np
import cvxopt

from cvxopt import solvers
from cvxopt import matrix

from numpy.linalg import eigh
import time

start_time = time.time()

cvxopt.solvers.options['show_progress'] = True
cvxopt.solvers.options['maxiters'] = 1000000
cvxopt.solvers.options['abstol'] = 1e-14
cvxopt.solvers.options['reltol'] = 1e-14

srng = np.random.RandomState

eucn = np.linalg.norm
sqrt = np.sqrt


def linf(x):
    return (np.max(np.abs(x)))


N = 500  # number of securities
K = 4  # number of factors

# seed for beta and other factor exposures

seed = np.random.randint(0, 100000)
seed = 31877
fmrng = srng(seed)

ones = np.ones(N)
IN = np.diag(ones)

##########################################
# CONSTRUCT ONE FACTOR MODEL             #
# Y = B X + Z                         #
##########################################


# factor volatilities and variances
fvol = fmrng.exponential(5, K) / 100
fvar = fvol ** 2

# specific variance
svol = fmrng.uniform(0, 100, N) / 100
svar = svol ** 2

# construct the factor matrix B
B = np.zeros((N, K))
for k in range(K):
    cents = np.array([0, 1 / 4, 1 / 2, 3 / 4, 1])
    disps = np.array([1 / 8, 1 / 4, 1 / 2, 3 / 4, 1])
    cent = fmrng.choice(cents)
    disp = fmrng.choice(disps)
    sgn = fmrng.choice([-1.0, 1.0])
    beta = fmrng.normal(sgn * cent, disp, N)
    B[:, k] = beta

V = np.diag(fvar)
V_inv = np.diag(1.0 / fvar)
V_sqr = np.diag(sqrt(fvar))
V_inv_sqr = sqrt(V_inv)

# B.T is transpose
Sigma = B @ V @ B.T + np.diag(svar)

# reorient B
# signs = 1.0 * (((B.T / svar) @ ones) < 0)
# B = B * (1 - 2*signs)

def ls_minvar():
    M = B.T / svar
    A = V_inv + M @ B
    b = np.sum(M, 1)
    theta = np.linalg.solve(A, b)
    w = (1.0 - B @ theta) / svar

    r = dict(x=w / sum(w))
    r.update(w=w)
    r.update(theta=theta)

    return (r)


def phi(th):
    w = np.maximum(0.0, ones - B @ th) / svar

    return ((B * fvar).T @ w)


def psi(th):
    domg = 1.0 * (ones > B @ th) / svar
    A = V_inv + (domg * B.T) @ B
    b = B.T @ domg
    theta = np.linalg.solve(A, b)

    return (theta)


def ffp(t0):
    th_old = t0 - np.inf
    th_new = t0

    # list of positions that are zero on each iteration
    active_list = list()

    it = 0
    while (linf(th_new - th_old) > 1e-15):
        it = it + 1
        th_old = th_new
        th_new = psi(th_old)

        chi = 1.0 * (ones > B @ th_old)
        idx = np.where(chi < 0.5)[0]
        active_list.append(idx)

    r = dict(ffp_it=it)
    r.update(theta=th_new)
    r.update(psi=psi(th_new))
    r.update(phi=phi(th_new))
    r.update(active_it=active_list)

    return (r)


def ffp_ortho():
    G = np.zeros((N, K))
    om_old = -1.0 * np.inf
    om_new = np.zeros(K)

    # list of positions that are zero on each iteration
    active_list = list()

    it = 0
    while (linf(om_new - om_old) > 1e-15):
        om_old = om_new
        chi = 1.0 * (ones > G @ om_old)
        Omg = np.diag(chi / svar)

        idx = np.where(chi < 0.5)[0]
        active_list.append(idx)

        A = V_sqr @ B.T @ Omg @ B @ V_sqr
        kappa, O = eigh(A)

        O = np.fliplr(O)
        G = B @ V_sqr @ O
        # ensure positively oriented columns of G
        signs = 1.0 * (G.T @ Omg @ ones < 0)
        G = G * (1 - 2 * signs)

        gg = np.diag(G.T @ Omg @ G)
        ge = G.T @ Omg @ ones

        om_new = ge / (1 + gg)
        it = it + 1

    r = dict(ffp_it_ortho=it)
    r.update(G=G)
    r.update(omega=om_new)
    r.update(kappa=kappa[::-1])
    r.update(active_it_ortho=active_list)

    return (r)


def lo_minvar():
    ffpit = ffp(np.zeros(K))
    theta = ffpit['theta']

    w = np.maximum(0.0, ones - B @ theta) / svar
    x = w / sum(w)

    # run the same thing but by orthogonalizing the factors
    ffpit_ortho = ffp_ortho()

    chi = 1.0 * (w > 0)
    Omg = np.diag(chi / svar)

    # Largrange multiplier calculations
    BTx = B.T @ x
    G = ffpit_ortho['G']
    GTx = G.T @ x
    Oe = Omg @ ones
    lam = (1 + BTx @ (B.T @ Oe)) / sum(Oe)
    eta = (1 + GTx @ (G.T @ Oe)) / sum(Oe)

    r = dict(x=x)
    r.update(w=w)
    r.update(chi_x=chi)
    r.update(theta=theta)
    r.update(lam=lam)
    r.update(eta=eta)
    r.update(G=G)
    r.update(GTx=GTx)
    r.update(BTx=BTx)
    r.update(ffpit)
    r.update(ffpit_ortho)

    return (r)


def ls_numeric():
    q = matrix(np.zeros(N), tc='d')
    h = matrix(0.0, tc='d')
    G = matrix(np.zeros(N), tc='d').trans()
    A = matrix(np.ones(N), tc='d').trans()
    b = matrix([1], tc='d')

    sol = solvers.qp(matrix(2 * Sigma), q, G, h, A, b)
    w = [x for plist in np.asarray(sol['x']) for x in plist]

    return np.asarray(w)


def lo_numeric():
    q = matrix(np.zeros(N), tc='d')
    h = matrix(np.zeros(N), tc='d')
    G = matrix(-1.0 * IN, tc='d')
    A = matrix(np.ones(N), tc='d').trans()
    b = matrix([1], tc='d')

    sol = solvers.qp(matrix(2 * Sigma), q, G, h, A, b)
    w = [x for plist in np.asarray(sol['x']) for x in plist]

    return np.asarray(w)


xls = ls_numeric()
xlo = lo_numeric()

ls_var_numeric = xls @ (Sigma @ xls)
lo_var_numeric = xlo @ (Sigma @ xlo)

print("\nCVX Opt benchmarks:")

print(f"\nLS MinVar: {ls_var_numeric}")
print(f"LO MinVar: {lo_var_numeric}\n")

print(f"\nFFP (model seed {seed}).")

ls_mv = ls_minvar()
lo_mv = lo_minvar()

ls_var = ls_mv['x'] @ (Sigma @ ls_mv['x'])
lo_var = lo_mv['x'] @ (Sigma @ lo_mv['x'])

print(f"\nLS MinVar: {ls_var}")
print(f"LO MinVar: {lo_var}\n")

# indicator of long positions
chi = 1.0 * (lo_mv['w'] > 0)
print(f"Long positions: {np.int(sum(chi))}\n")
domg = chi / svar
G = lo_mv['G']
omega = lo_mv['omega']
theta = lo_mv['theta']
kappa = lo_mv['kappa']
GG = (domg * G.T) @ G
gg = np.diag(GG)

if (min(omega) < 0):
    print("****************************************")
    print("* Warning! Negative omega encountered.  *")
    print("****************************************")

print("Passing correctness checks (above 1e-6 is concerning).\n")
print(f"LS check: {eucn(ls_mv['x'] - xls)}")
print(f"LO check: {eucn(lo_mv['x'] - xlo)}\n")
print(f"ortho check 1: {eucn(G @ omega - B @ theta)}")
print(f"ortho check 2: {eucn(gg - kappa)}")
print(f"ortho check 3: {eucn(GG - np.diag(gg))}")

print("\nFactor orientation info.")

print(f"\nB.T D_inv e = {np.round((B.T / svar) @ ones, 1)}")
print(f"B.T Omega e = {np.round((domg * B.T) @ ones, 1)}")
print(f"G.T D_inv e = {np.round((G.T / svar) @ ones, 1)}")
print(f"G.T Omega e = {np.round((domg * G.T) @ ones, 1)}\n")

print("FFP iterates info.\n")
print(f"FFP iterations: {lo_mv['ffp_it']}")

if (lo_mv['ffp_it'] != lo_mv['ffp_it_ortho']):
    print("****************************************")
    print("* Warning! FFP and FFP Othro mismatch. *")
    print("****************************************")

for i in range(lo_mv['ffp_it']):
    active = lo_mv['active_it'][i]
    print(f" {len(active)} active positions at iteration {i + 1}")
    inter = np.intersect1d(active, lo_mv['active_it_ortho'][i])
    if (len(inter) != len(active)):
        print("***************************************")
        print("* Warning! FFP vs FFP_ortho mismatch. *")
        print("***************************************")

    if (i > 0):
        inter = np.intersect1d(active, lo_mv['active_it'][i - 1])
        if (len(inter) != len(lo_mv['active_it'][i - 1])):
            print(f"-- active set flipped at iteration {i + 1}")

psi_at_fp = lo_mv['psi']
phi_at_fp = lo_mv['phi']
print(f"\n|phi - psi| = {eucn(psi_at_fp - phi_at_fp)}\n")

print("Theta/Omega.\n")

print(f"LS theta: {np.round(ls_mv['theta'], 3)}")
print(f"LO theta: {np.round(lo_mv['theta'], 3)}")
print(f"LS omega: <to be implemented>")
print(f"LO omega: {np.round(lo_mv['omega'], 3)}\n")

print("Lagrange multpliers.\n")

print(f"B.T x: {np.round(lo_mv['BTx'], 4)}")
print(f"lam : {np.round(lo_mv['lam'], 4)}")
print(f"G.T x: {np.round(lo_mv['GTx'], 4)}")
print(f"eta : {np.round(lo_mv['eta'], 4)}")

if (lo_var > lo_var_numeric):
    print("****************************************")
    print("* Warning! Numerical minimum is lower. *")
    print("****************************************")

print("execution time: %s seconds" % (time.time() - start_time))
