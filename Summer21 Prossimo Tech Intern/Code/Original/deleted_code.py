
"""
# The r part. For graphical model I guess.
from rpy2.robjects import r
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri

rpy2.robjects.numpy2ri.activate()
utils = importr('utils')
utils.install_packages('glasso')  # Graphical Lasso
pd.options.display.float_format = '{:,.4f}'.format


# Chang Yuan's code for Volatility and my code for Volatility and Dispersion. 
#   Very likely abandonned!

from scipy.linalg import eigh
def pca(S, rank): # S is symmetric positive semi-definite
    eigenvalues, eigenvectors = eigh(S)
    eigenvalues_ = eigenvalues[:-(rank+1):-1] # last 'rank' elements in reversed order (large to small) 
    eigenvectors_ = (eigenvectors.T)[:-(rank+1):-1] # corresponding eigenvectors
    eigenvectors_[eigenvectors_.mean(axis = 1) < 0] *= -1 # orient in the way that the sum of eigenvectors >= 0
    d = np.diag(S - np.dot(eigenvectors_.T @ np.diag(eigenvalues_), eigenvectors_))
    return eigenvalues_, eigenvectors_, d

res = []
df = df.dropna(axis = 'columns')
years = range(start, end+1)
for year in years:
    df_ = df[df.index.year == year]
    S = r2S(df_) # sample cov
    v, _, _ = pca(S, rank = 1)
    res.append(v)
plt.plot(years, res)
plt.xlabel('year')
plt.ylabel('variance')
plt.show()

# Input: 
#   handler: class data_handler.
#   model: string. Specifies what model you want to use.
#   freq: string. Specifies what freuqency of the data you want to calculate
# Return: numpy.ndarray containing the volatility of the raw_data stored in the data_handler


def volatility(handler, model, freq):
    # Volatility using PCA refers to the eigen-portfolio variance: inner product (h, Sh) 
    # where h is the eigenvector and S is the sample covariance matrix. So in fact, 
    # the volatility is just the first eigenvalue. 
    if model == "pca":
        # volatility: 1*n array where each element refers to the first eigenvalue of each year
        volatility = []
        years = range(handler.start_year, handler.end_year+1)
        return_data = None
        if freq == "daily":
            return_data = handler.daily_return
        elif freq == "weekly":
            return_data = handler.weekly_return
        else:
            return_data = handler.monthly_return
        # Compute volatility vector
        for year in years:
            single_year_return_data = return_data[return_data.index.year == year]
            S = cov_matrix(single_year_return_data) # sample cov
            eigenvalue, _, _ = pca.pca(S, rank=1)
            volatility.append(eigenvalue[0])
        
        return volatility
    else:
        pass


# Input:
#   S: sample covariance matrix
def dispersion(handler, model, freq):
    dis = []
    years = range(handler.start_year, handler.end_year+1)
    return_data = None
    if freq == "daily":
        return_data = handler.daily_return
    elif freq == "weekly":
        return_data = handler.weekly_return
    else:
        return_data = handler.monthly_return
    # Compute volatility vector
    for year in years:
        single_year_return_data = return_data[return_data.index.year == year]
        S = cov_matrix(single_year_return_data)
        _, vec = eigsh(S, 1, which='LA')
        if (np.sum(vec) < 0):
            vec = -1.0 * vec
        vave = np.mean(vec)
        vvar = np.mean((vec - vave)**2)
        dis.append(sqrt(vvar)/vave)

    return dis
"""
    

