import sys, os

from numpy.core.fromnumeric import shape
""" 
# This part for normal run
sys.path.append(os.path.abspath(os.path.join('..', '')))
from Data.data_handler import data_handler
"""
# This part for normal run
from Data.data_handler import data_handler
from Model.pca import pca
import numpy as np

if __name__ == "__main__":
    yahoo_data_csv = data_handler("index05312021.csv")
    daily_return = yahoo_data_csv.get_return("daily")
    #weekly_return = yahoo_data_csv.get_return("weekly")
    #monthly_return = yahoo_data_csv.get_return("monthly")

    #smaller_matrix = np.random.rand(100,40)
    
    pca_daily = pca(daily_return, 4, js=1, sped=1, corrad=1)
    print("pca_daily B:", pca_daily["B"])
    print("pca_daily V:", pca_daily["V"])
    print("pca_daily delta:", pca_daily["delta"])