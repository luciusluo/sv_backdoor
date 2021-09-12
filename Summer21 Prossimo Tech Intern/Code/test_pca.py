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
import matplotlib.pyplot as plt


if __name__ == "__main__":
    yahoo_data_csv = data_handler("index05312021.csv")
    daily_return = yahoo_data_csv.get_return("daily")

    #print(type(yahoo_data_csv.start_date))
    #df = yahoo_data_csv.raw_data.loc["2021-05-22":"2021-05-29"]
    #print(yahoo_data_csv.start_date)
    #print(daily_return)
    #print(yahoo_data_csv.raw_data.index)
    
    pca_daily = pca(daily_return, 2, corrad=1) #, js=1
    #print("pca_daily B:", pca_daily["B"])
    #print("pca_daily V:", pca_daily["V"])
    #print("pca_daily delta:", pca_daily["delta"])

    B = pca_daily["B"]
    beta = B[:, 0]
    print(beta)
    plt.plot(beta)
    plt.show()
    