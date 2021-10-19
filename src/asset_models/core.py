#!/usr/bin/env python3
"""! @brief Asset return covariance models. """

##
# @mainpage Asset return covariance models
#
# @section description_main Description
# A collection of methods to estimate covariance models for
# asset returns (PCA, MTFA, etc).
#
# @section notes_main Notes
# - None.
#
# Copyright (c) 2021 Prossimo Tech Inc.  All rights reserved.

##
# @file models.py
#
# @brief Asset returns covariance models manager
#
# @section description_doxygen_example Description
# Estimates models of the form (Pi, Psi, Omega) 
# <am_standards link>
#
# @section libraries_main Libraries/Modules
# - PCA
# - MTFA
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

from datetime import datetime, timedelta
from helpers import *
import pickle

DEBUG = 1
MODELS = {'PCA', 'MTFA'}

from PCA import *


def init():
    """! Initializes."""
    if DEBUG:
        print("Debug mode on ...")
    
    print("-------------------------------------")
    print(" Initializing PCA module : ", end='')
    fancy_wait(0)


def main():
    """! Main program entry."""
    init()  

    waste_time = 5

    n = 50   # num observations
    p = 100  # num assets
    q = 5    # num factors

    Y = sample_basic_return(n,p,q)
    
    end_time = datetime.now()
    start_time = end_time - timedelta(days=n)
 
    # Deconstruct PCA model, input options + outputs

    return_data = dict()
    return_data.update(n=n)
    return_data.update(p=p)
    return_data.update(source='basic_return_sampler_v1')
    return_data.update(data=Y)
    return_data.update(freq='day')
    return_data.update(start = start_time)
    return_data.update(end = end_time)

    print("-------------------------------------")
    print(" Unpack return data : ", end='')
    fancy_wait(waste_time)
    print("")
    for key, val in return_data.items():
        print(f"{key} = {str(val)}")
    print("")
    
    options = dict()
    options.update(debug=DEBUG)
    options.update(number_factors=5)
    
    adj_one = dict()
    adj_one.update(factor_id=1)
    adj_one.update(type='james-stein')
    adj_two = dict()
    adj_two.update(factor_id=2)
    adj_two.update(type='correlation-matrix')
    adj_list = list([adj_one, adj_two])
    
    options.update(exposure_adjustments = adj_list)

    adj_one = dict()
    adj_one.update(factor_id=1)
    adj_one.update(type='marchenko-pastur')
    adj_list = list([adj_one])
    
    options.update(variance_adjustments = adj_list)

    adj_one = dict()
    adj_one.update(factor_id=0)
    adj_one.update(type='mimick-trace')
    adj_list = list([adj_one])

    print("-------------------------------------")
    print(" Unpack pca options : ", end='')
    fancy_wait(waste_time)
    print("")
    for key, val in options.items():
        print(f"{key} = {str(val)}")
    print("")
    

    print("-------------------------------------")
    print(" Estimate PCA model : ", end='')
    fancy_wait(waste_time)
    print("")
   
    model = pca_model(return_data, options)
    
    print("-------------------------------------")
    print(" Pickle-ing model : ", end='')
    pickle.dump(model, open("pca.p", "wb" ) )
    fancy_wait(waste_time)
    print("")

    #model = pickle.load(open("pca.p", "rb"))
 


if __name__ == "__main__":
    main()




