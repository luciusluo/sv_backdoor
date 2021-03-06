

"""
Created: Wed Aug 17 10:30:00 2021
@author: Yuxiao
"""
# -*- coding: utf-8 -*-
######### Required Modules ##########
from Data.data_handler import data_handler, print_df_instring
from Analyzer import calculator
#from Analyzer import plotter
#####################################


if __name__ == "__main__":
    yahoo_data_csv = data_handler("index05312021.csv")
    yahoo_daily = yahoo_data_csv.get_price("daily", "1995-02-28", "2000-02-28")
    yahoo_weekly = yahoo_data_csv.get_price("weekly", "2000-02-28", "2010-02-28")
    yahoo_monthly = yahoo_data_csv.get_price("monthly", "2000-02-28", "2010-02-28")

    daily_return = yahoo_data_csv.get_return("daily", start_date="2000-02-28", end_date="2010-02-28")
    weekly_return = yahoo_data_csv.get_return("weekly", start_date="2000-02-28", end_date="2010-02-28")
    #monthly_return = yahoo_data_csv.get_return("monthly", start_date="2000-02-28", end_date="2002-02-28")

    print(yahoo_daily)

    #print(daily_return)
    #print(weekly_return)
    #print(monthly_return.shape)

    """


    
    print(daily_return)
    print(weekly_return)
    print(monthly_return)
    
    
    daily_cov = calculator.cov_matrix(daily_return)
    daily_cor = calculator.cor_matrix(daily_cov)
    print(daily_cor)

    
    plotter.plot(yahoo_data_csv, "pca", "daily", "time", "volatility", 1)
    plotter.plot(yahoo_data_csv, "pca", "daily", "dispersion", "volatility", 2)
    plotter.show()

    # print_df_instring(yahoo_data_csv.daily_return)

    daily_volatility = helper.volatility(yahoo_data_csv, "pca", "daily")
    print(daily_volatility)
    years = range(yahoo_data_csv.start_year, yahoo_data_csv.end_year+1)
    
    # dis = helper.dispersion(yahoo_data_csv, "pca", "daily")
    # print(dis)
    """
