"""
First created: Thu Jul 8 16:51:59 2021
@author: Brennan

Modified: Mon Aug 16 10:30:00 2021
@author: Yuxiao
"""
# -*- coding: utf-8 -*-
######### Required Modules ##########
import pandas as pd
import numpy as np
import os
from pathlib import Path
#####################################


# Helper functions: 
price_2_return = lambda df: (df.shift(periods = -1)/df -1).head(-1)    # compute the return of weekly data in decimals 
price_2_log_return = lambda df: (np.log(df).shift(periods = -1) - np.log(df)).head(-1)  # compute the log return of weekly data in decimals 
print_df_instring = lambda df: print(df.describe().to_string()+'\n')


# ** Combine all functions into one based on user input variations: 
#   get_return(param1, param2, param3...), get_price(param1, param2, param3...)
#   p, n, filename,  
class data_handler:
  filepath = "" # maybe private too
  filename = ""
  datatype = ""
  raw_data = None  # make private ***


  ###### Class constructor of data_handler ######
  def __init__(self, filename):
      self.filename = filename
      self.filepath = Path(__file__).parent/ "stored_data" / self.filename
      # Successful filename and filepath
      if os.path.isfile(self.filepath):
        if filename[-4:] == "json":
          self.datatype = "json"
          self.raw_data = pd.read_json(self.filepath)
        else:
          self.datatype = "csv"
          self.raw_data = pd.read_csv(self.filepath)
      else:
        print("Wrong filepath or filename. Please check again!")


  # Private helper function that returns price in pandas.Dataframe
  def __get_price(self, freq):
    # Get daily from raw_data
    if freq == "daily":
      daily = self.raw_data
      daily.Date = pd.to_datetime(daily.Date)
      daily = daily.set_index('Date') 
      # Drop the columns where at least one element is missing.
      if self.datatype == "json":
        # bc read_json will return a DataFrame containing a string "#N/A N/A"
        # in it, we need to remove those columns just like in csv.
        daily = daily.loc[:, ~(self.daily == "#N/A N/A").any()]
      else:
        daily = daily.dropna(axis = 'columns') 
      return daily
    # Get weekly from daily
    elif freq == "weekly":
      daily = self.__get_price("daily")
      # Select
      weekly = daily.loc[daily.index.weekday == 2]
      return weekly
    # Get monthly price
    elif freq == "monthly":
      weekly = self.__get_price("weekly")
      l = [weekly.index[0]]
      for d in weekly.index:
        if l[-1].month != d.month:
          l.append(d)
      monthly = weekly.loc[pd.DatetimeIndex(l)]
      return monthly


  # Actual function to return price in numpy.ndarray
  def get_price(self, freq):
    if freq not in ["daily", "weekly", "monthly"]:
      raise ValueError("Wrong FREQ value! Please recheck.")
    price = self.__get_price(freq).to_numpy()
    return price


  # Actual function that returns returns in pandas.Dataframe
  def get_return(self, freq, flag="reg"):
    if flag not in ["reg","log"]:
      raise ValueError("Wrong FLAG value! Please recheck.")
    price = self.__get_price(freq)
    price_return = price_2_return(price) if flag == "reg" else price_2_log_return(price)
    return price_return.to_numpy()
