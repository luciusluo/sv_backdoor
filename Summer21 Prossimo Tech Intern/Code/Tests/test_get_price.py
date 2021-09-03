# -*- coding: utf-8 -*-
######### Required Modules ##########
import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '')))
from Data.data_handler import data_handler

if __name__ == "__main__":
    yahoo_data_csv = data_handler("index05312021.csv")
    yahoo_daily = yahoo_data_csv.get_price("daily")
    yahoo_weekly = yahoo_data_csv.get_price("weekly")
    yahoo_monthly = yahoo_data_csv.get_price("monthly")

    print(yahoo_daily)
    print(yahoo_weekly)
    print(yahoo_monthly)