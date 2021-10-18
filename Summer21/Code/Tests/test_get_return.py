import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '')))
from Data.data_handler import data_handler

if __name__ == "__main__":
    yahoo_data_csv = data_handler("index05312021.csv")
    daily_return = yahoo_data_csv.get_return("daily")
    weekly_return = yahoo_data_csv.get_return("weekly")
    monthly_return = yahoo_data_csv.get_return("monthly")

    print(daily_return)
    print(weekly_return)
    print(monthly_return)