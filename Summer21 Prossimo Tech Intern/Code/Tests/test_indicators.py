import sys, os
sys.path.append(os.path.abspath(os.path.join('..', '')))
from Data.data_handler import data_handler
from Analyzer import calculator

if __name__ == "__main__":
    yahoo_data_csv = data_handler("index05312021.csv")
    daily_return = yahoo_data_csv.get_return("daily")
    weekly_return = yahoo_data_csv.get_return("weekly")
    monthly_return = yahoo_data_csv.get_return("monthly")

    # Test cov, cor, and ave_cor
    daily_cov = calculator.cov_matrix(daily_return)
    daily_cor = calculator.cor_matrix(daily_cov)
    daily_ave_cor = calculator.ave_corr(daily_cov)

    print(daily_cov)
    print(daily_cor)
    print(daily_ave_cor)
