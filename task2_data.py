"""
Task 2. RNN

Develop RNN, GRU and LSTM to predict Usage_kWh. Dataset - http://archive.ics.uci.edu/dataset/851/steel+industry+energy+consumption.

Hyperparameters are at your discretion

Compare the quality of the MSE, RMSE and R^2 models
"""

from ucimlrepo import fetch_ucirepo
from sklearn.preprocessing import LabelEncoder

# fetch dataset
steel_industry_energy_consumption = fetch_ucirepo(id=851)

# data (as pandas dataframes)
X = steel_industry_energy_consumption.data.features
y = steel_industry_energy_consumption.data.targets

encoder = LabelEncoder()
encoder.fit_transform(X['WeekStatus'].to_frame())
X['WeekStatus'] = encoder.transform(X['WeekStatus'].to_frame())

encoder = LabelEncoder()
encoder.fit_transform(X['Day_of_week'].to_frame())
X['Day_of_week'] = encoder.transform(X['Day_of_week'].to_frame())

encoder = LabelEncoder()
encoder.fit_transform(y['Load_Type'].to_frame())
y['Load_Type'] = encoder.transform(y['Load_Type'].to_frame())

dataset = X
dataset['Load_Type'] = y['Load_Type']
dataset.to_csv('kw_dataset.csv', index=False)
