import pandas as pd


data = pd.read_csv("F:\\corrected.csv", header=None, index_col=False)

data_dummies = pd.get_dummies(data)

print(data_dummies.head())
