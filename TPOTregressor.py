import numpy as np
import pandas as pd
# example of tpot for the insurance regression dataset
from pandas import read_csv
from sklearn.model_selection import RepeatedKFold
from tpot import TPOTRegressor
# load dataset
df = pd.read_csv('C:/Users/amr_r/Desktop/civil/DATASET/new3out2.csv')  # load data set
df.dropna(inplace=True)
df.describe()
X = df.iloc[:,0:4]# 
y = df.iloc[:,4]# 
# define evaluation procedure
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# define search
model = TPOTRegressor(generations=10, population_size=50, scoring='neg_mean_absolute_error', cv=cv, verbosity=2, random_state=1, n_jobs=-1)
# perform the search
model.fit(X, y)
# export the best model
model.export('tpot_insurance_best_model.py') #
