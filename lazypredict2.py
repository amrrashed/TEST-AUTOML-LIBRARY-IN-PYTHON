#link https://towardsdatascience.com/how-to-run-40-regression-models-with-a-few-lines-of-code-5a24186de7d
#https://towardsdatascience.com/how-to-run-30-machine-learning-models-with-2-lines-of-code-d0f94a537e52
# Importing important libraries
import numpy as np
import pandas as pd  # To read data
from lazypredict.Supervised import LazyRegressor
from pandas.plotting import scatter_matrix
# Scikit-learn packages
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
# Hide warnings
import warnings
warnings.filterwarnings("ignore")
# Setting up max columns displayed to 100
pd.options.display.max_columns = 100
data = pd.read_csv('G:/new researches/civil/DATASET/DB2.csv')  # load data set
data.dropna(inplace=True)
a=data.describe()
X1 = data.iloc[:,0:7]# 0:4 for DB1
# normalize input
normz = MinMaxScaler()
X = normz.fit_transform(X1)
#X = preprocessing.normalize(X1) 
## standarize input
ss = StandardScaler()
X2 = ss.fit_transform(X1)

y = data.iloc[:,7]# 4:8 FOR DB1
# Call train_test_split on the data and capture the results
X_train, X_test, y_train, y_test = train_test_split(X1, y, random_state=3,test_size=0.2)
reg = LazyRegressor(ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)
print(models)
