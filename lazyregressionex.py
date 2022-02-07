from lazypredict.Supervised import LazyRegressor
from sklearn import datasets
#from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import numpy as np

boston = datasets.load_boston()
X=boston.data
y=boston.target
X = X.astype(np.float32)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state =123)

reg = LazyRegressor(verbose=0, ignore_warnings=False, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

print(models)