import numpy as np
import pandas as pd
import math 
from sklearn.model_selection import RepeatedKFold
from sklearn.datasets import fetch_california_housing
from sklearn import linear_model
from sklearn import model_selection
from sklearn import metrics
from sklearn.model_selection import  cross_validate, cross_val_score
from sklearn.ensemble import AdaBoostRegressor  , ExtraTreesRegressor   ,HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.ensemble import IsolationForest   , RandomForestRegressor   , VotingRegressor ,BaggingRegressor
from sklearn.linear_model import LogisticRegression,Ridge,RidgeCV,MultiTaskLassoCV,LassoLars,LassoCV
from sklearn.dummy import DummyRegressor
from sklearn import  neighbors
from sklearn.svm import SVR,NuSVR,LinearSVR
from sklearn.linear_model import SGDRegressor ,LinearRegression 
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessRegressor
from xgboost import XGBRFRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error,explained_variance_score
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import make_scorer
from sklearn import preprocessing
import warnings
from warnings import simplefilter
warnings.filterwarnings("ignore")
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=Warning)
### CONSTANTS
SEED = 123
NUM_FEATURES = 4
TRAIN_PCT = 0.8
MAX_DEPTH = 4
MAX_ITER = 500
N_NEIGHBORS = 5
score_all = []


def maxr2(score_all):
    a = np.array(score_all)
    maxR2=np.amax(a[:,0])
    print("maximum R2")
    print(maxR2)
    
def calculateMetrics(results, model_name):
   print(model_name)
   #print(results)
   S=[round(np.mean(results['test_R2']), 4), round(np.mean(results['test_MSE']), 4), round(np.mean(results['test_MSE']), 4),round(np.mean(results['test_var_score']), 4)]
   print("R2 score: {0:.2%} (+/- {1:.2%})".format(np.mean(results['test_R2']), np.std(results['test_R2']) * 2))
   print("MSE score: {0:.2%} (+/- {1:.2%})".format(np.mean(results['test_MSE']), np.std(results['test_MSE']) * 2))
   print("MAE score: {0:.2%} (+/- {1:.2%})".format(np.mean(results['test_MAE']), np.std(results['test_MAE']) * 2))
   print("VAR score: {0:.2%} (+/- {1:.2%})".format(np.mean(results['test_var_score']), np.std(results['test_var_score']) * 2))
   
   score_all.append (S)
   if len(score_all)==22:
       maxr2(score_all)


def train_test_model(model_name, model, X, y):
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scoring = {'R2': make_scorer(r2_score), 'MAE': make_scorer(mean_absolute_error),
           'MSE': make_scorer(mean_squared_error), 'var_score': make_scorer(explained_variance_score)}
    results = model_selection.cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
    calculateMetrics(results, model_name)

def evaluateIndividualregressors(X, y):
    """
    evaluateIndividualregressors
        x : The features of the dataset to be used for predictions
        y : The target class for each row in "x"
        train_size_pct : {float in the range(0.0, 1.0)} the percentage of the dataset that should be used for training
    """
    max_depth_x2 = MAX_DEPTH * 2
    n_neighbors_x2 = N_NEIGHBORS * 2

    lr1=LinearRegression() 
    rf_x2 = RandomForestRegressor(max_depth=max_depth_x2, random_state=SEED)
    et = ExtraTreesRegressor(max_depth=MAX_DEPTH, random_state=SEED)
    dectree = DecisionTreeRegressor(max_depth=MAX_DEPTH, random_state=SEED)
    knn = KNeighborsRegressor(n_neighbors=N_NEIGHBORS)
    knn_x2 = KNeighborsRegressor(n_neighbors=n_neighbors_x2)
    knn3=KNeighborsRegressor(n_neighbors=20,metric='euclidean')
    dumm=DummyRegressor()
    knb=neighbors.KNeighborsRegressor()
    #SVR1=MultiOutputRegressor(NuSVR()) #for multioutput
    SVR1=NuSVR()
    #ada1 = MultiOutputRegressor(AdaBoostRegressor()) #for multipoutput
    ada1 = AdaBoostRegressor()
    gpc1=GaussianProcessRegressor()
    bag= BaggingRegressor(base_estimator=ExtraTreesRegressor(),n_estimators=10, random_state=0)
    #svr1 = MultiOutputRegressor(SVR()) #for multipoutput
    svr1 = SVR()
    r1=Ridge()
    r2=RidgeCV()
    #xgbrf=MultiOutputRegressor(XGBRFRegressor()) #for multipoutput
    #xgb=MultiOutputRegressor(XGBRegressor())    #for multipoutput
    xgbrf=XGBRFRegressor()
    xgb=XGBRegressor()
    """gbr = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,loss='squared_error'))
    lasso= MultiTaskLassoCV(random_state=42)
    Bay= MultiOutputRegressor(linear_model.BayesianRidge())
    linsvr=MultiOutputRegressor(LinearSVR(max_iter=2000))"""
    gbr = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,loss='squared_error')
    lasso= LassoCV(random_state=42)
    Bay= linear_model.BayesianRidge()
    lassolars= linear_model.LassoLars(alpha=.1, normalize=False)
    linsvr=LinearSVR(max_iter=2000)
    regressor_mapping = {
        f'1-linear regression' : lr1,
        f'2-RandomForest case2-{max_depth_x2}' : rf_x2,
        f'3-ExtraTrees-{MAX_DEPTH}' : et,
        f'4-DecisionTree-{MAX_DEPTH}' : dectree,
        f'5-KNeighbors case1-{N_NEIGHBORS}' : knn,
        f'5-KNeighbors case2-{n_neighbors_x2}' : knn_x2,
        f'6-knn case 3' : knn3,
        f'7-dummy-' : dumm,
        f'8-neighbors.KNeighbors-' : knb,
        f'9-NuSVR-' : SVR1,
        f'10- adaboost-'  : ada1,
        f'11- GaussianProcessRegressor'  :  gpc1,
        f'12- bagging'  :  bag,
        f'13- svr1'  : svr1,
        f'14- ridge'  : r1,
        f'15- ridgecv'  : r2,
        f'16- xgbrf'  : xgbrf,
        f'17- xgboost'  : xgb,  
        f'18- GradientBoosting'  : gbr,
        f'19- lasso'  : lasso,
        f'20- BayesianRidge'  : Bay,
        f'21- lassolars'  : lassolars,
        f'22- linsvr'  : linsvr  
    }

    for model_name, model in regressor_mapping.items():

        train_test_model(model_name, model, X, y)


#db

#df = pd.read_csv('C:/Users/amr_r/Desktop/civil/DATASET/new3out2.csv')  # load data set
#df.dropna(inplace=True)
#df.describe()
#X = df.iloc[:,0:4]# 
#y = df.iloc[:,4:8]# 
#x.shape
#y.shape
#df.describe()
#df.hist()
housing = fetch_california_housing()
X=housing.data[0:100,:]
y=housing.target[0:100]

# Look at the dataset again
#print(f'Number of Rows: {df.shape[0]}')
#print(f'Number of Columns: {df.shape[1]}')
#print(df.head())

print(f'[*] Beginning evaluations: All Features')
evaluateIndividualregressors(X, y)











