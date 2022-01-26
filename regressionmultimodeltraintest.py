import numpy as np
import pandas as pd
import math 
from sklearn import linear_model
from sklearn.datasets import fetch_california_housing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor  , ExtraTreesRegressor   ,HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.ensemble import IsolationForest   , RandomForestRegressor   , VotingRegressor ,BaggingRegressor
from sklearn.linear_model import LogisticRegression,Ridge,RidgeCV,MultiTaskLassoCV
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
    
def calculateMetrics(y_test, y_pred, model):
   # predictions = [round(value) for value in y_pred]
    R2 = r2_score(y_test, y_pred)
    MAE = mean_absolute_error(y_test, y_pred)
    MSE = mean_squared_error(y_test, y_pred)
  #  RMSE=root_mean_squared_error(y_test, y_pred)
    var=explained_variance_score(y_test, y_pred)
    S=[round(R2, 4), round(MAE, 4), round(MSE, 4),round(var, 4)]
    score_all.append (S)
    print("R2", "MAE", "MSE","var")
    print(S)
    if len(score_all)==23:
        maxr2(score_all)


def train_test_model(model_name, model, x, y, train_size_pct):

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=train_size_pct,random_state=SEED)
    #scaler = StandardScaler()   
    #X_train= scaler.fit_transform(X_train)
    #X_test= scaler.transform(X_test)
    # Training
    print(f'\n {model_name}')
    model.fit(X_train, Y_train)

    ### Testing
    y_pred = model.predict(X_test)
    ### Analyze Testing
    calculateMetrics(Y_test, y_pred, model)

def evaluateIndividualregressors(x, y, train_size_pct):
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
    SVR1=MultiOutputRegressor(NuSVR())
    ada1 = MultiOutputRegressor(AdaBoostRegressor())
    gpc1=GaussianProcessRegressor()
    bag= BaggingRegressor(base_estimator=ExtraTreesRegressor(),n_estimators=10, random_state=0)
    svr1 = MultiOutputRegressor(SVR())
    r1=Ridge()
    r2=RidgeCV()
    xgbrf=MultiOutputRegressor(XGBRFRegressor())
    xgb=MultiOutputRegressor(XGBRegressor())
    gbr = MultiOutputRegressor(GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=1, random_state=0,loss='squared_error'))
    lasso= MultiTaskLassoCV(random_state=42)
    Bay= MultiOutputRegressor(linear_model.BayesianRidge())
    lassolars= linear_model.LassoLars(alpha=.1, normalize=False)
    linsvr=MultiOutputRegressor(LinearSVR())
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

        train_test_model(model_name, model, x, y, train_size_pct)

#db

df = pd.read_csv('C:/Users/amr_r/Desktop/civil/DATASET/new3out2.csv')  # load data set
df.dropna(inplace=True)
df.describe()
X = df.iloc[:,0:4]# 
Y = df.iloc[:,4:8]# 
#X.shape
#Y.shape
df.describe()
df.hist()


# Look at the dataset again
print(f'Number of Rows: {df.shape[0]}')
print(f'Number of Columns: {df.shape[1]}')
print(df.head())

print(f'[*] Beginning evaluations: All Features')
evaluateIndividualregressors(X,Y, TRAIN_PCT)











