# -*- coding: utf-8 -*-
"""multiclassifier.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yg8sEE8Kiyvhe5Tpa1sveTaKNYnEhvSn
"""

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn import metrics
#from sklearn.metrics import average_precision_score ,precision_recall_curve,f1_score
from sklearn.model_selection import cross_val_score , train_test_split
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.ensemble import IsolationForest, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, auc, confusion_matrix, f1_score, mean_squared_error, plot_confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.dummy import DummyClassifier
from sklearn import  neighbors
from sklearn.svm import SVC, NuSVC, LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process import GaussianProcessClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.multiclass import OneVsRestClassifier, OutputCodeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

### CONSTANTS
SEED = 123
NUM_FEATURES = 9
TRAIN_PCT = 0.8
MAX_DEPTH = 4
MAX_ITER = 1000
N_NEIGHBORS = 5

def calculateMetrics(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average="macro")
    precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    mse = mean_squared_error(y_test, y_pred)
    f1score = f1_score(y_test, y_pred, average='weighted')
    auc1 = roc_auc_score(y_test, y_pred)
    print(">>> Metrics")
    print(f'- Accuracy  : {acc}')
    print(f'- Recall    : {recall}')
    print(f'- Precision : {precision}')
    print(f'- MSE       : {mse}')
    print(f'- F1 Score  : {f1score}')
    print(f'- Auc Score  : {auc1}')
    return [round(acc, 6), round(recall, 6), round(precision, 6), round(mse, 6), round(f1score, 6), round(auc1, 6)]

def train_test_model(model_name, model, x, y, train_size_pct):

    # Split the data
    X_train, X_test, Y_train, Y_test = train_test_split(x, y, train_size=train_size_pct)

    # Training
    print(f'\n[Training] {model_name}')
    start_train = datetime.now()
    model.fit(X_train, Y_train)
    print(f'>>> Training time: {datetime.now() - start_train}')

    ### Analyze Training
    train_acc = model.score(X_train, Y_train)
    print(f'>>> Training accuracy: {train_acc}')

    ### Testing
    start_predict = datetime.now()
    y_pred = model.predict(X_test)
    print(f'>>> Testing time: {datetime.now() - start_predict}')
    ### Analyze Testing
    calculateMetrics(Y_test, y_pred)

def evaluateIndividualClassifiers(x, y, train_size_pct):
    """
    evaluateIndividualClassifiers
        x : The features of the dataset to be used for predictions
        y : The target class for each row in "x"
        train_size_pct : {float in the range(0.0, 1.0)} the percentage of the dataset that should be used for training
    """
    max_depth_x2 = MAX_DEPTH * 2
    max_iter_x2 = MAX_ITER * 2
    max_iter_x10 = MAX_ITER * 10
    n_neighbors_x2 = N_NEIGHBORS * 2
    n_neighbors_d2 = N_NEIGHBORS // 2

    rf = RandomForestClassifier(max_depth=MAX_DEPTH, criterion='entropy', random_state=SEED)
    rf_x2 = RandomForestClassifier(max_depth=max_depth_x2, criterion='entropy', random_state=SEED)
    et = ExtraTreesClassifier(max_depth=MAX_DEPTH, criterion='entropy', random_state=SEED)
    dectree = DecisionTreeClassifier(max_depth=MAX_DEPTH, random_state=SEED)
    knn = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    knn_x2 = KNeighborsClassifier(n_neighbors=n_neighbors_x2)
    knn_d2 = KNeighborsClassifier(n_neighbors=n_neighbors_d2)
    mlpnn = MLPClassifier(max_iter=MAX_ITER)
    mlpnnE = MLPClassifier(max_iter=MAX_ITER, early_stopping=True)
    mlpnn_x2 = MLPClassifier(max_iter=max_iter_x2)
    mlpnnE_x2 = MLPClassifier(max_iter=max_iter_x2, early_stopping=True)
    XGB1=XGBClassifier()
    GNB1=GaussianNB()
    dumm=DummyClassifier()
    knb=neighbors.KNeighborsClassifier()
    LR1=LogisticRegression(max_iter=max_iter_x10)
    SVC1=SVC(max_iter=max_iter_x10)
    ovr1=SGDClassifier(max_iter=max_iter_x2)
    ada1=AdaBoostClassifier()
    gpc1=GaussianProcessClassifier()
    GBclass1=GradientBoostingClassifier(n_estimators=100, learning_rate=1.0,max_depth=1, random_state=0)
    histgclass=HistGradientBoostingClassifier(max_iter=max_iter_x2)
    bagclass=BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
    ridge1 = RidgeClassifier(max_iter=max_iter_x10)
    #Mnb = MultinomialNB()
    SVC2 = NuSVC(max_iter=max_iter_x10)
    linear1=LinearSVC(max_iter=max_iter_x10)
    classifier_mapping = {
        f'1-RandomForest-{MAX_DEPTH}' : rf,
        f'2-RandomForest-{max_depth_x2}' : rf_x2,
        f'3-ExtraTrees-{MAX_DEPTH}' : et,
        f'4-DecisionTree-{MAX_DEPTH}' : dectree,
        f'5-KNeighbors case1-{N_NEIGHBORS}' : knn,
        f'5-KNeighbors case2-{n_neighbors_x2}' : knn_x2,
        f'5-KNeighbors case3-{n_neighbors_d2}' : knn_d2,
        f'6-MLP case1-{MAX_ITER}' : mlpnn,
        f'6-MLP  case2-{MAX_ITER}-early' : mlpnnE,
        f'6-MLP case3-{max_iter_x2}' : mlpnn_x2,
        f'6-MLP case4-{max_iter_x2}-early' : mlpnnE_x2,
        f'7-XGB-' : XGB1,
        f'8-GNB-' : GNB1,
        f'9-dumm-' : dumm,
        f'10-knb-' : knb,
        f'11-LR1-' : LR1,
        f'12-SVC1-' : SVC1,
        f'13-ovr-'  : ovr1,
        f'14-ada-'  : ada1,
        f'15-gpc'  :  gpc1,
        f'16-GBclass':GBclass1,
        f'17-histgclas':histgclass,
        f'18-bagclas':bagclass,
        f'19-ridge' : ridge1,
        f'20-SVC2' : SVC2,
        f'21-linear SVC' : linear1,
    }

    for model_name, model in classifier_mapping.items():

        train_test_model(model_name, model, x, y, train_size_pct)

from google.colab import drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
X = np.array(df.drop(['class'], 1))
X = X.astype('float32')
y = np.array(df['class'])
y = LabelEncoder().fit_transform(y.astype(str))
# Look at the dataset again
print(X.shape, y.shape)
print(df.head())

print(f'[*] Beginning evaluations: All Features')
evaluateIndividualClassifiers(X,y, TRAIN_PCT)