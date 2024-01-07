import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier  # Import Extra Trees classifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from bayes_opt import BayesianOptimization

df = pd.read_csv('D:/new researches/autism paper/Databases/datasets/DB5/Autism_Prediction/DB5_preprocessed.csv')
a = df.describe()

# Basic data preparation
X = np.array(df.drop(['class'], 1))  # Input
y = np.array(df['class'])  # Output

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a black_box_function specific to Extra Trees
def black_box_function(n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
    model = ExtraTreesClassifier(
        n_estimators=int(n_estimators),
        max_depth=int(max_depth),
        min_samples_split=int(min_samples_split),
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

# Set the parameter bounds for optimization
pbounds = {
    "n_estimators": (10, 200),
    "max_depth": (1, 100),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 20),
    "max_features": (0.1, 1.0),
}

# Create a BayesianOptimization optimizer
optimizer = BayesianOptimization(f=black_box_function, pbounds=pbounds, verbose=2, random_state=4)

# Perform the optimization
optimizer.maximize(init_points=20, n_iter=100)

# Get the best result
best_params = optimizer.max["params"]
best_accuracy = optimizer.max["target"]

print("Best result: Parameters={}, Accuracy={:.2f}".format(best_params, best_accuracy))
