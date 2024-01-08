import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from scipy.stats import randint, uniform

class HyperparameterOptimizer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.X = np.array(self.df.drop(['class'], 1))
        self.y = np.array(self.df['class'])
        self.results_df = pd.DataFrame(columns=['Model', 'Parameters', 'Test Accuracy'])

    def optimize_hyperparameters(self, model_name, model_creation_function, param_space, n_iter=100):
        search = RandomizedSearchCV(model_creation_function(), param_space, n_iter=n_iter, scoring='accuracy', n_jobs=-1, cv=5, random_state=1)

        # Execute the search
        result = search.fit(self.X, self.y)

        # Get the best hyperparameters
        best_params = result.best_params_

        # Save results to DataFrame
        self.results_df = self.results_df.append({
            'Model': model_name,
            'Parameters': best_params,
            'Accuracy': result.best_score_
        }, ignore_index=True)

        print(f"Best result for {model_name}: Parameters={best_params}, Accuracy={result.best_score_:.2f}")

    def save_results_to_csv(self, file_path):
        self.results_df.to_csv(file_path, index=False)


# Example usage:
data_path = 'D:/new researches/autism paper/Databases/datasets/DB5/Autism_Prediction/DB5_preprocessed.csv'

# Define parameter bounds for each model
random_forest_bounds = {
    "n_estimators": randint(10, 200),
    "max_depth": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

xgboost_bounds = {
    "learning_rate": (0.01, 0.3),
    "n_estimators": randint(10, 200),
    "max_depth": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "min_child_weight": [1, 2, 3, 4, 5],
    "subsample": uniform(0.5, 1.0),
    "colsample_bytree": uniform(0.5, 1.0),
}

svm_bounds = {
    "C": uniform(0.1, 10),
    "gamma": uniform(0.001, 1),
}

mlp_bounds = {
    "hidden_layer_sizes": randint(20, 100),
    "alpha": uniform(0.0001, 0.1),
    "max_iter":[1000],
}

naive_bayes_bounds = {
    "var_smoothing": uniform(1e-9, 1e-1),
}

extra_trees_bounds = {
    "n_estimators": randint(10, 200),
    "max_depth": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": uniform(0.1, 1.0),
}

# Create an instance of HyperparameterOptimizer
optimizer = HyperparameterOptimizer(data_path)

# Optimize hyperparameters for each model
optimizer.optimize_hyperparameters("RandomForest", RandomForestClassifier, random_forest_bounds)
optimizer.optimize_hyperparameters("XGBoost", XGBClassifier, xgboost_bounds)
optimizer.optimize_hyperparameters("SVM", SVC, svm_bounds)
optimizer.optimize_hyperparameters("MLP", MLPClassifier, mlp_bounds)
optimizer.optimize_hyperparameters("NaiveBayes", GaussianNB, naive_bayes_bounds)
optimizer.optimize_hyperparameters("ExtraTrees", ExtraTreesClassifier, extra_trees_bounds)

# Save results to CSV
results_file_path = 'D:/new researches/autism paper/optimization_results3.csv'
optimizer.save_results_to_csv(results_file_path)
