import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.models import Sequential
from keras.layers import Conv1D, LSTM, Dense, Flatten
from bayes_opt import BayesianOptimization

class HyperparameterOptimizer:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path)
        self.X = np.array(self.df.drop(['class'], 1))
        self.y = np.array(self.df['class'])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        self.results_df = pd.DataFrame(columns=['Model', 'Parameters', 'Accuracy'])

    def black_box_extra_trees(self, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        model = ExtraTreesClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            random_state=42,
        )
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def black_box_random_forest(self, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features):
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            max_features=max_features,
            random_state=42,
        )
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def black_box_svm(self, C, gamma):
        model = SVC(C=C, gamma=gamma, random_state=42)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def black_box_mlp(self, hidden_layer_sizes, alpha):
        model = MLPClassifier(
            hidden_layer_sizes=int(hidden_layer_sizes),
            alpha=alpha,
            random_state=42,
        )
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def black_box_xgboost(self, learning_rate, n_estimators, max_depth, min_child_weight, subsample, colsample_bytree):
        model = XGBClassifier(
            learning_rate=learning_rate,
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=42,
        )
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def black_box_naive_bayes(self, var_smoothing):
        model = GaussianNB(var_smoothing=var_smoothing)
        model.fit(self.X_train, self.y_train)
        y_pred = model.predict(self.X_test)
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def black_box_cnn_lstm(self, filters, kernel_size, lstm_units, dense_units, dropout):
        model = Sequential()
        model.add(Conv1D(filters=int(filters), kernel_size=int(kernel_size), activation='relu', input_shape=(self.X_train.shape[1], 1)))
        model.add(LSTM(int(lstm_units)))
        model.add(Dense(int(dense_units), activation='relu'))
        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        # Reshape the input data for CNN+LSTM
        X_train_reshaped = self.X_train.reshape((self.X_train.shape[0], self.X_train.shape[1], 1))
        X_test_reshaped = self.X_test.reshape((self.X_test.shape[0], self.X_test.shape[1], 1))

        model.fit(X_train_reshaped, self.y_train, epochs=10, batch_size=32, verbose=0)

        y_pred = (model.predict(X_test_reshaped) > 0.5).astype(int).flatten()
        accuracy = accuracy_score(self.y_test, y_pred)
        return accuracy

    def optimize_hyperparameters(self, model_name, pbounds, init_points=20, n_iter=100):
        if model_name == "extra_trees":
            black_box_function = self.black_box_extra_trees
        elif model_name == "random_forest":
            black_box_function = self.black_box_random_forest
        elif model_name == "svm":
            black_box_function = self.black_box_svm
        elif model_name == "mlp":
            black_box_function = self.black_box_mlp
        elif model_name == "xgboost":
            black_box_function = self.black_box_xgboost
        elif model_name == "naive_bayes":
            black_box_function = self.black_box_naive_bayes
        elif model_name == "cnn_lstm":
            black_box_function = self.black_box_cnn_lstm
        else:
            raise ValueError("Invalid model name")

        optimizer = BayesianOptimization(
            f=black_box_function,
            pbounds=pbounds,
            verbose=2,
            random_state=4,
        )

        optimizer.maximize(init_points=init_points, n_iter=n_iter)

        best_params = optimizer.max["params"]
        best_accuracy = optimizer.max["target"]

        # Save results to DataFrame
        self.results_df = self.results_df.append({
            'Model': model_name,
            'Parameters': best_params,
            'Accuracy': best_accuracy
        }, ignore_index=True)

        print(f"Best result for {model_name}: Parameters={best_params}, Accuracy={best_accuracy:.2f}")

    def save_results_to_csv(self, file_path):
        self.results_df.to_csv(file_path, index=False)


# Example usage:
data_path = 'D:/new researches/autism paper/Databases/datasets/DB5/Autism_Prediction/DB5_preprocessed.csv'

# Define parameter bounds for each model
extra_trees_bounds = {
    "n_estimators": (10, 200),
    "max_depth": (1, 100),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 20),
    "max_features": (0.1, 1.0),
}

random_forest_bounds = {
    "n_estimators": (10, 200),
    "max_depth": (1, 100),
    "min_samples_split": (2, 20),
    "min_samples_leaf": (1, 20),
    "max_features": (0.1, 1.0),
}

svm_bounds = {
    "C": (0.1, 10),
    "gamma": (0.001, 1),
}

mlp_bounds = {
    "hidden_layer_sizes": (5, 100),
    "alpha": (0.0001, 0.1),
}

xgboost_bounds = {
    "learning_rate": (0.01, 0.3),
    "n_estimators": (10, 200),
    "max_depth": (1, 10),
    "min_child_weight": (1, 10),
    "subsample": (0.5, 1.0),
    "colsample_bytree": (0.5, 1.0),
}

naive_bayes_bounds = {
    "var_smoothing": (1e-9, 1e-1),
}

cnn_lstm_bounds = {
    "filters": (16, 64),
    "kernel_size": (3, 5),
    "lstm_units": (50, 150),
    "dense_units": (32, 128),
    "dropout": (0.2, 0.5),
}

optimizer = HyperparameterOptimizer(data_path)

optimizer.optimize_hyperparameters("extra_trees", extra_trees_bounds)
optimizer.optimize_hyperparameters("random_forest", random_forest_bounds)
optimizer.optimize_hyperparameters("svm", svm_bounds)
optimizer.optimize_hyperparameters("mlp", mlp_bounds)
optimizer.optimize_hyperparameters("xgboost", xgboost_bounds)
optimizer.optimize_hyperparameters("naive_bayes", naive_bayes_bounds)
optimizer.optimize_hyperparameters("cnn_lstm", cnn_lstm_bounds)

# Save results to CSV
results_file_path = 'D:/new researches/autism paper/optimization_results.csv'
optimizer.save_results_to_csv(results_file_path)
