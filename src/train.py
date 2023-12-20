import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import joblib
import os
from dvclive import Live
import yaml

def load_params():
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    return params

params = load_params()['train']
n_jobs   = params['n_jobs']
penalty  = params['penalty']
max_iter = params['max_iter']
solver   = params['solver']


df_train = pd.read_csv('data/prepared/train.csv', sep=',', header=0)

X_train = df_train.drop(columns=['target'], axis=1)
y_train = df_train.target

print(y_train.shape[0])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

param_grid = {
    "n_jobs": n_jobs,
    "penalty": penalty,
    "C": np.logspace(-5, 5, 20)[14],
    "max_iter": max_iter,
    "solver": solver
}

log_r = LogisticRegression(multi_class='multinomial', **param_grid)
log_r.fit(X_train, y_train)

predictions = log_r.predict(X_train)

accuracy = accuracy_score(y_train, predictions)
f1 = f1_score(y_train, predictions, average='weighted')

# print(f'The accuracy of the Logistic Regression is {accuracy * 100:.4f}%')
# print(f'The f1 score of the Logistic Regression is {f1 * 100:.4f}%')

cm = confusion_matrix(y_train, log_r.predict(X_train))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

f_importance = np.abs(log_r.coef_[0])
f_names = df_train.drop(columns=['target']).columns
datapoints = [{"name": name, "importance": importance} for name, importance in zip(f_names, f_importance)]

new_directory = 'models/'
os.makedirs(new_directory, exist_ok=True)

joblib_logr = 'models/logistic_regression.pkl'
joblib.dump(log_r, joblib_logr)

with Live() as live:
    live.log_metric("test accuracy", accuracy, plot=True)
    live.log_metric("test f1 score", f1, plot=True)

    live.log_sklearn_plot(
        "confusion_matrix", 
        y_train, 
        predictions, 
        name="train_confusion_matrix",
        title="Test Confusion Matrix")

    live.log_plot(
        "iris_feature_importance_train",
        datapoints,
        x="importance",
        y="name",
        template="bar_horizontal",
        title="Iris Dataset: Feature Importance",
        y_label="Feature Name",
        x_label="Feature Importance"
    )