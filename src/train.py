import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.preprocessing import StandardScaler

import joblib
import os

df_train = pd.read_csv('data/prepared/train.csv', sep=',', header=0)

X_train = df_train.drop(columns=['target'], axis=1)
y_train = df_train.target

print(y_train.shape[0])

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

param_grid = {
    "n_jobs": 4,
    "penalty": 'l2',
    "C": np.logspace(-5, 5, 20)[14],
    "max_iter": 1000,
    "solver": 'lbfgs'
}

log_r = LogisticRegression(multi_class='multinomial', **param_grid)
log_r.fit(X_train, y_train)

predictions = log_r.predict(X_train)

accuracy = accuracy_score(y_train, predictions)
f1 = f1_score(y_train, predictions, average='weighted')

print(f'The accuracy of the Logistic Regression is {accuracy * 100:.4f}%')
print(f'The f1 score of the Logistic Regression is {f1 * 100:.4f}%')

cm = confusion_matrix(y_train, log_r.predict(X_train))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.unique(y_train), yticklabels=np.unique(y_train))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')

new_directory = 'models/'
os.makedirs(new_directory, exist_ok=True)

joblib_logr = 'models/logistic_regression.pkl'
joblib.dump(log_r, joblib_logr)