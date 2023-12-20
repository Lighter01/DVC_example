import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dvclive import Live

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

import joblib
import json
import sys

# if len(sys.argv) != 3:
#     sys.stderr.write('Arguments error. Usage:\n')
#     sys.stderr.write(
#         '\tpython3 evaluate.py test-path model-path\n'
#     )
#     sys.exit(1)

# test_path = sys.argv[1]
# model_path = sys.argv[2]

test_path  = 'data/prepared/test.csv'
model_path = 'models/logistic_regression.pkl'

df_test = pd.read_csv(test_path, sep=',', header=0)

model = joblib.load(model_path)

X = df_test.drop(columns=['target'])
y = df_test.target

scaler = StandardScaler()
X = scaler.fit_transform(X)

predictions = model.predict(X)

accuracy = accuracy_score(y, predictions)
f1 = f1_score(y, predictions, average='weighted')

# print(f'The accuracy of the Logistic Regression is {accuracy * 100:.4f}%')
# print(f'The f1 score of the Logistic Regression is {f1 * 100:.4f}%')

cm = confusion_matrix(y, predictions)
    
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=np.unique(y), yticklabels=np.unique(y))
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
    
f_importance = np.abs(model.coef_[0])
f_names = df_test.drop(columns=['target']).columns
datapoints = [{"name": name, "importance": importance} for name, importance in zip(f_names, f_importance)]

with Live() as live:
    live.log_metric("test accuracy", accuracy, plot=True)
    live.log_metric("test f1 score", f1, plot=True)
    live.log_sklearn_plot(
        "confusion_matrix", 
        y, 
        predictions, 
        name="test_confusion_matrix",
        title="Test Confusion Matrix")

    live.log_plot(
        "iris_feature_importance",
        datapoints,
        x="importance",
        y="name",
        template="bar_horizontal",
        title="Iris Dataset: Feature Importance",
        y_label="Feature Name",
        x_label="Feature Importance"
    )
