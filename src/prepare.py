import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd

# create folder to save file
data_path = os.path.join('data', 'prepared')
os.makedirs(data_path, exist_ok=True)

#fetch data
iris = load_iris(as_frame=True).frame
iris.columns = ['_'.join(str(name)[:-5].strip().split()) for name in iris.columns[:-1]] + ['target']
train, test = train_test_split(iris, test_size=0.3, shuffle=True, stratify=iris.target, random_state=42)

train.to_csv(os.path.join(data_path, "train.csv"), index=False)
test.to_csv(os.path.join(data_path, "test.csv"), index=False)