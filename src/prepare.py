import os
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import yaml

def load_params():
    with open('params.yaml', 'r') as file:
        params = yaml.safe_load(file)
    return params

params = load_params()['prepare']
seed = params['seed']
split_size = params['split']

# create folder to save file
data_path = os.path.join('data', 'prepared')
os.makedirs(data_path, exist_ok=True)

#fetch data
iris = load_iris(as_frame=True).frame
iris.columns = ['_'.join(str(name)[:-5].strip().split()) for name in iris.columns[:-1]] + ['target']
train, test = train_test_split(iris, test_size=split_size, shuffle=True, stratify=iris.target, random_state=seed)

train.to_csv(os.path.join(data_path, "train.csv"), index=False)
test.to_csv(os.path.join(data_path, "test.csv"), index=False)