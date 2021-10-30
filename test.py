import pandas as pd
import requests
import json
import os
# Чтобы проверить как работает API нужно раскоментировать код. Classification для классификации и linear_regression для регрессии
# classification.
DF_train_cl = pd.read_csv('train.csv')
DF_test_cl = pd.read_csv('test.csv')
headers = {'Content-Type': 'application/json'}

json_1 = {'data': DF_train_cl.to_json(), 'args': {'max_d': 3}, 'target': 'Survived', 'model_name': 'my_model'}
respons = requests.post('http://127.0.0.1:5000/api/ml_models/train/1', json=json_1)
print(respons.json())

json_2 = {'data': DF_test_cl.to_json(), 'model_name': 'my_model'}
respons = requests.post('http://127.0.0.1:5000/api/ml_models/test/1', json=json_2)
print(respons.json())


# linear_regression
# DF_train_lr = pd.read_csv('boston_train.csv')
# DF_test_lr = pd.read_csv('boston_test.csv')
# headers = {'Content-Type': 'application/json'}
#
# json_1 = {'data': DF_train_lr.to_json(), 'args': {}, 'target': 'medv', 'model_name': 'my_model2'}
# respons = requests.post('http://127.0.0.1:5000/api/ml_models/train/2', json=json_1)
# print(respons.json())
#
# json_2 = {'data': DF_test_lr.to_json(), 'model_name': 'my_model2'}
# respons = requests.post('http://127.0.0.1:5000/api/ml_models/test/2', json=json_2)
# print(respons.json())
#
# json_3 = {'model_name': 'my_model2'}
# requests.delete('http://127.0.0.1:5000/api/ml_models/train/2', json=json_3)

