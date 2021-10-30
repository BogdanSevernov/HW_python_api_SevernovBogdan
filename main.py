import pandas as pd
from flask import Flask
from flask_restx import Api, Resource
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import joblib
import shutil
import os

app = Flask(__name__)
api = Api(app)

class MLModels():
    def __init__(self):
        self.ml_models = [{'id': 1, 'task': 'classification', 'model_name': 'tree'},
                          {'id': 2, 'task': 'regression', 'model_name': 'linear_regression'}]
        self.model_directory = 'model'

    def data_preprocessing(self, data):
        """Этот метод нужен для предварительной обработки данных"""
        data = data
        # работа с null значениями
        for col in data.columns:
            # Количество пустых значений
            temp_null_count = data[data[col].isnull()].shape[0]
            dt = str(data[col].dtype)
            # Ищем поля с типом 'float64' или 'int64'
            if temp_null_count > 0 and (dt == 'float64' or dt == 'int64'):
                temp_data = data[[col]]
                imp_num = SimpleImputer(strategy='mean')
                data_num_imp = imp_num.fit_transform(temp_data)
                data[col] = data_num_imp
            # Ищем поля с типом 'object'
            elif temp_null_count > 0 and (dt == 'object'):
                temp_data = data[[col]]
                imp_num = SimpleImputer(strategy='most_frequent')
                data_num_imp = imp_num.fit_transform(temp_data)
                data[col] = data_num_imp

        # работа с категориальными признаками
        le = LabelEncoder()
        for col in data.columns:
            dt = str(data[col].dtype)
            if dt == 'object':
                cat_enc_le = le.fit_transform(data[col])
                data[col] = cat_enc_le
        return data

    def train_test_split_fun(self, data, target, id):
        """Метод для разделения данных на тренеровочные и тестовые. Он нужен чтобы выводить метрики качества"""
        if target not in data.columns:
            api.abort(404, 'target={} not in data columns'.format(target))
        data = self.data_preprocessing(data)
        X = data.drop(columns=[str(target)])
        y = data[str(target)]
        if int(id) == 1:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=777, stratify=y)
        elif int(id) == 2:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=777)

        return X_train, X_test, y_train, y_test

    def train_models(self, data, id, target, model_name, **params):
        """Метод для обучения моделей"""
        data = data
        X_train, X_test, y_train, y_test = self.train_test_split_fun(data, target, id)
        if int(id) == 1:
            clf = DecisionTreeClassifier(max_depth=int(params['max_d']), random_state=0)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            acc = round(metrics.accuracy_score(y_test, y_pred), 2)
            model_file_name = '{}/{}_id1.pkl'.format(self.model_directory, model_name)
            joblib.dump(clf, model_file_name)
            return {'acc': acc, 'model_name': model_name}
        elif int(id) == 2:
            lin_model = LinearRegression()
            lin_model.fit(X_train, y_train)
            y_pred = lin_model.predict(X_test)
            mse = round(metrics.mean_squared_error(y_test, y_pred))
            model_file_name = '{}/{}_id2.pkl'.format(self.model_directory, model_name)
            joblib.dump(lin_model, model_file_name)
            return {'mse': mse, 'model_name': model_name}

    def predict(self, id, model_name, data):
        """Метод для прогнозирования"""
        if int(id) == 1:
            model_path = '{}/{}_id1.pkl'.format(self.model_directory, model_name)
            # если файл не существует, то выводится ошибка
            if os.path.exists(model_path):
                clf = joblib.load(model_path)
                y_pred = clf.predict(data)
                return {'prediction': list(map(int, y_pred))}
            else:
                api.abort(404, 'ml_model {} doesnt exist'.format(model_name))
        elif int(id) == 2:
            model_path = '{}/{}_id2.pkl'.format(self.model_directory, model_name)
            # если файл не существует, то выводится ошибка
            if os.path.exists(model_path):
                lin_model = joblib.load(model_path)
                y_pred = lin_model.predict(data)
                return {'prediction': list(map(float, y_pred))}
            else:
                api.abort(404, 'ml_model {} doesnt exist'.format(model_name))

    def delete(self, del_model_path):
        """Метод для удаления существующих моделей"""
        os.remove(del_model_path)


ml = MLModels()

@api.route('/api/ml_models_info')
class MLModelsInfo(Resource):
    def get(self):
        models = ml.ml_models
        return models

@api.route('/api/ml_models/train/<int:id>')
class ModelTrain(Resource):
    def post(self, id):
        data = pd.read_json(api.payload['data'])
        target = api.payload['target']
        args = api.payload['args']
        model_name = api.payload['model_name']
        train_info = ml.train_models(data, id, str(target), model_name, **dict(args))
        return train_info
    def delete(self, id):
        model_name = api.payload['model_name']
        model_del = '{}/{}_id{}.pkl'.format(ml.model_directory, model_name, id)
        ml.delete(model_del)

@api.route('/api/ml_models/test/<int:id>')
class ModelTest(Resource):
    def post(self, id):
        DF_test = pd.read_json(api.payload['data'])
        model_name = api.payload['model_name']
        DF_test = ml.data_preprocessing(DF_test)
        X_test = DF_test.to_numpy()
        pred = ml.predict(id, model_name, X_test)
        return pred


if __name__ == '__main__':
    app.run()

# DF_train = pd.read_csv('boston_train.csv')
# DF_test = pd.read_csv('boston_test.csv')
# a = MLModels()
# train_data = a.data_preprocessing(DF_test)
# X = train_data.to_numpy()
# p = {'max_d': 7}
# j = a.train_models(DF_train, 2, 'Survived', 'my_model', **p)
# print(j)
# pred = a.predict(2, 'my_model', X)
# print(pred)



