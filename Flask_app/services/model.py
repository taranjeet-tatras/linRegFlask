"""This file will contain the main class used to train models"""
import numpy as np
import statistics
import config
from sklearn.linear_model import LinearRegression
import pandas as pd



def impute(column_names, data):
    """filling missing data with median values"""
    for column_name in column_names:
        column_data = data.loc[:, column_name]
        column_data = column_data.fillna(statistics.median(column_data.values)).values
        data[column_name] = column_data
    return data


def create_test_train(file_path,relevant_columns):
    """preprocessing the data"""
    data = pd.read_csv(file_path)
    partial_data = data[relevant_columns].copy()
    imputed_data = impute(column_names = relevant_columns , data=partial_data)
    labels = data['SalePrice']
    labels_without_nan = labels.fillna('unknown')
    train_y = labels_without_nan[:1460]
    train_y = (np.asarray(train_y)).reshape(1460, 1)
    train_x_pd1_arr = np.asarray(imputed_data)
    train_X = train_x_pd1_arr
    train_X = train_x_pd1_arr[:1460].reshape(1460, 10)
    return train_X,train_y


def train_model(train_X, train_y):
    model = LinearRegression()
    model.fit(train_X, train_y)
    return model

#doing the same thing in oops


class MODEL:
    def __init__(self, relevant_columns, file_path):
        self.relevant_columns = relevant_columns
        self.file_path = file_path
        self.train_X = None
        self.train_y = None

    def impute(self,data):
        """filling missing data with median values"""
        for column_name in self.relevant_columns:
            column_data = data.loc[:, column_name]
            column_data = column_data.fillna(statistics.median(column_data.values)).values
            data[column_name] = column_data
        return data

    def create_test_train(self):
        """preprocessing the data"""
        data = pd.read_csv(self.file_path)
        partial_data = data[self.relevant_columns].copy()
        imputed_data = impute(column_names=self.relevant_columns, data=partial_data)
        labels = data['SalePrice']
        labels_without_nan = labels.fillna('unknown')
        train_y = labels_without_nan[:1460]
        train_y = (np.asarray(train_y)).reshape(1460, 1)
        train_x_pd1_arr = np.asarray(imputed_data)
        train_X = train_x_pd1_arr
        train_X = train_x_pd1_arr[:1460].reshape(1460, 10)
        self.train_X = train_X
        self.train_y = train_y
        return train_X, train_y

    def train_model(self):
        model = LinearRegression()
        model.fit(self.train_X, self.train_y)
        return model






