from tkinter import N
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):
    """ return (X, y) 
        X : (n_lags, features(6))
        y: (n_lags, adj_close)
     """
    def __init__(self, X, y):
        super().__init__()
        self.X = torch.flatten(X, start_dim=1,end_dim=2).transpose(0, 1).transpose(1, 2)
        self.y = torch.flatten(y[0], end_dim=1)
        # print("Input data Size: ", self.X.size())
        # print("Label Size: ", self.y.size())

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        return (self.X[idx], self.y[idx])

    def __len__(self):
        return self.X.shape[0]


def create_input_data(data,n_lags,y_days,input_size,test_size,lags):
    '''
    path: dataset file path
    n_lags: length of input data
    y_days: length of output data
    '''
    X, y = [], []
    X_test, y_test = [[]],[[]]
    X_train, y_train = [[]],[[]]

    series = data.values.T
    for step in range(len(series) - n_lags - y_days - lags + 1):
        end_step = step + n_lags
        X.append(series[step:end_step])
        y.append(series[end_step+lags:end_step+y_days+lags])

    X_train_ = X[:input_size]
    y_train_ = y[:input_size]
    X_test_ = X[input_size:input_size+test_size]
    y_test_ = y[input_size:input_size+test_size]

    # normalize the time series with Min-Max
    max_ = max(np.amax(X_train_), np.amax(y_train_),np.amax(X_test_), np.amax(y_test_))
    min_ = min(np.amin(X_train_), np.amin(y_train_),np.amin(X_test_), np.amin(y_test_))
    X_train_ = (X_train_ - min_) / (max_ - min_)
    y_train_ = (y_train_ - min_) / (max_ - min_)
    X_test_ = (X_test_ - min_) / (max_ - min_)
    y_test_ = (y_test_ - min_) / (max_ - min_)

    i = 0 # only one dimension
    X_test[i].append(X_test_)
    y_test[i].append(y_test_)
    X_train[i].append(X_train_)
    y_train[i].append(y_train_)
    X =[]
    y =[]

    """ return a tensor with shape: X:(num_features, 1, num_samples, n_lags) 
                                    y:(num_features, 1, num_samples, y_days)"""
                                    
    return torch.FloatTensor(X_train), torch.FloatTensor(y_train),torch.FloatTensor(X_test), torch.FloatTensor(y_test),min_,max_


def create_prediction_input(data,n_lags,y_days,input_size,lags):
    '''
    path: dataset file path
    n_lags: length of input data
    y_days: length of output data
    '''
    X, y = [], []
    X_train, y_train = [[]],[[]]

    series = data.values.T
    for step in range(len(series) - n_lags - y_days - lags + 1 - input_size,len(series) - n_lags - y_days - lags + 1):
        end_step = step + n_lags
        X.append(series[step:end_step])
        y.append(series[end_step+lags:end_step+y_days+lags])
    
    X_train_ = X[:input_size]
    y_train_ = y[:input_size]

    # normalize the time series with Min-Max
    max_ = max(np.amax(X_train_), np.amax(y_train_))
    min_ = min(np.amin(X_train_), np.amin(y_train_))
    X_train_ = (X_train_ - min_) / (max_ - min_)
    y_train_ = (y_train_ - min_) / (max_ - min_)

    i = 0 # only one dimension
    X_train[i].append(X_train_)
    y_train[i].append(y_train_)
    X =[]
    y =[]

    """ return a tensor with shape: X:(num_features, 1, num_samples, n_lags) 
                                    y:(num_features, 1, num_samples, y_days)"""
                                    
    return torch.FloatTensor(np.array(X_train)), torch.FloatTensor(np.array(y_train)), min_, max_
