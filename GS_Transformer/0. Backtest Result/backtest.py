#coding=utf-8

import pandas as pd
import numpy as np
from prediction import predict, train_model, model_predict


data = pd.read_csv('data/processed.csv')

K = 450
N = 25
M = 4
L = M-2
b_sig = 5e-4
s_sig = -5e-4


def backtest(temp_data, buy_range=b_sig, short_range=s_sig, k=K, m=M, n=N, l=L):
    """
    temp_data: data for specific trading day
    buy_range: min return when initiating a buy action
    short_range: max return when initiating a sell action

    when training a model at time t
    k：# of data used to train a model，i.e. t-k ~ t-1 are used to train
    m：price at time t + m is predicted
    n：# of data used as input for the model to predict
    l：the model will be used to predict from time t+l+1, due to the model training time, i.e. time t ~ t+l-1 are used
       to train the model, the model will start function from t+l+1
    """
    wealth = 0  # accumulated wealth
    ordercount = 0  # number of order during the day
    Position = 0  # position, 0: no position, 1: long position, 2: short position
    stats = pd.DataFrame(columns=['action', 'time', 'wealth', 'Position'])  # a dataframe used to store info

    priceList = temp_data['LAST_PRICE'].values.tolist()  # price list on that day

    # start to train the model from time n+m+k -> to generate k complete entries for training
    train_data = temp_data.loc[:n+m+k-1, 'LAST_PRICE']  # N+M+K entries
    model, max_, min_ = train_model(train_data, M)  # get the model
    train_time = [120*_ for _ in range(6, 27)]  # time to train a new model; train a new model every 10 mins

    # get info from first n+m+k time
    return_mean = np.mean(
        (np.array(priceList[2:n + m + k]) - np.array(priceList[:n + m + k - 2])) / np.array(priceList[:n + m + k - 2]))
    return_std = np.std(
        (np.array(priceList[2:n + m + k]) - np.array(priceList[:n + m + k - 2])) / np.array(priceList[:n + m + k - 2]))
    adj_buy_range = max(return_mean + 1.5 * return_std, buy_range)  # adjusted min return to initiate a buy action
    adj_short_range = min(return_mean - 1.5 * return_std, short_range)  # adjusted max return to initiate a short action

    # traverse time n+m+k+1 to the end
    for i in range(n + m + k + 1, len(temp_data) - m):

        if i in train_time:  # update the model
            train_data = temp_data.loc[i-(n + m + k):i-1, 'LAST_PRICE']  # N+M+K entries
            model, max_, min_ = train_model(train_data, M)

        predict_data = temp_data.loc[i - n:i - 1, 'LAST_PRICE']  # N entries, as the input for the model to predict
        predictedPsl = model_predict(model, predict_data, max_, min_)  # get the predicted price at time i+m

        # model finished training at i+l, calculate the return from time i+l to i+m
        predictedReturn = (predictedPsl - priceList[i + l]) / priceList[i + l]

        if ordercount == 20:  # if already 20 transactions
            return stats
        elif ordercount <= 19:
            if Position == 0:  # no position
                if predictedReturn >= adj_buy_range:  # initiate a long position
                    wealth -= priceList[i + l + 1]  # make the decision at time i+l, complete the transaction at i+l+1
                    Position += 1  # position from 0 to 1
                    ordercount += 1
                    print(temp_data.loc[i, 'DATETIME'], 'wealth: ', wealth, 'Position: ', Position, 'ordercount: ',
                          ordercount, 'buy')

                    stats.loc[ordercount, 'action'] = 'buy'
                    stats.loc[ordercount, 'time'] = temp_data.loc[i, 'DATETIME']
                    stats.loc[ordercount, 'wealth'] = wealth
                    stats.loc[ordercount, 'Position'] = Position
                elif predictedReturn <= adj_short_range:  # initiate a short position
                    wealth += priceList[i + l + 1]
                    Position -= 1
                    ordercount += 1
                    print(temp_data.loc[i, 'DATETIME'], 'wealth: ', wealth, 'Position: ', Position, 'ordercount: ',
                          ordercount, 'sell')

                    stats.loc[ordercount, 'action'] = 'sell'
                    stats.loc[ordercount, 'time'] = temp_data.loc[i, 'DATETIME']
                    stats.loc[ordercount, 'wealth'] = wealth
                    stats.loc[ordercount, 'Position'] = Position
            elif Position == 1:  # when holding a long position
                if predictedReturn <= short_range * 0.6:  # if return worsens, close the position
                    wealth += priceList[i + l + 1]
                    Position -= 1
                    ordercount += 1
                    print(temp_data.loc[i, 'DATETIME'], 'wealth: ', wealth, 'Position: ', Position, 'ordercount: ',
                          ordercount, 'sell')

                    stats.loc[ordercount, 'action'] = 'sell'
                    stats.loc[ordercount, 'time'] = temp_data.loc[i, 'DATETIME']
                    stats.loc[ordercount, 'wealth'] = wealth
                    stats.loc[ordercount, 'Position'] = Position
            elif Position == -1:  # when holding a short position
                if predictedReturn >= buy_range * 0.6:  # if return worsens, close the position
                    wealth -= priceList[i + l + 1]
                    Position += 1
                    ordercount += 1
                    print(temp_data.loc[i, 'DATETIME'], 'wealth: ', wealth, 'Position: ', Position, 'ordercount: ',
                          ordercount, 'buy')

                    stats.loc[ordercount, 'action'] = 'buy'
                    stats.loc[ordercount, 'time'] = temp_data.loc[i, 'DATETIME']
                    stats.loc[ordercount, 'wealth'] = wealth
                    stats.loc[ordercount, 'Position'] = Position
        else:
            break

    # if the market comes to close, close the current position at the end
    if Position == 1:
        wealth += priceList[-1]
        Position -= 1
        ordercount += 1
        print(temp_data.loc[i, 'DATETIME'], 'wealth: ', wealth, 'Position: ', Position, 'ordercount: ',
              ordercount, 'sell close')

        stats.loc[ordercount, 'action'] = 'sell close'
        stats.loc[ordercount, 'time'] = temp_data.loc[i, 'DATETIME']
        stats.loc[ordercount, 'wealth'] = wealth
        stats.loc[ordercount, 'Position'] = Position
    elif Position == -1:
        wealth -= priceList[-1]
        Position += 1
        ordercount += 1
        print(temp_data.loc[i, 'DATETIME'], 'wealth: ', wealth, 'Position: ', Position, 'ordercount: ',
              ordercount, 'buy close')

        stats.loc[ordercount, 'action'] = 'buy close'
        stats.loc[ordercount, 'time'] = temp_data.loc[i, 'DATETIME']
        stats.loc[ordercount, 'wealth'] = wealth
        stats.loc[ordercount, 'Position'] = Position

    return stats


dates = list(data['DATE'].unique())
dates.reverse()


for date in dates[:]:
    temp_data = data[data['DATE'] == date].reset_index()
    stat = backtest(temp_data)
    file = 'GS_Transformer/results_csv/' + str(date) + '.csv'
    stat.to_csv(file)
