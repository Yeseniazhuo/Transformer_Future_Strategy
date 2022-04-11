import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
from datetime import datetime
import random

from models import Transformer
from create_data import create_input_data,StockDataset

def eval(net, path, test_loader):
    # evalute the model using MSE
    net.load_state_dict(torch.load(path))
    criterion = nn.MSELoss(reduction = 'sum') #Square error
    net.eval()

    total_loss = 0.0
    prices_pred = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            y_pred = net(x)
            total_loss += criterion(y_pred, y).item()
            if i % 1000 == 999:
                print("processing")
        mse = total_loss / len(test_loader.dataset)
    print("The MSE is ", mse)
    return mse

def train(net, N_EPOCHS, train_loader, LR, path):
    # initailize the network, optimizer and loss function
    optimizer = optim.Adam(net.parameters(), lr=LR)
    criterion = nn.MSELoss()
    #writer = SummaryWriter(log_dir='runs/trans')

    running_loss = 0
    for epoch in range(N_EPOCHS):
        for i, (x_batch, y_batch) in enumerate(train_loader, 0):
            # prediction
            x_batch = x_batch
            y_batch = y_batch
        
            y_pred = net(x_batch)
            loss = criterion(y_pred, y_batch)

            # update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #learning_rate = LR / (1 + ((i + 1) / 250))
            #optimizer = scheduler(optimizer, learning_rate)
            ''' To see the running loss during training
            running_loss += loss.item()
            if (i + 1) % 30 == 0:
                # ...log the running loss
                print("loss:", running_loss / 30, " batch:", (i + 1))
                running_loss = 0.0'''

    torch.save(net.state_dict(), path)
    #writer.close()

def train_eval(data):
    # fixed hyperparams
    BATCH_SIZE = 10
    Y_OUT = 1 #length of output
    NUM_WORKERS = 0
    LR = 0.001 #learning rate
    TRAIN_SIZE =  450
    # 450: 0.007309428891167045
    # 300: 0.017398580908775323
    # 600: 0.015148541885428126
    N_EPOCHS = 5
    N_IN = 25 #length of input
    K = 4 # lags 
    TEST_SIZE = 10

    # model parameters
    dim_input = 1 #number of features
    output_sequence_length = 1 #length of output
    dec_seq_len = 1 #length of output
    dim_val = 64
    dim_attn = 12#12
    n_heads = 8 
    n_encoder_layers = 4
    n_decoder_layers = 2

    MODEL_PATH = 'weights/{e}_{d}_{v}_{n}_{y}_{k}_seed{seed}'.format(e=n_encoder_layers, 
    d=n_decoder_layers, v=dim_val, n=N_IN, y=Y_OUT,k=K, seed=SEED)

    #init network
    net = Transformer(dim_val, dim_attn, dim_input, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers, n_heads)
    #net.load_state_dict(torch.load(MODEL_PATH))
    # load the dataset

    X_train, y_train, X_test, y_test , min_, max_= create_input_data(data, N_IN, Y_OUT,TRAIN_SIZE,TEST_SIZE,K)

    train_dataset = StockDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset,batch_size=BATCH_SIZE)
    test_dataset = StockDataset(X_test, y_test)
    test_loader = DataLoader(dataset=test_dataset,batch_size=BATCH_SIZE)

    train(net, N_EPOCHS, train_loader, LR, MODEL_PATH)
    return eval(net, MODEL_PATH, test_loader)


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(datetime.today().strftime('%Y-%m-%d'))
    print(f"Using {device}")
    # fix the random seed
    # 0 999 333 111 123
    SEED = 0
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # path of data
    PATHS = 'dataset/processed.csv'
    #load data
    df =  pd.read_csv(PATHS, index_col=0)
    
    for i in range(1):
        date_index=random.randint(0,len(df["DATE"]))
        data = df.interpolate().dropna(axis=1)[df['DATE'] == df['DATE'][date_index]]['LAST_PRICE']
        # !!!! If the consecutive null values is > 2, interpolate function will do nothing
        # we drop those tickers with consecutive null values greater than 2
        mse = train_eval(data) #此处，data应为一个Series，并且长度大于450+20+25+2+1
        # train_size + test+size + input_dimension + lags + output_dimension
    