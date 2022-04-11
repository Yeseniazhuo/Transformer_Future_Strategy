import numpy as np
import pandas as pd
import torch
from datetime import datetime

from prediction import predict

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(datetime.today().strftime('%Y-%m-%d'))
    print(f"Using {device}")

    PATHS = 'dataset/processed.csv'
    df =  pd.read_csv(PATHS, index_col=0)
    df = df.interpolate().dropna(axis=1)
    data = df[df['DATE'] == 20180209]['LAST_PRICE']

    result = predict(data[0:478],data[478-25:478])
    print("prediction: ",result) 
    print("actual: ",data[480])
    print("last price: ",data[477])