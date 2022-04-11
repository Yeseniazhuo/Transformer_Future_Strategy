import pandas as pd
import os
import numpy as np
from datetime import datetime


df=pd.read_csv('data/aggregation.csv')
df.sort_values(['DATE','TIME'],inplace=True)
df.reset_index(drop=True,inplace=True)
df=df[['DATE','TIME','LAST_PRICE']]
df.drop_duplicates(inplace=True)
df['DATETIME']=pd.to_datetime(df['DATE'].astype('str')+df['TIME'].astype('str'),format='%Y%m%d%H%M%S%f')
df['TIMEDELTA']=df.groupby('DATE')['DATETIME'].shift(0)-df.groupby('DATE')['DATETIME'].shift(1)
# df['TIMEDELTA'].value_counts()
df=df.set_index('DATETIME')
# df.head()
df.to_csv('data/processed.csv')