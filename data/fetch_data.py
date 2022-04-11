import boto3
from boto3.session import Session
import os
import pandas as pd

ACCESS_KEY = 'AKIAR5ZTUIZKG7UGB3RF'
SECRET_KEY = '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'
REGION_NAME = 'ap-southeast-1'

session = boto3.Session(aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=REGION_NAME)
s3 = session.resource('s3')

data_bucket = s3.Bucket('apix-model-fintech-1-15-12-2021')
for s3_file in data_bucket.objects.all():
    path, filename = os.path.split(s3_file.key)
    if 'TAIEX_20220105/TAIEX_3836295534' in s3_file.key and filename!='':
        print(filename)
        # path=path.split('/')[1]
        # try:
        #     os.makedirs('data/'+path)
        # except:
        #     pass
        data_bucket.download_file(s3_file.key, 'data/'+filename)


data_path='data/'
list_of_files=[data_path+x for x in os.listdir(data_path)]
list_of_files.sort(key=lambda x: int(x[10:-11]))

cols=['TICKER','ISO_CODE','DATE','TIME','MSG_TYPE','LAST_PRICE','LAST_EXCH','LAST_DATE','LAST_TIME','SECURITY_TYPE','ORDER_NUM']
df=pd.read_csv(list_of_files[0], compression='gzip', error_bad_lines=False)
df[cols].to_csv('data/aggregation.csv',header=True,index=False)

for file in list_of_files[1:]:
    df=pd.read_csv(file, compression='gzip', error_bad_lines=False)
    df[cols].to_csv('data/aggregation.csv',mode='a',header=False,index=False)
