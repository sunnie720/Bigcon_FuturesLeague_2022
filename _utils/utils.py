import pandas as pd
import numpy as np
import os

def split_comma(x):
    '''어떤 큰 정수에 대해 손쉽게 읽기 위해 1,000 단위로 구분지어 출력'''
    x_str = str(int(x))
    first_comma = len(x_str)%3
    if first_comma==0:
        comma_str = ''
    else:
        comma_str = x_str[:first_comma] + ','
    remain_str = x_str[first_comma:]
    while len(remain_str)>=3:
        comma_str += remain_str[:3]+','
        remain_str = remain_str[3:]
    return comma_str[:-1]

def df_length(df):
    '''DataFrame의 행 수를 1,000 단위로 구분지어 출력'''
    return split_comma(df.shape[0])

def is_unique(df, col, standard='user_id'):
    '''어떤 DataFrame(df)의 열(col)에 대해 user_id 별로 몇개의 값을 갖고 있는지 체크하기 위함'''
    # nan 값 역시 하나의 원소로 취급
    return df.groupby(standard)[col].agg(lambda x:x.nunique(dropna=False))

def split_train_test(df, time):
    '''6월 데이터를 test set으로, 6월 데이터가 아닌것(3,4,5월) 데이터는 train set으로 분리하기 위함'''
    test_cond = pd.to_datetime(df[time]).dt.month==6
    df_train = df[~test_cond]
    df_test = df[test_cond]
    return df_train, df_test

def make_date_format(x : str) -> str:
    '''날짜형 포맷 통일 : 202205 -> 20220501(문자형)'''
    if np.isnan(x)==False:
        x_str = str(int(x))
        if len(x_str) == 6:
            x_str += '01'
        return x_str
    else:
        return np.nan
    
def as_float(x, as_int=False):
    '''숫자로 바꿀 수 있으면 숫자로, 아니면 null값으로 변형
        null이 포함된 dataframe이나 series에 .astype('float')를 적용하기 위함'''
    try:
        if as_int==True:
            return int(x)
        else:
            return float(x)
    except:
        return np.nan
    
def save_pickle(obj, path):
    '''큰 데이터를 압축하여 저장'''
    import gzip
    import pickle
    with gzip.open(path,'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
    print(f'The Object saved in {path}.')

def load_pickle(path):
    '''gzip으로 압축된 데이터를 불러옴'''
    import gzip
    import pickle
    with gzip.open(path, 'rb') as f:
         obj = pickle.load(f)
    return obj
            