import pandas as pd
import numpy as np
import os
from typing import Union

def get_mode(x : pd.Series) -> Union[float,None]:
    '''mode 값 추출'''
    mode_val = x.mode(dropna=True)
    if len(mode_val)==0:
        return np.nan
    else:
        return mode_val[0]
    

def interpolate_by_time(df, col_name, time='insert_time') -> pd.DataFrame:
    '''시간에 따라 보간값 채워넣기'''
    df[time] = pd.to_datetime(df[time])
    df_index = df.index
    df_columns = df.columns
    df = df.set_index(time)
    if df[col_name].nunique() == 0:
        df[col_name] = np.nan
    else:
        df[col_name] = df[col_name].interpolate(method='time',limit_direction='both').round(-1)
    df = df.reset_index()
    df.index = df_index
    df = df[df_columns]
    return df

def fillna_by_order(df, col_name, time='insert_time') -> pd.DataFrame:
    '''시간에 따라 이전값으로 채우기(이전값이 없으면 이후값으로 채우기)'''
    df_index = df.index
    df_columns = df.columns
    df = df.set_index(time)

    if df[col_name].nunique() == 0:
        df[col_name] = np.nan
    else:
        df['temp_order'] = range(df.shape[0])
        df = df.sort_index()
        # 기본적으로 이전 정보로 채워넣되 이전 정보가 없다면 나중 정보로 채워넣기
        df[col_name] = df[col_name].fillna(method="ffill").fillna(method='bfill')
        df = df.sort_values(by = 'temp_order')
        df = df.drop('temp_order',axis=1)
    df = df.reset_index()
    df.index = df_index
    df = df[df_columns]
    return df

def fillna(df, col_name, method, time='insert_time') -> pd.DataFrame:
    null_user_id = df.loc[df[col_name].isnull(),'user_id']
    df_not_full = df[df['user_id'].isin(null_user_id)]
    if method=='order':
        fill_series = df_not_full.groupby('user_id').apply(lambda x:fillna_by_order(x,col_name))[col_name]
    elif method=='interpolate':
        fill_series = df_not_full.groupby('user_id').apply(lambda x:interpolate_by_time(x,col_name))[col_name]
    elif method=='mode':
        fill_series = df_not_full.groupby('user_id')[col_name].transform(get_mode)
    else:
        assert 1==2, 'check your method parameter'
    df.loc[fill_series.index.tolist(), col_name] = fill_series
    return df
