import pandas as pd
import numpy as np
import os
from itertools import product
from typing import Union, Dict




def aggregation(df, A_col, B_col,is_conti=True,
                statics_conti=\
                ['max','min','mean','std',
                 ('quantile_first',lambda x:x.quantile(0.25)),('quantile_second',lambda x:x.quantile(0.5)),('quantile_third',lambda x:x.quantile(0.75))]
                ) -> Dict[str,Dict[str,float]]:
    '''
    A변수의 범주별로 얻은 B의 대표값
    - B가 연속형일 때는 최대값, 최소값, 평균, 표준편차, 1분위수, 2분위수, 3분위수를 대표값으로 한다.
    - B가 범주형일 때는 각 범주별로 차지하는 비율을 대표값으로 한다.
    '''
    if is_conti:
        df_group = df.groupby(A_col)[B_col].agg(statics_conti)
    else:
        df_group = df.groupby(A_col)['purpose'].apply(lambda x:x.value_counts(normalize=True)).unstack()
    col_name = df_group.apply(dict, axis=1).index.name
    col_value = df_group.apply(dict, axis=1).to_dict()
    return col_value

def get_mapping_dict(df, A:list, B_conti:list, B_cat:list) -> Dict[tuple, Dict[str,Dict[str,float]]]:
    '''모든 쌍에 대해 A변수별 B의 대표값을 추출한다.'''
    is_conti_list = [True]*len(B_conti) + [False]*len(B_cat)
    mapping = {}
    for A_col, (B_col, is_conti) in product(A,zip(B_conti+B_cat, is_conti_list)):
        if A_col == B_col:
            continue
        col_value = aggregation(df, A_col,B_col,is_conti=is_conti)
        mapping[(A_col,B_col)] = col_value
    return mapping

def mapping(df,mapping,application_id,A,B):
    '''(미완)mapping dict를 이용해 대표값을 얻는다. 
    --> - 시간이 오래 걸리는 관계로 train data에 대해서는 지양
    --> - test data는 원칙적으로 새 feature를 train data를 통해 얻어야하므로 test data에 대해 이 함수를 쓴다.'''
    temp = df.query(f'application_id=={application_id}')
    temp_dict = mapping[(A,B)]
    AB_val = temp[A].values[0]
    prefix = f'{B}_by_{A}'
    if pd.isna(AB_val):
        temp_df = pd.Series([np.nan]*len(statics_conti),index = list(map(lambda x:x[0] if type(x)==type((1,2)) else x,statics_conti)))
    else:
        temp_df = pd.Series(temp_dict[AB_val])
    temp_df.index = prefix+'_'+temp_df.index
    return temp_df



transform_conti_pd = lambda x:f'{x.max()},{x.min()},{x.mean()},{x.std()},{x.quantile(0.25)},{x.quantile(0.5)},{x.quantile(0.75)}'
transform_conti_np = lambda x:f'{np.max(x)},{np.min(x)},{np.mean(x)},{np.std(x)},{np.quantile(x,0.25)},{np.quantile(x,0.5)},{np.quantile(x,0.75)}'

categories_conti = pd.Series(['max','min','mean','std','quantile_first','quantile_second','quantile_third'], dtype='str')
def table_transform_conti(df, A, B):
    '''
    - B변수가 연속형인 경우 사용
    - table 단위로 대표값 추출을 수행
    '''
    df[A] = df[A].fillna('-1')
    transformed = df.groupby(A)[B].transform(transform_conti_pd)
    result = transformed.str.split(pat=',',expand=True)
    result = result.astype(float)
    prefix = f'{B}_by_{A}'
    result.columns = (prefix +'_'+ categories_conti).tolist()
    return result


def value_counts(x,extra_indexes,fill_null,suffix=None,normalize=True):
    if type(x) != pd.Series:
        x = pd.Series(x)
    value_counted = x.value_counts(normalize=normalize)
    for extra_index in extra_indexes:
        if extra_index not in value_counted.index:
            value_counted[extra_index] = fill_null
    value_counted = value_counted.sort_index()
    if suffix != None:
        value_counted.index = value_counted.index +'_' + suffix
    return value_counted

transform_cat = lambda x:f'{list(value_counts(x,["-1"], np.nan, normalize=True).values)}'.replace('\n','')
def table_transform_cat(df, A, B):
    '''
    - B변수가 범주형일 경우 사용
    - table 단위로 대표값 추출을 수행
    '''
    df[A] = df[A].fillna('-1')
    df[B] = df[B].fillna('-1')
    categories = np.sort(df[B].value_counts().index)
    transformed = df.groupby(A)[B].transform(transform_cat)
    result = transformed.str[1:-1].str.split(',',expand=True).astype(float)
    prefix = f'{B}_by_{A}'
    result.columns = prefix + '_' +categories
    return result


def freq_count(df, 
               valid_actions= ['UseLoanManage','CompleteIDCertification','UsePrepayCalc','UseDSRCalc','GetCreditInfo','is_applied_Y']):
    '''
    log data의 대출 승인 이전 행동 비율 계산
    '''
    is_applied_Y_index = np.where(df['event']=='is_applied_Y')[0]
    if len(is_applied_Y_index)>=2:
        first = is_applied_Y_index[0]
        last = is_applied_Y_index[-1]
        first_action = value_counts(df['event'].values[:first],valid_actions,0, suffix='first')
        last_action = value_counts(df['event'].values[first:last],valid_actions,0,suffix='last')
    elif len(is_applied_Y_index)==1:
        first = is_applied_Y_index[0]
        first_action = value_counts(df['event'].values[:first],valid_actions,0,suffix='first')
        last_action = pd.Series([0]*len(valid_actions), index=valid_actions).sort_index()
        last_action.index = last_action.index +'_last'
    else:
        first_action = pd.Series([0]*len(valid_actions), index=valid_actions).sort_index()
        last_action = pd.Series([0]*len(valid_actions), index=valid_actions).sort_index()
        first_action.index = first_action.index +'_first'
        last_action.index = last_action.index +'_last'
    actions_freq = first_action.append(last_action)
    return actions_freq

def timedelta_hour(x,y):
    x_time = pd.to_datetime(x)
    y_time = pd.to_datetime(y)
    if x_time > y_time:
        components = (x_time - y_time).components
    else:
        components = (y_time - x_time).components
    delta = components.days*24 + components.hours + components.minutes/60 + components.seconds/3600
    return delta

def timedelta_hour_dataframe(df,i,j,col='timestamp'):
    if i!=j:
        x = df.iloc[i][col]
        y = df.iloc[j][col]
        delta = timedelta_hour(x,y)
    elif i==j:
        delta = np.nan
    return delta

def timedelta_static(df : pd.DataFrame) -> str:
    '''
    대출 신청을 한 시간 간격(정확히는 대출 신청 이후 다음 행동 순간부터)의 대표값 추출
    '''
    control = False
    while df.iloc[0]['event']=='is_applied_Y':
        if df.shape[0]==1:
            control = True
            break
        else:
            df = df.iloc[1:]
    if control:
        return f'{np.nan},{np.nan},{np.nan},{np.nan},{np.nan},{np.nan},{np.nan}'
    else:
        idx = np.where(df['event'] == 'is_applied_Y')[0]
        if len(idx)>=2:
            idx_shift = [0] + list(idx+1)[:-1]
            time_delta_array = np.array([timedelta_hour_dataframe(df,idx1,idx2) for idx1, idx2 in zip(idx_shift, idx) if idx1!=idx2])
            return transform_conti_np(time_delta_array)
        elif len(idx)==1:
            time_delta = timedelta_hour_dataframe(df,0,idx[0])
            return f'{time_delta},{time_delta},{time_delta},{time_delta},{time_delta},{time_delta},{time_delta}'        
        else:
            return f'{np.nan},{np.nan},{np.nan},{np.nan},{np.nan},{np.nan},{np.nan}'

def timedelta_static_wrapper(se : pd.Series) -> str:
    '''
    groupby 적용을 위해 구조 수정
    '''
    df = se.str.split('/',expand=True)
    df.columns = ['event','timestamp']
    return timedelta_static(df)