'''
Created on Apr 9, 2019

@author: malte
'''
import numpy as np
import pandas as pd
from numpy import dtype
import time
from math import ceil
import gc
from inspect import signature

def expand(df, columns=None, contains=None):
    
    tstart = time.time()
    
    if columns is None:
        if contains is None:
            groups = df.columns.to_series().groupby( df.dtypes ).groups
            columns = groups[dtype('object')]
        else:
            columns = list( filter( lambda x: contains in x, df.columns.values ) )
    
#     for key in columns:
#         print( key )
#         print( df[df[key].isnull()] )
#         print( df[key] )
#         print( len( np.concatenate(df[key].values ) ) )
    df = reduce_mem_usage(df)
    df = pd.DataFrame({
        col:np.repeat(df[col].values, df[columns[0]].str.len())
        for col in df.columns.difference(columns)
    }).assign(**{col:np.concatenate(df[col].values) for col in columns})[df.columns.tolist()]
    df = reduce_mem_usage(df)
    
    print( 'expand in {}'.format( (time.time() - tstart) ) )
    
    return df

def check_cols( df ):
    
    for c in df.columns: 
        if sum( np.isinf( df[c] )) > 0:
            print('inf ',c)
        if sum( df[c].isnull() ) > 0:
            print('na ',c)

def copy_features( examples, features ):
    
    features = features[features.session_id.isin( examples.session_id.unique() )]
    features.reset_index(drop=True, inplace=True)
            
    add_cols = np.setdiff1d(features.columns, examples.columns)
    for col in add_cols:
        examples[col] = features[col]
        del features[col]
        gc.collect()
        
    examples['impressions_test'] = features['impressions']
    test = examples[examples.impressions != examples.impressions_test]
    
    if len(test) > 0:
        print( examples[['impressions','impressions_test']] )
        print( test )
        print( np.setdiff1d(examples.session_id.unique(), features.session_id.unique()) )
        print( 'features not in line' )
        exit()
    del examples['impressions_test']
    
    return examples
    
def copy_features2( examples, features ):
    
    features = features[features.session_id.isin( examples.session_id.unique() )]
    #features.reset_index(drop=True, inplace=True)
            
    add_cols = list(np.setdiff1d(features.columns, examples.columns))
    before = len(examples)

    examples = examples.merge( features[['session_id','impressions'] + add_cols], on=['session_id','impressions'], how='inner' )

    del features
    gc.collect()

    if len(examples) != before:
        print( np.setdiff1d(examples.session_id.unique(), features.session_id.unique()) )
        print( 'features not in line' )
        exit()
    
    return examples

def reduce_series( series, fillna=False ):
    
    IsInt = False
    mx = series.max()
    mn = series.min()
    
    series = series.replace([np.inf, -np.inf], np.nan)  
    
    # Integer does not support NA, therefore, NA needs to be filled
    if not np.isfinite(series).all() and fillna: 
        series.fillna(mn-1,inplace=True)  
           
    # test if column can be converted to an integer
    if series.isnull().sum() == 0:
        
        asint = series.fillna(0).astype(np.int64)
        result = (series - asint)
        result = result.sum()
        if result > -0.01 and result < 0.01:
            IsInt = True
    
    if not np.isfinite(series).all(): 
        IsInt = False
    
    # Make Integer/unsigned Integer datatypes
    if IsInt:
        if mn >= 0:
            if mx < 255:
                return series.astype(np.uint8)
            elif mx < 65535:
                return series.astype(np.uint16)
            elif mx < 4294967295:
                return series.astype(np.uint32)
            else:
                series.astype(np.uint64)
        else:
            if mn > np.iinfo(np.int8).min and mx < np.iinfo(np.int8).max:
                return series.astype(np.int8)
            elif mn > np.iinfo(np.int16).min and mx < np.iinfo(np.int16).max:
                return series.astype(np.int16)
            elif mn > np.iinfo(np.int32).min and mx < np.iinfo(np.int32).max:
                return series.astype(np.int32)
            elif mn > np.iinfo(np.int64).min and mx < np.iinfo(np.int64).max:
                return series.astype(np.int64)    
    
    # Make float datatypes 32 bit
    else:
        return series.astype(np.float32)
    
def reduce_mem_usage(props, cols=None):
    tstart = time.time()
    #start_mem_usg = props.memory_usage().sum() / 1024**2 
    #print("Memory usage of properties dataframe is :",start_mem_usg," MB")
    if cols is None:
        cols = props.columns
    for col in cols:
        if props[col].dtype != object:  # Exclude strings
            
            # Print current column type
#             print("******************************")
#             print("Column: ",col)
#             print("dtype before: ",props[col].dtype)
            
            props[col] = reduce_series(props[col])
            
            # Print new column type
#             print("dtype after: ",props[col].dtype)
#             print("******************************")
    
    # Print final result
#     print("___MEMORY USAGE AFTER COMPLETION:___")
    #mem_usg = props.memory_usage().sum() / 1024**2 
#     print("Memory usage is: ",mem_usg," MB")
    #print("This is ",100*mem_usg/start_mem_usg,"% of the initial size")
    print( 'reduced mem usage in {}'.format( (time.time() - tstart) ) )
    return props


def humansize(nbytes):
    suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    i = 0
    while nbytes >= 1024 and i < len(suffixes)-1:
        nbytes /= 1024.
        i += 1
    f = ('%.3f' % nbytes).rstrip('0').rstrip('.')
    return '%s %s' % (f, suffixes[i])

def train_test_split( X, y, test_size=0.2, shuffle=False, keep='session_id' ): 
    
    sessions = X.session_id.unique()
    if shuffle:
        msk = np.random.rand( len(sessions) ) > test_size
    else:
        msk = np.arange( len(sessions) ) < ceil( len(sessions) * ( 1 - test_size ) )
    
    train = sessions[msk]
    
    X_train = X[X.session_id.isin(train)]
    X_valid = X[~X.session_id.isin(train)]
    
    y_train = y.ix[X_train.index.values]
    y_valid = y.ix[X_valid.index.values]
    
    return X_train, X_valid, y_train, y_valid

def train_test_split_idx( X, y, test_size=0.2, shuffle=False, keep='session_id' ): 
    
    sessions = X.session_id.unique()
    if shuffle:
        msk = np.random.rand( len(sessions) ) > test_size
    else:
        msk = np.arange( len(sessions) ) < ceil( len(sessions) * ( 1 - test_size ) )
    
    train = sessions[msk]
    
    X_train = X[X.session_id.isin(train)].index.values
    X_valid = X[~X.session_id.isin(train)].index.values
    
    return X_train, X_valid

def train_test_cv( X, y, splits=5, shuffle=False, keep='session_id' ): 
    
    split_idxs = []
    
    sessions = X.session_id.unique()
    if shuffle: 
        np.random.shuffle(sessions)
        
    size = ceil( len(sessions) / splits )
    tmp = np.arange( len(sessions) )
    
    for i in range(splits):
        min_sess = i * size
        max_sess = (i+1) * size
        msk = ( tmp >= min_sess ) & ( tmp < max_sess )
        
        test_session = sessions[msk]
        split_idxs.append( (X[~X.session_id.isin(test_session)].index.values, X[X.session_id.isin(test_session)].index.values) )
    
    return split_idxs

def apply( df, cols, f, verbose=0 ):
    
    tstart = time.time()
    
    vals = df[cols].values
    res = []
    save = {}
    
    for i in range(len(df)):
        row = vals[i]
        sig = signature( f )
        if len(sig.parameters) > 1:
            res.append( f(row, save=save) )
        else:
            res.append( f(row) )
        if verbose > 0 and i % verbose == 1:
            gone = time.time() - tstart
            eta = ( len(df) - i ) * ( gone / i )
            print( 'processed {} of {} in {}: eta {}'.format( i, len(df), gone, eta ) )

    return res