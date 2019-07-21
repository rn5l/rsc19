'''
Created on Mar 20, 2019

@author: malte
'''
from datetime import datetime, timedelta

from config.globals import BASE_PATH
from domain.action_types import CLICK
from helper.df_ops import expand, reduce_mem_usage
from helper.loader import write_hdfs, load_hdfs
import numpy as np
import pandas as pd
from random import shuffle


FROM = BASE_PATH + 'preprocessed/'

TARGET = BASE_PATH + 'competition/'
#TARGET = BASE_PATH + 'test/'

TYPE = 'competition'
#TYPE = 'sample'

DAYS_TRAIN_COMPETITION = None

DAYS_TRAIN = 0.05
DAYS_TEST = 0.01

RANDOM_TRAIN = False

def main():
    #data = tf.load_feather('all3')
    data = load_hdfs( FROM + 'joined_final.hd5' )
    
    method = globals()['split_'+TYPE]
    method( data )

def split_competition( data ):
    
    if DAYS_TRAIN_COMPETITION is not None:
        maxtr = datetime.fromtimestamp( data[data.train==1].timestamp.max() )
        mintr = datetime.fromtimestamp( data[data.train==1].timestamp.min() )
        start = maxtr - timedelta( days=DAYS_TRAIN_COMPETITION )
         
        data['mintimestamp'] = data.groupby('user_id').timestamp.transform( min )
        
        if RANDOM_TRAIN:
            keep = ((data['mintimestamp'] >= start.timestamp()) & (data.train == 1))
            num_sess = data[keep].session_id.nunique()
            
            sess = list(data[data.train==1].session_id.unique())
            shuffle( sess )
            keep = sess[:num_sess]
            
            keep = (data.train == 0) | data.session_id.isin(keep)
            data = data[keep]
        else: 
            keep = ((data['mintimestamp'] >= start.timestamp()) & (data.train == 1))
            keep = keep | (data.train == 0)
            data = data[keep]
        
        mintr = data[data.train == 1].timestamp.min()
        minva = data[data.train == 0].timestamp.min()
        maxva = data[data.train == 0].timestamp.max()
        
        print( datetime.fromtimestamp( mintr ) )
        print( datetime.fromtimestamp( minva ) )
        print( datetime.fromtimestamp( maxva ) )
    
        del data['mintimestamp']
    data['hidden'] = 0
    
    hide_test = data.reference.isnull() & (data.action_type == CLICK)
    data.ix[ hide_test, 'reference' ] = np.nan
    data.ix[ hide_test, 'hidden' ] = 1
    
    hide_train = data[ (data.train == 1 ) & (data.action_type == CLICK) ].copy() # filter clickout
    hide_train = hide_train.drop_duplicates( 'user_id', keep='last' )
    data.ix[ hide_train.index, 'reference' ] = np.nan
    data.ix[ hide_train.index, 'hidden' ] = 1
    
    tmp = pd.DataFrame()
    tmp['maxstamp'] = data[data.hidden == 1].groupby('session_id').timestamp.max()
    data = data.merge( tmp, right_index=True, left_on='session_id', how='left' )
    data['maxstamp'] = data['maxstamp'].fillna(data.timestamp.max())
    
    data['exclude'] = 0
    data.ix[data.timestamp > data.maxstamp, 'exclude'] = 1
    del data['maxstamp'], tmp
    
    data = reduce_mem_usage(data)
    write_hdfs( data, TARGET + 'data_log.hd5' )
    #data.to_csv( TARGET + 'data_log.csv', index=False )
    data[data.train == 0].to_csv( TARGET + 'data_log_test.csv' )
    
    examples = expand_and_label(data)
    
    write_hdfs( examples, TARGET + 'data_examples.hd5' )
    #examples.to_csv( TARGET + 'data_examples.csv', index=False )

def split_sample( data ):
    
    data = data[ data.train == 1 ].copy()
    data = reduce_mem_usage(data)
    
    maxtr = datetime.fromtimestamp( data.timestamp.max() )
    mintr = datetime.fromtimestamp( data.timestamp.min() )
    minva = maxtr - timedelta( days=DAYS_TEST )
    
    if DAYS_TRAIN is not None: 
        mintr = maxtr - timedelta( days=DAYS_TEST+DAYS_TRAIN )
    
    print(mintr)
    print(maxtr)
    print(minva)
     
    data['mintimestamp'] = data.groupby('user_id').timestamp.transform( min )
    
    data['train'] = (data['mintimestamp'] >= mintr.timestamp()).astype(int)
    if RANDOM_TRAIN:
        data.ix[data['mintimestamp'] >= minva.timestamp(),'train'] = 0
        num_sess = data[data.train==1].session_id.nunique()
        
        data['train'] = 1
        data.ix[data['mintimestamp'] >= minva.timestamp(),'train'] = 0
        
        sess = list(data[data.train==1].session_id.unique())
        shuffle( sess )
        keep = sess[:num_sess]
        
        keep = (data.train == 0) | data.session_id.isin(keep)
        data = data[keep]
    else: 
        data = data[data.train == 1]
        data.loc[data['mintimestamp'] >= minva.timestamp(),'train'] = 0

    print( data[['session_id','timestamp','mintimestamp','train']] )
    
    mintr = data[data.train == 1].timestamp.min()
    minva = data[data.train == 0].timestamp.min()
    maxva = data[data.train == 0].timestamp.max()
    
    print( datetime.fromtimestamp( mintr ) )
    print( datetime.fromtimestamp( minva ) )
    print( datetime.fromtimestamp( maxva ) )
     
    print(len(data[data.train == 1]))
    print(len(data[data.train == 0]))
    
    data = data.reset_index(drop=True)
    del data['mintimestamp']
    
    #print( len( set(test.session_id.unique()) & set(train.session_id.unique()) ) )
    
    data['hidden'] = 0
    data['exclude'] = 0
    
    examples_log = data[data.action_type == CLICK ].copy() # filter clickout
    examples_log = examples_log.drop_duplicates( 'user_id', keep='last' )
    truth = examples_log[examples_log.train == 0]
    
    #hide all
    data.loc[ examples_log.index.values, 'reference' ] = np.nan
    data.loc[ examples_log.index.values, 'hidden' ] = 1
    
    print( 'hidden test sum ', data[(data.hidden == 1) & (data.train == 0)].hidden.sum() )
    
    tmp = pd.DataFrame()
    tmp['maxstamp'] = data[data.hidden == 1].groupby('session_id').timestamp.max()
    data = data.merge( tmp, right_index=True, left_on='session_id', how='left' )
    data['maxstamp'] = data['maxstamp'].fillna(data.timestamp.max())
    
    data.loc[data.timestamp > data.maxstamp, 'exclude'] = 1
    del data['maxstamp'], tmp
    
    print( 'hidden test sum ', data[(data.hidden == 1) & (data.train == 0)].hidden.sum() )
    
    examples = expand_and_label(data)
    
    #hide test completely
    data.loc[ examples_log[examples_log.train == 0].index.values, 'item_id' ] = np.nan
    data.loc[ examples_log[examples_log.train == 0].index.values, 'price_session' ] = np.nan
    
    data = reduce_mem_usage(data)
    write_hdfs(data, TARGET + 'data_log.hd5')
    #data.to_csv( TARGET + 'data_log.csv' )
    data[data.train == 0].to_csv( TARGET + 'data_log_test.csv' )
        
    write_hdfs( examples, TARGET + 'data_examples.hd5' )
    #examples.to_csv( TARGET + 'data_examples.csv', index=False )
    
    truth.to_csv( TARGET + 'truth.csv', index=False )
    
    with open( TARGET + 'size.txt', 'w' ) as out:
        out.write( 'train_size: {}, test_size: {}'.format( DAYS_TRAIN, DAYS_TEST ) )

def expand_and_label( data ):
    
    examples = data[ data.hidden == 1 ].copy()
    examples['position'] = examples.impressions.apply( lambda x: list(range(1,len(x)+1)) )
    examples['num_impressions'] = examples.impressions.apply( lambda x: len(x) )
    examples = expand( examples, columns=['impressions','prices','position'] )
    examples['label'] = (examples['impressions'] == examples['item_id']) * 1
    examples = examples.sort_values( ['session_id','impressions'] ).reset_index(drop=True)
    
    del examples['reference'], examples['item_id'], examples['hidden'], examples['exclude'], examples['current_filters']
    
    return examples

if __name__ == '__main__':
    main()
    