'''
Created on May 23, 2019

@author: malte
'''
import pandas as pd
import time
from pathlib import Path
from domain.action_types import IMAGE, RATING, INFO, DEALS
import os

def load_csv( path ):
    tstart = time.time()
    data = pd.read_csv( path )
    print( 'load data in {}'.format( (time.time()-tstart) ) )
    return data

def load_feather( path ):
    tstart = time.time()
    data = pd.read_feather( path )
    print( 'load data in {}'.format( (time.time()-tstart) ) )
    return data

def load_hdfs( path ):
    tstart = time.time()
    data = pd.read_hdf( path, key='table' )
    print( 'load data in {}'.format( (time.time()-tstart) ) )
    return data

def write_feather( data, path ):
    tstart = time.time()
    ensure_dir( path )
    data.to_feather( path )
    print( 'write data in {}'.format( (time.time()-tstart) ) )
    return data

def write_hdfs( data, path ):
    tstart = time.time()
    ensure_dir( path )
    data.to_hdf( path, key='table', mode='w' )
    print( 'write data in {}'.format( (time.time()-tstart) ) )
    return data

def load_truth( path ): 
    tstart = time.time()
    
    truth_path = path + '/' + 'truth.csv' 
    truth = pd.read_csv( Path( truth_path ) )
    
    truth_path = path + '/' + 'data_log.hd5' 
    test = pd.read_hdf( Path( truth_path ), key='table' )
    
    sess_with_item = test[ (test.train==0) & ~test.reference.isnull() & test.action_type.isin([IMAGE,RATING,INFO,DEALS]) ].session_id.unique()
    truth['with_item'] = truth.session_id.isin( sess_with_item )
    
    print( 'load data in {}'.format( (time.time()-tstart) ) )

    return truth

def load_maps( folder ): 
    
    user_map = pd.read_csv( folder + 'user_map.csv', index_col=1, header=None, names=['user_id_org', 'user_id'], dtype={0:str,1:int} )
    session_map = pd.read_csv( folder + 'session_map.csv', index_col=1, header=None, names=['session_id_org', 'session_id'], dtype={0:str,1:int} )

    return user_map['user_id_org'], session_map['session_id_org']

def ensure_dir(file_path):
    '''
    Create all directories in the file_path if non-existent.
        --------
        file_path : string
            Path to the a file
    '''
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
