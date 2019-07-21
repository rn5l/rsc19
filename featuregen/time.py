'''
Created on May 24, 2019

@author: malte
'''
from pathlib import Path

from config.globals import BASE_PATH
from helper.df_ops import copy_features, reduce_mem_usage, apply
from helper.loader import load_hdfs, write_feather, load_feather
import numpy as np
import pandas as pd
import time
from featuregen.popularity import print_col_list
import os
from pytz import country_timezones
import pytz
from datetime import datetime

TEST_FOLDER = BASE_PATH + 'competition/'
PREPROCESSED_FOLDER = BASE_PATH + 'preprocessed/'

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    time_features( TEST_FOLDER, log, examples, preprocessed_path=PREPROCESSED_FOLDER, redo=True )

def time_features(base_path, log, examples, preprocessed_path=PREPROCESSED_FOLDER, redo=False):
    
    name = 'time_features'
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( log, examples, preprocessed_path=preprocessed_path )
        examples = reduce_mem_usage(examples, cols=cols)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','label','step'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list( cols )
        
    return examples

def create_features( log, examples, preprocessed_path=None ):
    
    tstart = time.time()
    print( 'create_features time' )
    
    cols_pre = examples.columns.values

    platforms = get_platforms( preprocessed_path )
    examples = examples.merge( platforms[['timezone']], left_on='platform', right_index=True, how='left' )
    
    def to_time(row):
        res = datetime.fromtimestamp( row[0], pytz.timezone(row[1]) )
        return res
    
    examples['time'] = apply( examples, ['timestamp','timezone'], to_time, verbose=100000 )
    examples['hour'] = examples['time'].apply( lambda x: x.hour )
    examples['day'] = examples['time'].apply( lambda x: x.strftime('%w') ).astype(int)
    
    #time
    def hour_to_cat(x):
        if x >=5 and x <= 11: return 0
        elif x >=12 and x <= 17: return 1
        elif x >=18 and x <= 23: return 2
        else: return 3
        
    def hour_to_cat2(x):
        if x >=6 and x <= 11: return 0
        elif x >=12 and x <= 18: return 1
        elif x >=19 and x <= 24: return 2
        else: return 3
        
    examples['hour_cat1'] = examples['hour'].apply( hour_to_cat )    
    examples['hour_cat2'] = examples['hour'].apply( hour_to_cat2 )    
    
    del examples['time'], examples['timezone']
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( 'create_features time in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols

def get_platforms(folder, index='platform'):
    
    print('\tadd paltforms ')
    
    tstart = time.time()
    
    path = folder + 'platform_map.csv'
    
    if os.path.isfile(path):
        platforms = pd.read_csv(path, header=None, names=['platform_name', 'platform'])
    else:
        print( path, ' not found' )
        exit()
    
    print(platforms)
    platforms.index = platforms[index]
    del platforms[index]
        
    print('\t\tloaded platforms features in {}'.format( (time.time()-tstart) ) )
    
    def to_timezone(platform):
        if platform == 'UK':
            platform = 'GB'
        if platform == 'AA':
            platform = 'AU'
        return country_timezones( platform.lower() )[0]
    
    platforms['timezone'] = platforms['platform_name'].apply( to_timezone )
    
    return platforms
        

if __name__ == '__main__':
    main()
    