'''
Created on May 24, 2019

@author: malte
'''
from pathlib import Path

from config.globals import BASE_PATH
from helper.df_ops import copy_features, reduce_mem_usage
from helper.loader import load_hdfs, write_feather, load_feather
import numpy as np
import pandas as pd
import time
from featuregen.popularity import print_col_list
import os

TEST_FOLDER = BASE_PATH + 'competition/'
LATENT_FOLDER = BASE_PATH + 'competition/'
KEYS = ['bprc_click','bprc_item','bprc_click','bprc_item','d2v_click','d2v_item']
SIZES = [32,32,128,128,32,32]

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    latent_sim_features( TEST_FOLDER, log, examples, latent_path=LATENT_FOLDER, keys=KEYS, sizes=SIZES, redo=True )

def latent_sim_features(base_path, log, examples, latent_path=None, keys=KEYS, sizes=SIZES, redo=False):
    
    name = 'latent_sim_features'
    if latent_path is None:
        latent_path = base_path
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( log, examples, latent_path=latent_path, keys=keys, sizes=sizes )
        examples = reduce_mem_usage(examples, cols=cols)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','label','step'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list( cols )
        
    return examples

def create_features( log, examples, latent_path=None, keys=None, sizes=32 ):
    
    tstart = time.time()
    print( 'create_features session' )
    
    cols_pre = examples.columns.values
    
    latent_item = {}
    latent_session = {}
    res = {}
    
    for i, key in enumerate(keys):
        latent_session[ key + str(sizes[i]) ] = {}
        latent = get_latent(latent_path, key, type='session', size=sizes[i])
        latent = latent[latent.index.isin( examples.session_id.unique() )]
        latent_to_dict( latent_session[ key + str(sizes[i]) ], latent )
        
        latent_item[ key + str(sizes[i]) ] = {}
        latent = get_latent(latent_path, key, type='item', size=sizes[i])
        latent = latent[latent.index.isin( examples.impressions.unique() )]
        latent_to_dict( latent_item[ key + str(sizes[i]) ], latent )
                
        res[ 'latent_sim_' + key + str(sizes[i]) ] = []
    
    impressions = examples.impressions.values
    session_ids = examples.session_id.values
    
    for j in range(len(examples)):
        item_id = impressions[j]
        session_id = session_ids[j]
        
        for i, key in enumerate(keys):
            if item_id in latent_item[ key + str(sizes[i]) ] and session_id in latent_session[ key + str(sizes[i]) ]:
                sim = np.dot( latent_item[ key + str(sizes[i]) ][item_id], latent_session[ key + str(sizes[i]) ][session_id].T )
                res[ 'latent_sim_' + key + str(sizes[i]) ].append( sim )
            else:
                res[ 'latent_sim_' + key + str(sizes[i]) ].append( np.nan )
                        
        if j % 10000 is 0: 
            print( 'processed {} of {} in {}'.format( j, len(examples), time.time() - tstart ) ) 
           
    for key in res.keys():
        examples[key] = res[key]
        examples[key].fillna( -1, inplace=True )
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( 'create_features session in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols

def latent_to_dict( d, df ):
    for i in df.index: 
        d[i] = df.loc[i].values

def get_latent(folder, key='nmf', type='session', size=16):
    
    print('\tadd latent ',key)
    
    tstart = time.time()
    
    path = folder + 'latent/' + key + '_'+type+'_features.' + str(size) + '.csv'
    
    if os.path.isfile(path):
        latent = pd.read_csv(path)
    else:
        print( path, ' not found' )
        exit()
        
    if type=='session':
        index = 'session_id'
    else:
        index = 'item_id'
        
    latent.index = latent[index]
    del latent[index]
        
    print('\t\tloaded latent features in {}'.format( (time.time()-tstart) ) )
    
    return latent
        

if __name__ == '__main__':
    main()
    