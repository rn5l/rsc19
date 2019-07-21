'''
Created on May 24, 2019

@author: malte
'''
from pathlib import Path

from config.globals import BASE_PATH
from helper.df_ops import copy_features, reduce_mem_usage
from helper.loader import load_hdfs, write_feather, load_feather
import numpy as np
import time
from featuregen.popularity import print_col_list
from featuregen.crawl import crawl_features
from featuregen.geo import geo_features
from domain.features import GEO_FEATURES, CRAWL_FEATURES


TEST_FOLDER = BASE_PATH + 'competition/'
CRAWL_PATH = BASE_PATH + 'crawled/'
SHIFTS=4

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    
    examples = crawl_features( TEST_FOLDER, CRAWL_PATH, log, examples )
    examples = geo_features(TEST_FOLDER, CRAWL_PATH, log, examples )
    
    cols = GEO_FEATURES
    cols.remove('distance_last')
    cols.remove('distance_city')
    for c in cols:
        del examples[c]
        
    cols = CRAWL_FEATURES
    cols.remove('ri_rating_percentage')
    cols.remove('ri_rating_index')
    cols.remove('ci_stars')
    for c in cols:
        del examples[c]
    
    list_context_features( TEST_FOLDER, log, examples, shifts=SHIFTS, redo=True )

def list_context_features(base_path, log, examples, shifts=SHIFTS, redo=False):
    
    name = 'list_context_features_' + str(shifts)
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( log, examples, shifts=shifts )
        examples = reduce_mem_usage(examples)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','prices','label','position'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list(cols)
        
    return examples

def create_features( log, examples, shifts=SHIFTS ):
    
    tstart = time.time()
    print( 'create_features list context' )
    
    cols_pre = examples.columns.values
    
    examples.sort_values( ['session_id','position'], inplace=True )
    
    shift_list = [ i for i in range(1,shifts+1) ] + [ i*-1 for i in range(1,shifts+1) ]
    
    for i in shift_list:
        shift( examples, 'session_id', i )
    
    list_conext( examples, 'prices', shifts=shifts )
    list_conext( examples, 'ri_rating_percentage', shifts=shifts )
    list_conext( examples, 'ri_rating_index', shifts=shifts )
    list_conext( examples, 'ci_stars', shifts=shifts )
    #list_conext( examples, 'distance_city' )
    list_conext( examples, 'distance_last', shifts=shifts )
    
    for i in shift_list:
        clean( examples, 'session_id', i )
    
    examples.sort_values( ['session_id','impressions'], inplace=True )
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( 'create_features pop in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols

def list_conext( dataset, col, shifts=3 ):
    
    shift_list = [ i for i in range(1,shifts+1) ] + [ i*-1 for i in range(1,shifts+1) ]
    
    for i in shift_list:
        shift( dataset, col, i )
        clear( dataset, col, i )
    
        dist( dataset, col, i )
    
    for i in [ i for i in range(1,shifts+1) ]:
        permean( dataset, col, shifts=i )
    
    for i in shift_list:
        clean( dataset, col, i )
    
def clear( dataset, col, shift ):
    if shift > 0: 
        key = 'session_id_pre'+str(shift)
        col = col + '_pre'+str(shift)
    else:
        key = 'session_id_post'+str(abs(shift))
        col = col + '_post'+str(abs(shift))
        
    mask = dataset['session_id'] != dataset[key]
    dataset.loc[ mask, col ] = np.nan

def clean( dataset, col, shift ):
    if shift > 0: 
        col = col + '_pre'+str(shift)
    else:
        col = col + '_post'+str(abs(shift))
        
    del dataset[col]

def dist( dataset, col, shift ):
    if shift > 0: 
        col_shift = col + '_pre'+str(shift)
        col_target = 'lc_' + col + '_pre'+str(shift)
    else:
        col_shift = col + '_post'+str(abs(shift))
        col_target = 'lc_' + col + '_post'+str(abs(shift))
        
    dataset[ col_target ] = dataset[ col_shift ] - dataset[ col ]
    dataset[ col_target ].fillna( 0, inplace=True )

def permean( dataset, col, shifts=2 ):
    
    shifts_list = [ i for i in range(1,shifts+1) ] + [ i*-1 for i in range(1,shifts+1) ]
    
    mean_of = [col]
    for shift in shifts_list: 
        if shift > 0: 
            col_shift = col + '_pre'+str(shift)
        else:
            col_shift = col + '_post'+str(abs(shift))
        mean_of.append( col_shift )
    
    dataset[ 'lc_' + col + '_permean'+str(shifts)+'_mean'] = dataset[mean_of].mean( axis=1 )
        
    dataset[ 'lc_' + col + '_permean'+str(shifts) ] = dataset[ col ] / dataset[ 'lc_' + col + '_permean'+str(shifts)+'_mean']
    dataset[ 'lc_' + col + '_permean'+str(shifts) ].fillna( 1, inplace=True )
    del dataset[ 'lc_' + col + '_permean'+str(shifts)+'_mean']

    
def shift( dataset, col, shift ):
    if shift > 0: 
        col_shift = col + '_pre'+str(shift)
    else:
        col_shift = col + '_post'+str(abs(shift))
    dataset[col_shift] = dataset[col].shift(shift)
    

if __name__ == '__main__':
    main()
    