'''
Created on May 24, 2019

@author: malte
'''
from pathlib import Path
import time

from config.globals import BASE_PATH
from domain.action_types import CLICK
from featuregen.meta import add_from_file
from featuregen.popularity import print_col_list
from helper.df_ops import copy_features, reduce_mem_usage
from helper.loader import load_hdfs, write_feather, load_feather
import numpy as np
import pandas as pd


TEST_FOLDER = BASE_PATH + 'competition/'
META_FOLDER = BASE_PATH + 'preprocessed/'

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    log = clean_test(log)
    
    stars_features( TEST_FOLDER, META_FOLDER, log, examples, redo=True )

def clean_test(log):
    mask = (log.train == 0) & (log.hidden == 1)
    log.ix[mask,'item_id'] = np.nan
    return log

def stars_features(base_path, meta_path, log, examples, redo=False):
    
    name = 'stars_features'
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( meta_path, log, examples )
        examples = reduce_mem_usage(examples)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','prices','label','position'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list(cols)
        
    return examples

def create_features( meta_path, log, examples ):
    
    tstart = time.time()
    print( 'create_features stars' )
    
    cols_pre = examples.columns.values
    
    use_from_meta = ['stars','rating']
    if not 'stars' in cols_pre:
        remove = True
        examples = add_from_file( meta_path + 'item_metadata.csv', examples, to=['impressions'], filter=use_from_meta )
    log = add_from_file( meta_path + 'item_metadata.csv', log, filter=use_from_meta )
    
    mask_log = ~log.reference.isnull() & (log.hidden == 0)
    examples = click_ratio(log, examples, mask_log, group=['stars'], group_examples=['stars'], key='stars_click' )
    examples = click_ratio(log, examples, mask_log, group=['stars','device'], group_examples=['stars','device'], key='stars_click_device' )
    examples = click_ratio(log, examples, mask_log, group=['stars','city'], group_examples=['stars','city'], key='stars_click_city' )
    examples = click_ratio(log, examples, mask_log, group=['stars','platform'], group_examples=['stars','platform'], key='stars_click_platform' )
    
    examples = click_ratio(log, examples, mask_log, group=['rating'], group_examples=['rating'], key='rating_click' )
    examples = click_ratio(log, examples, mask_log, group=['rating','device'], group_examples=['rating','device'], key='rating_click_device' )
    examples = click_ratio(log, examples, mask_log, group=['rating','city'], group_examples=['rating','city'], key='rating_click_city' )
    examples = click_ratio(log, examples, mask_log, group=['rating','platform'], group_examples=['rating','platform'], key='rating_click_platform' )
    
    if remove:
        del examples['stars'], examples['rating']
    del log['stars'], log['rating']
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( 'create_features stars in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols

def click_ratio( log, examples, mask_log, group=[], group_examples=None, key=None ):
    
    tstart = time.time()
    print( '\t click_ratio {}'.format(key) )
    
    base_key = 'ratio_'
    if not key is None:
        base_key += key + '_'
    
    if group_examples is None:
        group_examples = group
        
    mask_log = mask_log & ( log.action_type == CLICK )
    
    pos = pd.DataFrame()
    pos[base_key + 'count'] = log[mask_log].groupby( group ).size()
    
    examples = examples.merge( pos, left_on=group_examples, right_index=True, how='left' )
    examples[base_key + 'count'].fillna( 0, inplace = True )
    
    if len( group ) > 1:
        pos = pd.DataFrame()
        pos[base_key + 'basecount'] = log[mask_log].groupby( group[1:] ).size()
    
        examples = examples.merge( pos, left_on=group_examples[1:], right_index=True, how='left' )
        examples[base_key + 'basecount'].fillna( 0, inplace = True )
    
        examples[base_key + 'ratio'] = examples[base_key + 'count'] / examples[base_key + 'basecount']
        examples[base_key + 'ratio'].fillna( 0, inplace = True )
        del examples[base_key + 'basecount']
    else:
        examples[base_key + 'ratio'] = examples[base_key + 'count'] / len(log[mask_log])
        examples[base_key + 'ratio'].fillna( 0, inplace = True )
        
    print( '\t pos_ratio in {}s'.format( (time.time() - tstart) ) )
    
    return examples

if __name__ == '__main__':
    main()
    