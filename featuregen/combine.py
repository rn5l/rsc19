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


TEST_FOLDER = BASE_PATH + 'competition/'
CRAWL_FOLDER = BASE_PATH + 'crawled/'

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    combine_features( TEST_FOLDER, log, examples, redo=True )

def combine_features(base_path, log, examples, redo=False):
    
    name = 'combine_features'
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( log, examples )
        examples = reduce_mem_usage(examples, cols=cols)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','prices','label'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list(cols)
    
    return examples

def create_features( log, examples ):
    
    tstart = time.time()
    print( 'create_features combine' )
    
    cols_pre = examples.columns.values
    
    examples['combine_pop_click_per_view_all_count'] = examples['pop_click_per_view'] * examples['session_item_all_count_rel']
    examples['combine_pop_click_per_impression_all_count'] = examples['pop_click_per_impression'] * examples['session_item_all_count_rel']
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( 'create_features combine in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols
        

if __name__ == '__main__':
    main()
    