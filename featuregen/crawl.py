'''
Created on May 24, 2019

@author: malte
'''
from pathlib import Path

from config.globals import BASE_PATH
from domain.action_types import CLICK, IMAGE, INFO, DEALS, RATING
from helper.df_ops import copy_features, reduce_mem_usage
from helper.loader import load_hdfs, write_feather, load_feather
import numpy as np
import pandas as pd
import time
from featuregen.geo import minmax_norm
from featuregen.popularity import print_col_list

TEST_FOLDER = BASE_PATH + 'competition/'
CRAWL_FOLDER = BASE_PATH + 'crawled/'

ACTION_MAP = {}
ACTION_MAP['all'] = [CLICK,IMAGE,INFO,DEALS,RATING]
ACTION_MAP['view'] = [IMAGE,INFO,DEALS,RATING]
ACTION_MAP['click'] = [CLICK]

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    crawl_features( TEST_FOLDER, CRAWL_FOLDER, log, examples, redo=True )

def crawl_features(base_path, crawl_path, log, examples, redo=False):
    
    name = 'crawl_features'
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( crawl_path, log, examples )
        examples = reduce_mem_usage(examples)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','prices','label'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list( cols )
    return examples

def create_features( crawl_path, log, examples ):
    
    tstart = time.time()
    print( 'create_features crawl' )
    
    cols_pre = examples.columns.values

    examples = add_from_file( crawl_path + 'item_info/crawl_ci.csv', examples )
    examples = add_from_file( crawl_path + 'item_rating/crawl_ri.csv', examples )
    examples = add_from_file( crawl_path + 'item_images/crawl_ii.csv', examples )
    
    #del examples['ci_lat'],examples['ci_lat']
    del examples['ci_rating_index'],examples['ci_rating_percentage']
    
#     ci_lat    ci_lng
#     ci_rating_index    ci_rating_percentage

    examples = minmax_norm(examples, 'ri_rating_percentage')
    examples = minmax_norm(examples, 'ci_stars')
    
    examples['ci_rating_per_price'] = examples['ri_rating_percentage'] / examples['prices']
    examples['ci_stars_per_price'] = examples['ci_stars'] / examples['prices']
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    #examples = fill_na(examples, new_cols, 0)
    
    print( 'create_features pop in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols
    
def add_from_file( file, examples, col=['item_id'], to=['impressions'] ):
    
    tstart = time.time()
    print( '\t add_from_file {}'.format(file) )
    
    toadd = pd.read_csv( file, index_col=0 )
        
    copy = False
    if col[0] in examples.columns:
        copy = True
    
    examples = examples.merge( toadd, left_on=to, right_on=col, how='left' )
    
    if copy: 
        examples[col[0]] = examples[col[0]+'_y']
        del examples[col[0]+'_y']
    else:
        del examples[col[0]]
    
    print( '\t add_from_file in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def fill_na( examples, cols, value ):
    for c in cols:
        examples[c] = examples[c].fillna(value)
    return examples

if __name__ == '__main__':
    main()
    