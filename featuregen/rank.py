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
from featuregen.popularity import print_col_list, pop_features
from featuregen.latent_sim import latent_sim_features
from featuregen.geo import geo_features
from featuregen.crawl import crawl_features
from featuregen.meta import meta_features


TEST_FOLDER = BASE_PATH + 'sample_test/'
CRAWL_FOLDER = BASE_PATH + 'crawled/'
LATENT_FOLDER = BASE_PATH + 'crawled/'
META_FOLDER = BASE_PATH + 'preprocessed/'

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    
    examples = pop_features(TEST_FOLDER, log, examples)
    #examples = crawl_features(TEST_FOLDER, CRAWL_FOLDER, log, examples)
    examples = geo_features(TEST_FOLDER, CRAWL_FOLDER, log, examples)
    examples = latent_sim_features(TEST_FOLDER, log, examples, LATENT_FOLDER)
    examples = meta_features(TEST_FOLDER, META_FOLDER, log, examples)
    
    rank_features( TEST_FOLDER, log, examples, redo=True )

def rank_features(base_path, log, examples, redo=False):
    
    name = 'rank_features'
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( log, examples )
        examples = reduce_mem_usage(examples)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','prices','label'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list(cols)
    
    return examples

def create_features( log, examples ):
    
    tstart = time.time()
    print( 'create_features rank' )
    
    cols_pre = examples.columns.values
    
    examples = rank( examples, order=['pop_click','pop_all','pop_impressions'], ascending=False, key='pop_count' )
    examples = rank( examples, order=['pop_click_per_view_sessions','pop_click_per_view','pop_all_per_impression'], ascending=False, key='pop_relative' )
    #examples = rank( examples, order=['prices','ri_rating_percentage'], ascending=[True,False], key='prices' )
    examples = rank( examples, order=['prices','rating'], ascending=[True,False], key='prices' )
    #examples = rank( examples, order=['ri_rating_percentage','prices'], ascending=[False,True], key='rating' )
    examples = rank( examples, order=['rating','prices'], ascending=[False,True], key='rating' )
    #examples = rank( examples, order=['ci_rating_per_price','prices'], ascending=[False,True], key='rating_price' )
    examples = rank( examples, order=['distance_last','prices'], ascending=[True,True], key='distance' )
    examples = rank( examples, order=['distance_last_per_price_norm','prices'], ascending=True, key='distance_price' )
    examples = rank( examples, order=['distance_last_per_rating','prices'], ascending=True, key='distance_rating' )
    examples = rank( examples, order=['latent_sim_bprc_click32','latent_sim_bprc_item32','pop_all'], ascending=False, key='latent' )
    
    examples['rank_sum'] = 0
    examples['rank_sum'] += examples['rank_pop_count']
    examples['rank_sum'] += examples['rank_pop_relative']
    examples['rank_sum'] += examples['rank_prices']
    examples['rank_sum'] += examples['rank_rating']
    #examples['rank_sum'] += examples['rank_rating_price']
    examples['rank_sum'] += examples['rank_distance']
    examples['rank_sum'] += examples['rank_distance_price']
    examples['rank_sum'] += examples['rank_distance_rating']
    examples['rank_sum'] += examples['rank_latent']
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( 'create_features geo in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols

# vectorized haversine function
def rank( examples, order=[], ascending=False, key='' ):
    
    examples['rank_'+key] = examples.sort_values(order, ascending=ascending).groupby('session_id').cumcount() + 1

    return examples
        

if __name__ == '__main__':
    main()
    