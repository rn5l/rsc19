'''
Created on May 24, 2019

@author: malte
'''
from pathlib import Path
import time

from config.globals import BASE_PATH
from domain.action_types import CLICK, IMAGE, INFO, DEALS, RATING, SEARCH, POI, \
    DEST, FILTER, SORT
from domain.features import PRICE_FEATURES, GEO_FEATURES
from featuregen.geo import geo_features
from featuregen.geo import haversine
from featuregen.popularity import print_col_list
from featuregen.price import price_features
from featuregen.session import counts_for_mask, prices_for_actions, \
    add_from_file, rating_for_actions, distance_for_actions, add_last_poi, \
    POI_FOLDER, stars_for_actions
from helper.df_ops import copy_features, reduce_mem_usage
from helper.loader import load_hdfs, write_feather, load_feather
import numpy as np


TEST_FOLDER = BASE_PATH + 'competition/'
CRAWL_FOLDER = BASE_PATH + 'crawled/'

ACTION_MAP = {}

ITEM_ACTIONS = [CLICK,IMAGE,INFO,DEALS,RATING]
COUNT_ACTIONS = [SEARCH,POI,SORT,DEST,FILTER]

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    examples = price_features( TEST_FOLDER, log, examples )
    
    cols = PRICE_FEATURES
    cols.remove('price_city_all_permean')
    cols.remove('price_city_all_mean')
    cols.remove('price_platform_all_permean')
    cols.remove('price_platform_all_mean')
    cols.remove('price_city_platform_all_permean')
    cols.remove('price_city_platform_all_mean')
    cols.remove('price_city_click_permean')
    cols.remove('price_city_click_mean')
    cols.remove('price_platform_click_permean')
    cols.remove('price_platform_click_mean')
    cols.remove('price_city_platform_click_permean')
    cols.remove('price_city_platform_click_mean')
    cols.remove('price_city_impressions_permean')
    cols.remove('price_city_impressions_mean')
    cols.remove('price_platform_impressions_permean')
    cols.remove('price_platform_impressions_mean')
    cols.remove('price_city_platform_impressions_permean')
    cols.remove('price_city_platform_impressions_mean')
    for c in cols:
        del examples[c]
        
    examples = geo_features( TEST_FOLDER, CRAWL_FOLDER, log, examples )
    cols = GEO_FEATURES
    cols.remove('distance_city')
    cols.remove('distance_last')
    for c in cols:
        del examples[c]
        
    user_features( TEST_FOLDER, log, examples, crawl_path=CRAWL_FOLDER, poi_path=POI_FOLDER, redo=True )

def user_features(base_path, log, examples, price_path=None, crawl_path=None, poi_path=None, redo=False):
    
    name = 'user_features'
    if price_path is None:
        price_path = base_path
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( log, examples, price_path=price_path, crawl_path=crawl_path, poi_path=poi_path )
        examples = reduce_mem_usage(examples, cols=cols)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','label','step'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list( cols )
        
    return examples

def create_features( log, examples, price_path=None, crawl_path=None, poi_path=None ):
    
    tstart = time.time()
    print( 'create_features user' )
    
    cols_pre = examples.columns.values
    
    log, examples = user_price_features(log, examples, price_path=price_path)
    log, examples = user_rating_features(log, examples, crawl_path=crawl_path)
    log, examples = user_stars_features(log, examples, crawl_path=crawl_path)
    log, examples = user_distance_features(log, examples, crawl_path=crawl_path, poi_path=poi_path)
    
    log['session_hidden'] = log.groupby( 'session_id' ).hidden.transform(sum)
    mask = log['session_hidden'] == 0
    
    log, examples = user_pop_features(log[mask], examples)
    #examples = session_time_features(log, examples, price_path=price_path)
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( 'create_features user in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols

def user_pop_features( log, examples ):
    
    log['ustep'] = log.groupby('user_id').cumcount() + 1
    log.loc[log.exclude == 1,'ustep'] = np.nan
    log['maxstep_all'] = log.groupby('user_id').ustep.transform( max )
    print(log)
    log['mrr'] = 1 / ( log['maxstep_all'] - log.ustep )
    log['mrr'] = log['mrr'].replace( [np.inf], np.nan ).fillna(0)
    
    log['stepsize'] = 1 / log['maxstep_all']
    log['linear'] = log.ustep * log['stepsize']
    log['mrr'] = log['mrr'].replace( [np.inf], np.nan ).fillna(0)
    del log['stepsize']
        
    key = 'user_all'
    examples = counts_for_mask(log, examples, group=['user_id'], key=key, step_key='user_all_maxstep')
    del examples['user_all_count_rel']
    
#     for action in COUNT_ACTIONS:
#         log_mask = log.action_type == action
#         key = 'user_' + NAMES[action]
#         examples = counts_for_mask(log, examples, mask=log_mask, group=['user_id'], key=key, step_key=key+'_maxstep')
#         
#     for action in ITEM_ACTIONS:
#         log_mask = log.action_type == action
#         key = 'user_' + NAMES[action]
#         examples = counts_for_mask(log, examples, mask=log_mask, group=['user_id'], key=key, step_key=key+'_maxstep')
#         
#         log_mask = (log.action_type == action) & ~log.reference.isnull()
#         key = 'user_item_' + NAMES[action]
#         examples = counts_for_mask(log, examples, mask=log_mask, decay=False, group=['user_id','reference'], group_examples=['user_id','impressions'], key=key, step_key=key+'_maxstep')
        
    log_mask = log.action_type.isin(ITEM_ACTIONS) & ~log.reference.isnull()
    key = 'user_item_all'
    examples = counts_for_mask(log, examples, mask=log_mask, decay=False, group=['user_id','reference'], group_examples=['user_id','impressions'], key=key, step_key=key+'_maxstep')
    
    
    del log['maxstep_all']
    
    return log, examples

def user_price_features( log, examples, price_path=None ):
    
#     price_map = pd.read_csv( price_path + 'tmp/' + 'city_price.csv', header=None, names=['city','price_city_impressions_mean'], dtype={0:np.int16, 1:np.float32}  )
#     price_map.index = price_map.city
#     price_map = price_map['price_city_impressions_mean']
#     
#     log['price_city_permean'] = log['city'].apply( lambda x: price_map.ix[x] if ~np.isnan(x) and x in price_map.index else np.nan )
#     log['price_city_permean'] = log['price_session'] / log['price_city_permean']
    
    examples = prices_for_actions(log, examples, actions=ITEM_ACTIONS, key='all', group=['user_id'], prefix='user')
    examples = prices_for_actions(log, examples, actions=[CLICK], key='click', group=['user_id'], prefix='user')
    examples = prices_for_actions(log, examples, actions=ITEM_ACTIONS, key='city_all', group=['user_id','city'], prefix='user')
    examples = prices_for_actions(log, examples, actions=[CLICK], key='city_click', group=['user_id','city'], prefix='user')
    
#     del log['price_city_permean']
    
    return log, examples

def user_rating_features( log, examples, crawl_path=None ):
    
    if 'ri_rating_percentage' not in examples.columns:  
        examples = add_from_file( crawl_path + 'item_info/crawl_ci.csv', examples, to=['impressions'], filter=['ci_rating_percentage'] )
    else:
        examples['ci_rating_percentage'] = examples['ri_rating_percentage']
    log = add_from_file( crawl_path + 'item_info/crawl_ci.csv', log, col=['item_id'], filter=['ci_rating_percentage'] )
    
    examples = rating_for_actions(log, examples, actions=ITEM_ACTIONS, group=['user_id'], key='all', prefix='user')
    examples = rating_for_actions(log, examples, actions=[CLICK], group=['user_id'], key='click', prefix='user')
    examples = rating_for_actions(log, examples, actions=ITEM_ACTIONS, group=['user_id','city'], key='city_all', prefix='user')
    examples = rating_for_actions(log, examples, actions=[CLICK], group=['user_id','city'], key='city_click', prefix='user')
    
    del log['ci_rating_percentage']
    del examples['ci_rating_percentage']
    
    return log, examples

def user_stars_features( log, examples, crawl_path=None ):
    
    if 'ci_stars' not in examples.columns:  
        examples = add_from_file( crawl_path + 'item_info/crawl_ci.csv', examples, to=['impressions'], filter=['ci_stars'] )
        examples['tmp_stars'] = examples['ci_stars']
        del examples['ci_stars']
    else:
        examples['tmp_stars'] = examples['ci_stars']
    log = add_from_file( crawl_path + 'item_info/crawl_ci.csv', log, col=['item_id'], filter=['ci_stars'] )
    
    examples = stars_for_actions(log, examples, actions=ITEM_ACTIONS, group=['user_id'], key='all', prefix='user')
    examples = stars_for_actions(log, examples, actions=[CLICK], group=['user_id'], key='click', prefix='user')
    examples = stars_for_actions(log, examples, actions=ITEM_ACTIONS, group=['user_id','city'], key='city_all', prefix='user')
    examples = stars_for_actions(log, examples, actions=[CLICK], group=['user_id','city'], key='city_click', prefix='user')
    
    del log['ci_stars']
    del examples['tmp_stars']
    
    return log, examples

def user_distance_features( log, examples, crawl_path=None, poi_path=None ):
    
    log = add_from_file( crawl_path + 'item_info/crawl_ci.csv', log, col=['item_id'], filter=['ci_lat','ci_lng'] )
    log = add_from_file( crawl_path + 'city/city_latlng.csv', log, col=['city'], filter=['city_lat','city_lng'] )
    
    log = add_last_poi( poi_path, log )
    
    log = add_from_file( crawl_path + 'poi/poi_latlng.csv', log, col=['poi'], to=['last_poi'], filter=['poi_lat','poi_lng'] )
    
    log['distance_city'] = haversine(log.ci_lat, log.ci_lng, log.city_lat, log.city_lng)
    log['distance_poi'] = haversine(log.ci_lat, log.ci_lng, log.poi_lat, log.poi_lng)
    log['distance_last'] = log['distance_poi']
    mask = ~log['distance_city'].isnull() & log['distance_poi'].isnull()
    log.loc[mask, 'distance_last'] = log.loc[mask, 'distance_city']
    
    examples = distance_for_actions(log, examples, group=['user_id'], actions=ITEM_ACTIONS, key='all', prefix='user')
    examples = distance_for_actions(log, examples, group=['user_id'], actions=[CLICK], key='click', prefix='user')
    examples = distance_for_actions(log, examples, group=['user_id','city'], actions=ITEM_ACTIONS, key='city_all', prefix='user')
    examples = distance_for_actions(log, examples, group=['user_id','city'], actions=[CLICK], key='city_click', prefix='user')
    
    del log['distance_city']
    del log['distance_poi']
    del log['last_poi']
    del log['poi_lat'], log['poi_lng']
    del log['city_lat'], log['city_lng']
    del log['ci_lat'], log['ci_lng']
    
    return log, examples

if __name__ == '__main__':
    main()
    