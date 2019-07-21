'''
Created on May 24, 2019

@author: malte
'''
from pathlib import Path

from config.globals import BASE_PATH
from domain.action_types import CLICK, IMAGE, INFO, DEALS, RATING, SEARCH, POI,\
    DEST, FILTER, SORT, NAMES
from helper.df_ops import copy_features, reduce_mem_usage, apply
from helper.loader import load_hdfs, write_feather, load_feather
import numpy as np
import pandas as pd
import time
from featuregen.price import price_features
from featuregen.popularity import print_col_list
from domain.features import PRICE_FEATURES


TEST_FOLDER = BASE_PATH + 'competition/'
CRAWL_FOLDER = BASE_PATH + 'crawled/'
POI_FOLDER = BASE_PATH + 'competition/'

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
    
#     examples = geo.geo_features( TEST_FOLDER, CRAWL_FOLDER, log, examples )
#     cols = GEO_FEATURES
#     cols.remove('distance_city')
#     cols.remove('distance_last')
#     for c in cols:
#         del examples[c]
    
    session_features( TEST_FOLDER, log, examples, crawl_path=CRAWL_FOLDER, poi_path=POI_FOLDER, redo=True )

def session_features(base_path, log, examples, price_path=None, crawl_path=CRAWL_FOLDER, poi_path=POI_FOLDER, redo=False):
    
    name = 'session_features'
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
    print( 'create_features session' )
    
    cols_pre = examples.columns.values
    
    log, examples = session_pop_features(log, examples)
    #log, examples = session_price_features(log, examples, price_path=price_path)
    #log, examples = session_rating_features(log, examples, crawl_path=crawl_path)
    #log, examples = session_distance_features(log, examples, crawl_path=crawl_path, poi_path=poi_path)
    examples = session_time_features(log, examples, price_path=price_path)
    examples = session_sort_features(log, examples)
    examples = session_filter_features(log, examples)
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( 'create_features session in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols

def session_pop_features( log, examples ):
    
    log['maxstep_all'] = log.groupby('session_id').step.transform( max )
    log['maxstep'] = log[log.exclude == 0].groupby('session_id').step.transform( max )
    log['mrr'] = 1 / ( log['maxstep'] - log.step )
    log['mrr'] = log['mrr'].replace( [np.inf], np.nan ).fillna(0)
    
    log['stepsize'] = 1 / log['maxstep']
    log['linear'] = log.step * log['stepsize']
    del log['stepsize'], log['maxstep']
        
    key = 'session_all'
    examples = counts_for_mask(log, examples, group=['session_id'], key=key)
    del examples['session_all_count_rel']
    
#     key = 'session_after_all'
#     examples = counts_for_mask(log, examples, after=True, group=['session_id'], key=key)
#     del examples['session_after_all_count_rel']
    
    for action in COUNT_ACTIONS:
        log_mask = log.action_type == action
        key = 'session_' + NAMES[action]
        examples = counts_for_mask(log, examples, mask=log_mask, group=['session_id'], key=key)
#         key = 'session_after_' + NAMES[action]
#         examples = counts_for_mask(log, examples, after=True, mask=log_mask, group=['session_id'], key=key)
        
    for action in ITEM_ACTIONS:
        log_mask = log.action_type == action
        key = 'session_' + NAMES[action]
        examples = counts_for_mask(log, examples, mask=log_mask, group=['session_id'], key=key)
#         key = 'session_after_' + NAMES[action]
#         examples = counts_for_mask(log, examples, after=True, mask=log_mask, group=['session_id'], key=key)
        
        log_mask = (log.action_type == action) & ~log.reference.isnull()
        key = 'session_item_' + NAMES[action]
        examples = counts_for_mask(log, examples, mask=log_mask, decay=True, group=['session_id','reference'], group_examples=['session_id','impressions'], key=key)
#         key = 'session_after_item_' + NAMES[action]
#         examples = counts_for_mask(log, examples, after=True, mask=log_mask, group=['session_id','reference'], group_examples=['session_id','impressions'], key=key)
    
    log_mask = log.action_type.isin(ITEM_ACTIONS) & ~log.reference.isnull()
    key = 'session_item_all'
    examples = counts_for_mask(log, examples, mask=log_mask, decay=True, group=['session_id','reference'], group_examples=['session_id','impressions'], key=key)
    
    examples['session_size'] = examples['step']
    
    del log['maxstep_all']
    
    return log, examples

def session_price_features( log, examples, price_path=None ):
    
#     price_map = pd.read_csv( price_path + 'tmp/' + 'city_price.csv', header=None, names=['city','price_city_impressions_mean'], dtype={0:np.int16, 1:np.float32}  )
#     price_map.index = price_map.city
#     price_map = price_map['price_city_impressions_mean']
#     
#     log['price_city_permean'] = log['city'].apply( lambda x: price_map.ix[x] if ~np.isnan(x) and x in price_map.index else np.nan )
#     log['price_city_permean'] = log['price_session'] / log['price_city_permean']
    
    examples = prices_for_actions(log, examples, actions=ITEM_ACTIONS, key='all')
    examples = prices_for_actions(log, examples, actions=[CLICK], key='click')
    examples = prices_for_actions(log, examples, group=['session_id','city'], actions=ITEM_ACTIONS, key='city_all')
    examples = prices_for_actions(log, examples, group=['session_id','city'], actions=[CLICK], key='city_click')
    
#     del log['price_city_permean']
    
    return log, examples

def session_rating_features( log, examples, crawl_path=None ):
    
    if 'ri_rating_percentage' not in examples.columns:  
        examples = add_from_file( crawl_path + 'item_info/crawl_ci.csv', examples, to=['impressions'], filter=['ci_rating_percentage'] )
    else:
        examples['ci_rating_percentage'] = examples['ri_rating_percentage']
    log = add_from_file( crawl_path + 'item_info/crawl_ci.csv', log, col=['item_id'], filter=['ci_rating_percentage'] )
    
    examples = rating_for_actions(log, examples, actions=ITEM_ACTIONS, key='all')
    examples = rating_for_actions(log, examples, actions=[CLICK], key='click')
    
    del log['ci_rating_percentage']
    del examples['ci_rating_percentage']
    
    return log, examples

def session_distance_features( log, examples, crawl_path=None, poi_path=None ):
    
    log = add_from_file( crawl_path + 'item_info/crawl_ci.csv', log, col=['item_id'], filter=['ci_lat','ci_lng'] )
    log = add_from_file( crawl_path + 'city/city_latlng.csv', log, col=['city'], filter=['city_lat','city_lng'] )
    
    log = add_last_poi(poi_path, log)
    
    log = add_from_file( crawl_path + 'poi/poi_latlng.csv', log, col=['poi'], to=['last_poi'], filter=['poi_lat','poi_lng'] )
    
    log['distance_city'] = haversine(log.ci_lat, log.ci_lng, log.city_lat, log.city_lng)
    log['distance_poi'] = haversine(log.ci_lat, log.ci_lng, log.poi_lat, log.poi_lng)
    log['distance_last'] = log['distance_poi']
    mask = ~log['distance_city'].isnull() & log['distance_poi'].isnull()
    log.loc[mask, 'distance_last'] = log.loc[mask, 'distance_city']
    
    examples = distance_for_actions(log, examples, actions=ITEM_ACTIONS, key='all')
    examples = distance_for_actions(log, examples, actions=[CLICK], key='click')
    
    del log['distance_city']
    del log['distance_poi']
    del log['last_poi']
    del log['poi_lat'], log['poi_lng']
    del log['city_lat'], log['city_lng']
    del log['ci_lat'], log['ci_lng']
    
    return log, examples


def add_last_poi( poi_path, log ):
    
    def _add_last_poi(row, save=None):
        
        session = row[0]
        action = row[1]
        ref = row[2]
        city = row[3]
        
        if 'session' in save and save['session'] != session or not 'session' in save:
            #new session
            save['session'] = session
            save['last_poi'] = -1
        
        if 'city' in save and save['city'] != city or not 'city' in save:
            #new session
            save['city'] = city
            save['last_poi'] = -1
        
        if action == POI and not np.isnan( ref ):
            save['last_poi'] = ref
        
        return save['last_poi']
    
    file = poi_path + 'last_poi.fthr'
    
    if not Path( file ).is_file():
        log_full = load_hdfs( poi_path + 'data_log.hd5' )
        log_full['last_poi'] = apply(log_full, ['session_id','action_type','reference','city'], _add_last_poi, verbose=100000)
        write_feather( log_full[['session_id','last_poi']], file )
    
    last_poi = load_feather( file )
    print( len(last_poi) )
    last_poi = last_poi[last_poi.session_id.isin( log.session_id.unique() )]
    
    print( len(last_poi) )
    print( len(log) )
    log['last_poi'] = last_poi['last_poi'].values
    del last_poi
    
    return log

def session_time_features( log, examples, price_path=None ):
    
    log['dwell'] = log['timestamp'].shift( -1 )
    log['dwell'] = log['dwell'] - log['timestamp']
    log['session_next'] = log['session_id'].shift( -1 )
    log.loc[ log['session_id'] != log['session_next'], 'dwell'] = np.nan
    
    del log['session_next']

    #DOOO
    examples = times_for_item(log, examples)
    mask = log.action_type.isin( ITEM_ACTIONS )
    examples = times_for_item(log, examples, mask, group=['session_id','reference'], group_example=['session_id','impressions'], key='item' )
    examples = times_for_item(log, examples, mask, group=['session_id','city'], group_example=['session_id','city'], key='city' )
    
    examples = last_for_reference(log, examples, action=POI, group=['session_id'], key='all_poi' )
    examples = last_for_reference(log, examples, action=SORT, group=['session_id'], key='all_sort' )
    examples = last_for_reference(log, examples, action=FILTER, group=['session_id'], key='all_filter' )
    
    examples = last_for_reference(log, examples, action=POI, group=['session_id','city'], key='poi' )
    examples = last_for_reference(log, examples, action=SORT, group=['session_id','city'], key='sort' )
    examples = last_for_reference(log, examples, action=FILTER, group=['session_id','city'], key='filter' )
    
    examples = last_action(log, examples)
    
#     dataset['session_sort_filter_last'] = dataset['session_sort_filter_last'].fillna(-1)
#     dataset['session_filter_special_last'] = dataset['session_filter_special_last'].fillna(-1)
    
    del log['dwell']
    
    return examples

def session_sort_features( log, examples ):
   
    sorts = list(range(7)) # originally 9
    
    for sid in sorts: 
        print(sid)
        examples = check_preference( log, examples, id=sid )
    
    # dataset['session_sort_filter_last'] = dataset['session_sort_filter_last'].fillna(-1)
    # dataset['session_filter_special_last'] = dataset['session_filter_special_last'].fillna(-1)
    
    return examples

def session_filter_features( log, examples ):

    #        sorts = [189, 191, 192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203]
    sorts = [194, 195, 196, 197, 198, 199, 200, 201, 202, 203]
    
    for sid in sorts: 
        print(sid)
        examples = check_preference( log, examples, id=sid, action=FILTER )
    
    # dataset['session_sort_filter_last'] = dataset['session_sort_filter_last'].fillna(-1)
    # dataset['session_filter_special_last'] = dataset['session_filter_special_last'].fillna(-1)
    
    return examples

def check_preference( log, examples, id=0, action=SORT, key='session_sort' ):
    
    mask = log.action_type == action
    mask = mask & (log.reference==id)
    
    sort = pd.DataFrame()
    sort['session_sort_'+str(id)] = log[mask].groupby( 'session_id' ).size()
    
    print( sum( sort['session_sort_'+str(id)] > 0 ) )
    
    examples = examples.merge( sort, left_on=['session_id'], right_index=True, how='left' )
    
    examples[key+'_'+str(id)].fillna( 0, inplace=True)
    
    return examples
    
def times_for_item(log, examples, mask_log=None, after=False, group=['session_id'], group_example=None, key=None):
    
    tstart = time.time()
    print( '\t times_for_item {}'.format(key) )
    
    base_key = 'session_'
    if key:
        base_key += key + '_'
        
    mask = mask_log if mask_log is not None else log.train > -1
    mask = mask & ~log.reference.isnull()
    
    if not after:
        mask = mask & (log.exclude == 0)
        
    if group_example is None:
        group_example = group
        
    grouped = log[mask].groupby( group )
        
    price = pd.DataFrame()
    price[ base_key + 'dwell' ] = grouped['dwell'].sum()
    price[ base_key + 'dwell_min' ] = grouped['dwell'].min()
    price[ base_key + 'dwell_max' ] = grouped['dwell'].max()
    price[ base_key + 'dwell_mean' ] = grouped['dwell'].mean()
    if len( group ) > 1:
        price[ base_key + 'min_step' ] = grouped['step'].min()
        price[ base_key + 'max_step' ] = grouped['step'].max()
        price[ base_key + 'min_time' ] = grouped['timestamp'].min()
        price[ base_key + 'max_time' ] = grouped['timestamp'].max()
    
    examples = examples.merge( price, left_on=group_example, right_index=True, how='left' )
    
    if len( group ) > 1:
        examples[base_key + 'max_time'] = examples['timestamp'] - examples[base_key + 'max_time']
        examples[base_key + 'min_time'] = examples['timestamp'] - examples[base_key + 'min_time']
        examples[base_key + 'max_step'] = ( examples['step'] - examples[base_key + 'max_step'] ) / examples['step']
        examples[base_key + 'min_step'] = ( examples['step'] - examples[base_key + 'min_step'] ) / examples['step']
    
    examples[ base_key + 'dwell' ].fillna( 0, inplace=True )
    examples[ base_key + 'dwell_min' ].fillna( 0, inplace=True )
    examples[ base_key + 'dwell_max' ].fillna( 0, inplace=True )
    examples[ base_key + 'dwell_mean' ].fillna( 0, inplace=True )
    if len( group ) > 1:
        pass
        examples[ base_key + 'min_time' ].fillna( -1, inplace=True )
        examples[ base_key + 'max_time' ].fillna( -1, inplace=True )
        examples[ base_key + 'min_step' ].fillna( -1, inplace=True )
        examples[ base_key + 'max_step' ].fillna( -1, inplace=True )
    
    print( '\t times_for_item in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def last_for_reference(log, examples, action=POI, group=['session_id'], group_example=None, key=None):
    
    tstart = time.time()
    print( '\t last_for_reference {}'.format(key) )
    
    base_key = 'session_last'
    if key:
        base_key += '_' + key
    
    mask = log.action_type == action
    mask = mask & (log.exclude == 0)
    
    if group_example is None:
        group_example = group
    
    grouped = log[mask].groupby( group )
        
    price = pd.DataFrame()
    price[ base_key ] = grouped['reference'].last()
    
    examples = examples.merge( price, left_on=group_example, right_index=True, how='left' )
    
    examples[ base_key ] = examples[ base_key ].fillna( -1 ).astype(np.int32)
    
    print( '\t last_for_reference in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def last_action(log, examples, group=['session_id'], key=None):
    
    tstart = time.time()
    print( '\t last_action {}'.format(key) )

    base_key = 'session_last_action'
    if key:
        base_key += '_' + key
    
    mask = log.hidden == 0
    mask = mask & (log.exclude == 0)
    
    grouped = log[mask].groupby( group )
        
    price = pd.DataFrame()
    price[ base_key ] = grouped['action_type'].last()
    
    examples = examples.merge( price, left_on=group, right_index=True, how='left' )
    examples[ base_key ] = examples[ base_key ].fillna( -1 ).astype(np.int32)
    
    print( '\t last_action in {}s'.format( (time.time() - tstart) ) )
    
    return examples


def prices_for_actions( log, examples, actions=[CLICK], after=False, group=['session_id'], key=None, prefix='session' ):
    
    tstart = time.time()
    print( '\t prices_for_actions {}'.format(key) )
    
    base_key = prefix + '_price_'
    if key:
        base_key += key + '_'
        
    mask = log.action_type.isin( actions )
    mask = mask & ~log.reference.isnull()
    mask = mask & ~log.price_session.isnull()
    
    if not after:
        mask = mask & (log.exclude == 0)
        
    grouped = log[mask].drop_duplicates(['session_id','reference','action_type']).groupby( group )
        
    price = pd.DataFrame()
    #price[ base_key + 'min' ] = grouped['price_session'].min()
    #price[ base_key + 'max' ] = grouped['price_session'].max()
    price[ base_key + 'mean' ] = grouped['price_session'].mean()
    #price[ base_key + 'city_permean' ] = grouped['price_city_permean'].mean()
    
    examples = examples.merge( price, left_on=group, right_index=True, how='left' )
    del price
    
    examples[base_key + 'mean_dist'] = examples[base_key + 'mean'] - examples['prices']
    #examples[base_key + 'min_dist'] = examples[base_key + 'min'] - examples['prices']
    #examples[base_key + 'max_dist'] = examples[base_key + 'max'] - examples['prices']
    
    if len(actions) > 1:
        key_comp = 'all'
    else: 
        key_comp = 'click'
    
    examples[base_key + 'city_'+key_comp+'_permean_dist'] = examples[base_key + 'mean'] / examples['price_city_'+key_comp+'_mean']
    examples[base_key + 'city_'+key_comp+'_permean_dist'] = examples[base_key + 'city_'+key_comp+'_permean_dist'] - examples['price_city_'+key_comp+'_permean']
    
    examples[base_key + 'platform_'+key_comp+'_permean_dist'] = examples[base_key + 'mean'] / examples['price_platform_'+key_comp+'_mean']
    examples[base_key + 'platform_'+key_comp+'_permean_dist'] = examples[base_key + 'platform_'+key_comp+'_permean_dist'] - examples['price_platform_'+key_comp+'_permean']
    
    examples[base_key + 'city_platform_'+key_comp+'_permean_dist'] = examples[base_key + 'mean'] / examples['price_city_platform_'+key_comp+'_mean']
    examples[base_key + 'city_platform_'+key_comp+'_permean_dist'] = examples[base_key + 'city_platform_'+key_comp+'_permean_dist'] - examples['price_city_platform_'+key_comp+'_permean']
    
    examples[base_key + 'city_impressions_permean_dist'] = examples[base_key + 'mean'] / examples['price_city_impressions_mean']
    examples[base_key + 'city_impressions_permean_dist'] = examples[base_key + 'city_impressions_permean_dist'] - examples['price_city_impressions_permean']
    
    examples[base_key + 'platform_impressions_permean_dist'] = examples[base_key + 'mean'] / examples['price_platform_impressions_mean']
    examples[base_key + 'platform_impressions_permean_dist'] = examples[base_key + 'platform_impressions_permean_dist'] - examples['price_platform_impressions_permean']
    
    examples[base_key + 'city_platform_impressions_permean_dist'] = examples[base_key + 'mean'] / examples['price_city_platform_impressions_mean']
    examples[base_key + 'city_platform_impressions_permean_dist'] = examples[base_key + 'city_platform_impressions_permean_dist'] - examples['price_city_platform_impressions_permean']
    
#     examples[base_key + 'city_'+key_comp+'_permed_dist'] = examples[base_key + 'mean'] / examples['price_city_'+key_comp+'_median']
#     examples[base_key + 'city_'+key_comp+'_permed_dist'] = examples[base_key + 'city_'+key_comp+'_permed_dist'] - examples['price_city_'+key_comp+'_permed']
#     
#     examples[base_key + 'platform_'+key_comp+'_permed_dist'] = examples[base_key + 'mean'] / examples['price_platform_'+key_comp+'_median']
#     examples[base_key + 'platform_'+key_comp+'_permed_dist'] = examples[base_key + 'platform_'+key_comp+'_permed_dist'] - examples['price_platform_'+key_comp+'_permed']
#     
#     examples[base_key + 'city_platform_'+key_comp+'_permed_dist'] = examples[base_key + 'mean'] / examples['price_city_platform_'+key_comp+'_median']
#     examples[base_key + 'city_platform_'+key_comp+'_permed_dist'] = examples[base_key + 'city_platform_'+key_comp+'_permed_dist'] - examples['price_city_platform_'+key_comp+'_permed']
#     
#     examples[base_key + 'city_impressions_permed_dist'] = examples[base_key + 'mean'] / examples['price_city_impressions_median']
#     examples[base_key + 'city_impressions_permed_dist'] = examples[base_key + 'city_impressions_permed_dist'] - examples['price_city_impressions_permed']
#     
#     examples[base_key + 'platform_impressions_permed_dist'] = examples[base_key + 'mean'] / examples['price_platform_impressions_median']
#     examples[base_key + 'platform_impressions_permed_dist'] = examples[base_key + 'platform_impressions_permed_dist'] - examples['price_platform_impressions_permed']
#     
#     examples[base_key + 'city_platform_impressions_permed_dist'] = examples[base_key + 'mean'] / examples['price_city_platform_impressions_median']
#     examples[base_key + 'city_platform_impressions_permed_dist'] = examples[base_key + 'city_platform_impressions_permed_dist'] - examples['price_city_platform_impressions_permed']
    
    if sum( np.isinf(examples[base_key + 'mean_dist']) ):
        print('mean inf')
        print( examples[np.isinf(examples[base_key + 'mean'])] )
        examples[np.isinf(examples[base_key + 'mean'])].to_csv('debug.csv')
        print( sum( np.isinf(examples[base_key + 'mean']) ) )
        exit()
    if sum( np.isinf(examples[base_key + 'city_'+key_comp+'_permean_dist']) ):
        print('city_all_permean_dist inf')
        print( examples[np.isinf(examples[base_key + 'city_'+key_comp+'_permean_dist'])] )
        print( examples[np.isinf(examples['price_city_'+key_comp+'_permean'])] )
        examples[np.isinf(examples[base_key + 'city_'+key_comp+'_permean_dist'])].to_csv('debug.csv')
        print( sum( np.isinf(examples[base_key + 'city_'+key_comp+'_permean_dist']) ) )
        exit()
        
    examples[ base_key + 'mean' ].fillna( -1, inplace=True )
    #examples[ base_key + 'min' ].fillna( -1, inplace=True )
    #examples[ base_key + 'max' ].fillna( -1, inplace=True )
    examples[ base_key + 'mean_dist' ].fillna( 0, inplace=True )
    #examples[ base_key + 'min_dist' ].fillna( 0, inplace=True )
    #examples[ base_key + 'max_dist' ].fillna( 0, inplace=True )
    examples[ base_key + 'city_'+key_comp+'_permean_dist' ].fillna( 0, inplace=True )
    examples[ base_key + 'platform_'+key_comp+'_permean_dist' ].fillna( 0, inplace=True )
    examples[ base_key + 'city_platform_'+key_comp+'_permean_dist' ].fillna( 0, inplace=True )
    examples[ base_key + 'city_impressions_permean_dist' ].fillna( 0, inplace=True )
    examples[ base_key + 'platform_impressions_permean_dist' ].fillna( 0, inplace=True )
    examples[ base_key + 'city_platform_impressions_permean_dist' ].fillna( 0, inplace=True )
    
#     examples[ base_key + 'city_'+key_comp+'_permed_dist' ].fillna( 0, inplace=True )
#     examples[ base_key + 'platform_'+key_comp+'_permed_dist' ].fillna( 0, inplace=True )
#     examples[ base_key + 'city_platform_'+key_comp+'_permed_dist' ].fillna( 0, inplace=True )
#     examples[ base_key + 'city_impressions_permed_dist' ].fillna( 0, inplace=True )
#     examples[ base_key + 'platform_impressions_permed_dist' ].fillna( 0, inplace=True )
#     examples[ base_key + 'city_platform_impressions_permed_dist' ].fillna( 0, inplace=True )
    
#     del examples[ base_key + 'city_permean' ]
    
    print( '\t prices_for_actions in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def rating_for_actions( log, examples, actions=[CLICK], after=False, group=['session_id'], key=None, prefix='session' ):
    
    tstart = time.time()
    print( '\t rating_for_actions {}'.format(key) )
    
    base_key = prefix + '_rating_'
    if key:
        base_key += key + '_'
        
    mask = log.action_type.isin( actions )
    mask = mask & ~log.ci_rating_percentage.isnull()
    mask = mask & (log.hidden == 0)
    
    if not after:
        mask = mask & (log.exclude == 0)
        
    grouped = log[mask].drop_duplicates(['session_id','reference','action_type']).groupby( group )
        
    rating = pd.DataFrame()
    #rating[ base_key + 'min' ] = grouped['ci_rating_percentage'].min()
    #rating[ base_key + 'max' ] = grouped['ci_rating_percentage'].max()
    rating[ base_key + 'mean' ] = grouped['ci_rating_percentage'].mean()
    
    examples = examples.merge( rating, left_on=group, right_index=True, how='left' )
    del rating
    
    examples[base_key + 'mean_dist'] = examples['ci_rating_percentage'] - examples[base_key + 'mean']
    #examples[base_key + 'min_dist'] = examples['ci_rating_percentage'] - examples[base_key + 'min']
    #examples[base_key + 'max_dist'] = examples['ci_rating_percentage'] - examples[base_key + 'max']
    
    if sum( np.isinf(examples[base_key + 'mean_dist']) ):
        print('mean inf')
        print( examples[np.isinf(examples[base_key + 'mean'])] )
        examples[np.isinf(examples[base_key + 'mean'])].to_csv('debug.csv')
        print( sum( np.isinf(examples[base_key + 'mean']) ) )
        exit()
        
    examples[ base_key + 'mean' ].fillna( -1, inplace=True )
    #examples[ base_key + 'min' ].fillna( -1, inplace=True )
    #examples[ base_key + 'max' ].fillna( -1, inplace=True )
    examples[ base_key + 'mean_dist' ].fillna( 0, inplace=True )
    #examples[ base_key + 'min_dist' ].fillna( 0, inplace=True )
    #examples[ base_key + 'max_dist' ].fillna( 0, inplace=True )
    
    print( '\t rating_for_actions in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def stars_for_actions( log, examples, actions=[CLICK], after=False, group=['session_id'], key=None, prefix='session' ):
    
    tstart = time.time()
    print( '\t stars_for_actions {}'.format(key) )
    
    base_key = prefix + '_stars_'
    if key:
        base_key += key + '_'
        
    mask = log.action_type.isin( actions )
    mask = mask & ~log.ci_stars.isnull()
    mask = mask & (log.hidden == 0)
    
    if not after:
        mask = mask & (log.exclude == 0)
        
    grouped = log[mask].drop_duplicates(['session_id','reference','action_type']).groupby( group )
        
    rating = pd.DataFrame()
    #rating[ base_key + 'min' ] = grouped['ci_stars'].min()
    #rating[ base_key + 'max' ] = grouped['ci_stars'].max()
    rating[ base_key + 'mean' ] = grouped['ci_stars'].mean()
    
    examples = examples.merge( rating, left_on=group, right_index=True, how='left' )
    del rating
    
    examples[base_key + 'mean_dist'] = examples['tmp_stars'] - examples[base_key + 'mean']
    #examples[base_key + 'min_dist'] = examples['ci_stars'] - examples[base_key + 'min']
    #examples[base_key + 'max_dist'] = examples['ci_stars'] - examples[base_key + 'max']
    
    if sum( np.isinf(examples[base_key + 'mean_dist']) ):
        print('mean inf')
        print( examples[np.isinf(examples[base_key + 'mean'])] )
        examples[np.isinf(examples[base_key + 'mean'])].to_csv('debug.csv')
        print( sum( np.isinf(examples[base_key + 'mean']) ) )
        exit()
        
    examples[ base_key + 'mean' ].fillna( -1, inplace=True )
    #examples[ base_key + 'min' ].fillna( -1, inplace=True )
    #examples[ base_key + 'max' ].fillna( -1, inplace=True )
    examples[ base_key + 'mean_dist' ].fillna( 0, inplace=True )
    #examples[ base_key + 'min_dist' ].fillna( 0, inplace=True )
    #examples[ base_key + 'max_dist' ].fillna( 0, inplace=True )
    
    print( '\t stars_for_actions in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def distance_for_actions( log, examples, actions=[CLICK], after=False, group=['session_id'], key=None, prefix='session' ):
    
    tstart = time.time()
    print( '\t distance_for_actions {}'.format(key) )
    
    base_key = prefix + '_distance_'
    if key:
        base_key += key + '_'
        
    mask = log.action_type.isin( actions )
    mask = mask & ~log.distance_city.isnull()
    mask = mask & (log.hidden == 0)
    
    if not after:
        mask = mask & (log.exclude == 0)
        
    grouped = log[mask].drop_duplicates(['session_id','reference','action_type']).groupby( group )
        
    rating = pd.DataFrame()
#     rating[ base_key + 'city_min' ] = grouped['distance_city'].min()
#     rating[ base_key + 'city_max' ] = grouped['distance_city'].max()
#     rating[ base_key + 'city_mean' ] = grouped['distance_city'].mean()
    #rating[ base_key + 'last_min' ] = grouped['distance_last'].min()
    #rating[ base_key + 'last_max' ] = grouped['distance_last'].max()
    rating[ base_key + 'last_mean' ] = grouped['distance_last'].mean()
    
    examples = examples.merge( rating, left_on=group, right_index=True, how='left' )
    del rating
    
#     examples[base_key + 'city_mean_dist'] = examples['distance_city'] - examples[base_key + 'city_mean']
#     examples[base_key + 'city_min_dist'] = examples['distance_city'] - examples[base_key + 'city_min']
#     examples[base_key + 'city_max_dist'] = examples['distance_city'] - examples[base_key + 'city_max']
    examples[base_key + 'last_mean_dist'] = examples['distance_last'] - examples[base_key + 'last_mean']
#     examples[base_key + 'last_min_dist'] = examples['distance_last'] - examples[base_key + 'last_min']
#     examples[base_key + 'last_max_dist'] = examples['distance_last'] - examples[base_key + 'last_max']
    
#     if sum( np.isinf(examples[base_key + 'city_mean_dist']) ):
#         print('mean inf')
#         print( examples[np.isinf(examples[base_key + 'mean'])] )
#         examples[np.isinf(examples[base_key + 'mean'])].to_csv('debug.csv')
#         print( sum( np.isinf(examples[base_key + 'mean']) ) )
#         exit()
    if sum( np.isinf(examples[base_key + 'last_mean_dist']) ):
        print('mean inf')
        print( examples[np.isinf(examples[base_key + 'mean'])] )
        examples[np.isinf(examples[base_key + 'mean'])].to_csv('debug.csv')
        print( sum( np.isinf(examples[base_key + 'mean']) ) )
        exit()
        
#     examples[ base_key + 'city_mean' ].fillna( -1, inplace=True )
#     examples[ base_key + 'city_min' ].fillna( -1, inplace=True )
#     examples[ base_key + 'city_max' ].fillna( -1, inplace=True )
#     examples[ base_key + 'city_mean_dist' ].fillna( 0, inplace=True )
#     examples[ base_key + 'city_min_dist' ].fillna( 0, inplace=True )
#     examples[ base_key + 'city_max_dist' ].fillna( 0, inplace=True )
    examples[ base_key + 'last_mean' ].fillna( -1, inplace=True )
#     examples[ base_key + 'last_min' ].fillna( -1, inplace=True )
#     examples[ base_key + 'last_max' ].fillna( -1, inplace=True )
    examples[ base_key + 'last_mean_dist' ].fillna( 0, inplace=True )
#     examples[ base_key + 'last_min_dist' ].fillna( 0, inplace=True )
#     examples[ base_key + 'last_max_dist' ].fillna( 0, inplace=True )
    
    print( '\t distance_for_actions in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def counts_for_mask( log, examples, mask=None, after=False, decay=False, group=[], group_examples=None, key=None, step_key='step' ):
    
    tstart = time.time()
    print( '\t counts_for_mask {}'.format(key) )
    
    base_key = key + '_'
    
    if group_examples is None:
        group_examples = group
    group_log = group
    
    if mask is None:
        mask = log.train > -1
    if after:
        mask = mask & (log.exclude == 1)
    else:
        mask = mask & (log.exclude == 0)
        
    grouped = log[mask].groupby( group_log )
    
    pop = pd.DataFrame()
    pop[base_key + 'count'] = grouped.size()
    pop[base_key + 'maxstep'] = grouped['maxstep_all'].max()
    if not after and decay:
        pop[base_key + 'mrr'] = grouped['mrr'].sum()
        pop[base_key + 'linear'] = grouped['linear'].sum()
       
    examples = examples.merge( pop, left_on=group_examples, right_index=True, how='left' )
    
    examples[base_key + 'count'].fillna( 0, inplace = True )
    if not after and decay:
        examples[base_key + 'count_rel'] = examples[base_key + 'count'] / examples[step_key]
        examples[base_key + 'mrr'] = examples[base_key + 'mrr'] / examples[step_key]
        examples[base_key + 'linear'] = examples[base_key + 'linear'] / examples[step_key]
    elif not after:
        examples[base_key + 'count_rel'] = examples[base_key + 'count'] / examples[step_key]
    else:
        examples[base_key + 'count_rel'] = examples[base_key + 'count'] / (examples[base_key + 'maxstep'] - examples[step_key])
    
    examples[ base_key + 'count_rel' ].fillna( 0, inplace=True )
    if not after and decay:
        examples[ base_key + 'mrr' ].fillna( 0, inplace=True )
        examples[ base_key + 'linear' ].fillna( 0, inplace=True )
    
    del examples[base_key + 'maxstep']
    
    print( '\t counts_for_mask in {}s'.format( (time.time() - tstart) ) )
    
    return examples
        
def counts_for_actions( log, examples, group=[], key=None ):
    
    base_key = 'session_'
    if key is not None:
        base_key += key + '_'    
        
    tstart = time.time()
    print( '\t counts_for_actions {}'.format(key) )
        
    mergeon = group + ['session_id']
    
    groupon = group + ['session_id', 'action_type']
    grouped = log.groupby( groupon ).size()
    grouped = grouped.unstack( level=len(groupon)-1 )
    grouped.rename( columns=lambda x: base_key + NAMES[x], inplace=True )
    grouped.fillna(0, inplace=True)
        
    examples = examples.merge( grouped, left_on=mergeon, right_index=True )
    
    print( '\t counts_for_actions in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def add_from_file( file, examples, col=['item_id'], to=None, filter=None ):
    tstart = time.time()
    print( '\t add_from_file {}'.format(file) )
    
    keep = False
    if to is None:
        to = col
        keep = True
    
    toadd = pd.read_csv( file, index_col=0 )
    if filter is not None:
        toadd = toadd[col+filter]
    
    copy = False
    if col[0] in examples.columns:
        copy = True
    
    examples = examples.merge( toadd, left_on=to, right_on=col, how='left' )
    
    if copy and not keep: 
        examples[col[0]] = examples[col[0]+'_y']
        del examples[col[0]+'_y']
    elif not copy and not keep:
        del examples[col[0]]
    
    print( '\t add_from_file in {}s'.format( (time.time() - tstart) ) )
    
    return examples

# vectorized haversine function
def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))

if __name__ == '__main__':
    main()
    