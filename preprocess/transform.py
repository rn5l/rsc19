'''
Created on Mar 9, 2019

@author: malte
'''

import pandas as pd
import numpy as np
import time
import swifter
from multiprocessing import Pool, cpu_count
from math import floor
from collections import Counter
from helper.apply_ops import join_pipe, map_list, split_pipe, change_type
from itertools import chain
import gc
from config.globals import BASE_PATH
from pathlib import Path
from helper.loader import load_feather, ensure_dir
from domain.action_types import DEST, POI, SORT, FILTER, CLICK
from helper.df_ops import expand

PATH_RAW = BASE_PATH + 'raw/'
PATH_PROCESSED = BASE_PATH + 'preprocessed/'

def main():
    
    if not Path( PATH_RAW + 'joined_raw.fthr' ).is_file():
        data = join_and_feather()
    else:
        data = load_feather(PATH_RAW + 'joined_raw.fthr')
    
    if not Path( PATH_PROCESSED + 'joined_tmp.fthr' ).is_file():
        data = clean_and_map(data)
        gc.collect()
    else:
        data = load_feather(PATH_PROCESSED + 'joined_tmp.fthr')
        
    if not Path( PATH_PROCESSED + 'joined_final.hd5' ).is_file():
        data = extend_mapping_and_meta(data)
        gc.collect()

def join_and_feather():
    train = pd.read_csv( PATH_RAW + 'train.csv' )
    test = pd.read_csv( PATH_RAW + 'test.csv' )
    
    train['train'] = 1
    test['train'] = 0
    
    complete = pd.concat( [train,test] )
    del train, test
    
    complete = complete.sort_values(['user_id','timestamp','step']).reset_index(drop=True)
    complete.to_feather( PATH_RAW + 'joined_raw.fthr' )
    
    return complete
    
    
def clean_and_map(data):
    
    #map cats 
    data = data.sort_values(['user_id','timestamp','step']).reset_index(drop=True)
    data = fix_sessions(data)
    data = fix_sessions_duplicate( data )
    ok = check_sessions(data)
    
    if not ok: 
        print('check data') 
        exit()
    
    data['item_id'] = pd.to_numeric( data.reference, errors='coerce' ).fillna(-1).astype(int)
    data = map_strings( data )
    #check_sessions(data)
    ensure_dir( PATH_PROCESSED )
    data.to_feather( PATH_PROCESSED + 'joined_tmp.fthr' )
    
    return data
    
def extend_mapping_and_meta(data):
    
    #extended mapping and meta info    
    data = split_list(data)
    data, meta = map_reference( data )
    data, meta = collect_prices( data, meta )
    
    meta.to_hdf( PATH_PROCESSED + 'meta_extended.hd5', key='table' )
    meta['price_list'] = meta['price_list'].apply( join_pipe )
    meta['properties'] = meta['properties'].apply( join_pipe )
    meta['properties_code'] = meta['properties_code'].apply( join_pipe )
    meta.to_csv( 'meta_extended.csv' )
    
    data['reference'] = pd.to_numeric(data.reference, downcast='integer', errors='coerce')
    data = data.merge( meta[['item_id','price_mean']], on='item_id', how='left' )

    data.to_hdf( PATH_PROCESSED + 'joined_final.hd5', key='table' )

def fix_sessions(data):
    
    data['session_id_pre'] = data.session_id.shift(1)
    data['step_pre'] = data.step.shift(1)
    
    stepcount = data.groupby( ['session_id','step'] ).size()
    stepcount = stepcount[ stepcount > 1 ].reset_index()
    problems = data[data.session_id.isin( stepcount.session_id.unique() )]
    puser = problems.user_id.unique()
    
    def check( row ):
        if row.session_id == row.session_id_pre and row.step <= row.step_pre:
            return 1
        return 0 
    problems['mark'] = problems[['session_id','step','session_id_pre','step_pre']]. apply( check, axis=1 )
    problems['mark'] = problems.groupby(['session_id']).mark.cumsum()
    problems['session_id'] = problems['session_id'] + '-' + problems['mark'].astype(str)
    test_session = problems[problems.reference.isnull()].session_id.unique()
    problems = problems[~problems.session_id.isin(test_session)]
    
#     data['tmp'] = data.groupby(['session_id','step']).cumcount().astype(str)
#     keep = problems & (data.action_type=='clickout_item') & (data.reference.isnull())
#     fix = problems & ~keep
    data.ix[problems.index, 'session_id'] = data.ix[problems.index].session_id + '-' + problems['mark'].astype(str)
    
    data[data.user_id.isin( puser )].to_csv('fixed.csv')
    
    del data['session_id_pre']
    
    return data
    

def fix_sessions_duplicate(data):
    
    print( 'fix_sessions_duplicate' )
    print( ' -- ', len( set(data[data.train==0].session_id.unique()) & set(data[data.train==1].session_id.unique()) ) )
    duplicate_sessions = set(data[data.train==0].session_id.unique()) & set(data[data.train==1].session_id.unique())
    mask = (data.train==1) & (data.session_id.isin(duplicate_sessions))
    data.ix[mask, 'session_id'] = data.ix[mask, 'session_id'] + '-1'
    print( ' -- ', len( set(data[data.train==0].session_id.unique()) & set(data[data.train==1].session_id.unique()) ) )

    return data

def check_sessions(data):
    
    cool = True
    
    data['sess_pre'] = data.session_id.shift(1)
    data['step_pre'] = data.step.shift(1)
    data['user_pre'] = data.user_id.shift(1)
    data['time_pre'] = data.timestamp.shift(1)
    
    show = ['user_id','session_id','step','timestamp','train']
    
    #data[data.user_id=='7X4FZTVRCDQA'][show].to_csv('debug.csv')
    
    double_step = data[data.duplicated( ['session_id','step'] )]
    if len( double_step ) > 0:
        print( 'dstep weird:', len(double_step), ' of ',len(data) )
        data[data.user_id.isin(double_step.user_id.unique())].to_csv('dstep_weird.csv')
        print( data[data.user_id.isin(double_step.user_id.unique())][show] )
        cool = False
        
#     double_click = data[ ( data.action_type == 'clickout item' ) & data.duplicated( ['session_id','action_type','timestamp','reference'] )]
#     if len( double_click ) > 0:
#         print( 'dclick weird:', len(double_click), ' of ',len(data) )
#         data[data.user_id.isin(double_click.user_id.unique())].to_csv('dstep_weird.csv')
#         print( data[data.user_id.isin(double_click.user_id.unique())][show] )
#         cool = False
    
    test = data[ (data.sess_pre == data.session_id) & (data.step <= data.step_pre) ]
    #print(test)
    #data[data.user_id.isin(test.user_id.unique())][show].to_csv('debug.csv')
    if len(test) > 0:
        print( 'step weird:', len(test), ' of ',len(data) )
        data[data.user_id.isin(test.user_id.unique())].to_csv('step_weird.csv')
        print( data[data.user_id.isin(test.user_id.unique())][show] )
        cool = False
        
    test = data[ (data.sess_pre == data.session_id) & (data.timestamp < data.time_pre) ]
    #print(test)
    if len(test) > 0:
        print( 'time weird:', len(test), ' of ',len(data) )
        data[data.user_id.isin(test.user_id.unique())].to_csv('time_weird.csv')
        print( data[data.user_id.isin(test.user_id.unique())][show] )
        cool = False
    
    del data['sess_pre'], data['step_pre'], data['user_pre'], data['time_pre']
    
    return cool

def load_meta( path=PATH_PROCESSED, return_list=False ):
    meta = pd.read_csv( path + 'item_metadata.csv' )
    del meta['list']
    
    unique = set()
    proplist = list()

    meta['properties'] = meta.properties.apply( split_pipe, convert=str, collect_unique=unique, collect_list=proplist if return_list else None  )
    
    return meta, unique, proplist

def map_strings(data):
    
    tstart = time.time()
    
    users = data.user_id.unique()
    sessions = data.session_id.unique()
    types = data.action_type.unique()
    cities = data.city.unique()
    platforms = data.platform.unique()
    devices = data.device.unique()
    
    print( ' -- uniques in {}'.format( (time.time() - tstart ) ) )
    
    tstart = time.time()
    
    user_map = pd.Series( index=users, data=range(len(users)) )
    session_map = pd.Series( index=sessions, data=range(len(sessions)) )
    type_map = pd.Series( index=types, data=range(len(types)) )
    city_map = pd.Series( index=cities, data=range(len(cities)) )
    platform_map = pd.Series( index=platforms, data=range(len(platforms)) )
    device_map = pd.Series( index=devices, data=range(len(devices)) )
    
    print( ' -- series in {}'.format( (time.time() - tstart ) ) )
    
    tstart = time.time()
    
    data['user_id'] = data['user_id'].swifter.apply( map_list, mapper=user_map ).values
    data['session_id'] = data['session_id'].swifter.apply( map_list, mapper=session_map ).values
    data['action_type'] = data['action_type'].swifter.apply( map_list, mapper=type_map ).values
    data['city'] = data['city'].swifter.apply(map_list, mapper=city_map ).values
    data['platform'] = data['platform'].swifter.apply( map_list, mapper=platform_map ).values
    data['device'] = data['device'].swifter.apply( map_list, mapper=device_map ).values
        
#     data['user_id'] = user_ids
#     data['session_id'] = session_ids
#     data['action_type'] = type_ids
#     data['city'] = city_ids
#     data['platform'] = platform_ids
#     data['device'] = device_ids
    
    print( ' -- apply in {}'.format( (time.time() - tstart ) ) )
        
    user_map.to_csv( PATH_PROCESSED + 'user_map.csv')
    session_map.to_csv( PATH_PROCESSED + 'session_map.csv')
    type_map.to_csv( PATH_PROCESSED + 'type_map.csv')
    city_map.to_csv( PATH_PROCESSED + 'city_map.csv')
    platform_map.to_csv( PATH_PROCESSED + 'platform_map.csv')
    device_map.to_csv( PATH_PROCESSED + 'device_map.csv')
    
    return data
    
def map_reference( data ):
    
    # city
    tstart = time.time()
    citymap = pd.Series.from_csv(PATH_PROCESSED+'city_map.csv').astype(int)
    tmp = data[data.action_type == DEST].reference.apply( map_list, mapper=citymap )
    data.ix[tmp.index, 'reference'] = tmp.values
    print( ' -- city mapping in {}'.format( (time.time() - tstart ) ) )
    
    #poi
    tstart = time.time()
    poiset = data[data.action_type == POI].reference.unique()
    poimap = pd.Series( index=list(poiset), data=range(len(poiset)) ).astype(int)
    tmp = data[data.action_type == POI].reference.apply( map_list, mapper=poimap )
    data.ix[tmp.index, 'reference'] = tmp.values
    poimap.to_csv( PATH_PROCESSED + 'poi_map.csv' )
    
    print( ' -- poi mapping in {}'.format( (time.time() - tstart ) ) )
    
    #sort
    tstart = time.time()
    sortset = data[data.action_type == SORT].reference.unique()
    sortmap = pd.Series( index=list(sortset), data=range(len(sortset)) ).astype(int)
    tmp = data[data.action_type == SORT].reference.apply( map_list, mapper=sortmap )
    data.ix[tmp.index, 'reference'] = tmp.values
    sortmap.to_csv( PATH_PROCESSED + 'sort_map.csv' )
    print( ' -- sort mapping in {}'.format( (time.time() - tstart ) ) )
    
    # filters
    meta, propset, proplist = load_meta( return_list = True )
    
    filterset = set()
    filterlist = list()
    
    currentset = set()
    currentlist = list()
    
    tstart = time.time()
    data[data.action_type == FILTER].reference.apply( split_pipe, convert=str, collect_unique=filterset, collect_list=filterlist )
    data['current_filters'] = data.current_filters.apply( split_pipe, collect_unique=currentset, collect_list=currentlist )

    print( ' -- apply in {}'.format( (time.time() - tstart ) ) )
    
    prop_count = Counter(proplist)
    filter_count = Counter(filterlist)
    current_count = Counter(currentlist)
        
    prop_count = pd.DataFrame.from_dict(prop_count, orient='index').rename(columns={0: 'count'}).sort_values( ['count'], ascending=False )
    filter_count = pd.DataFrame.from_dict(filter_count, orient='index').rename(columns={0: 'count'}).sort_values( ['count'], ascending=False )
    current_count = pd.DataFrame.from_dict(current_count, orient='index').rename(columns={0: 'count'}).sort_values( ['count'], ascending=False )

    prop_count.to_csv( PATH_PROCESSED + 'properties_count.csv' )
    filter_count.to_csv( PATH_PROCESSED + 'filter_count.csv' )
    current_count.to_csv( PATH_PROCESSED + 'current_count.csv' )
    
    prop_count = prop_count.reset_index()
    filter_count = filter_count.reset_index().sort_values( ['count'], ascending=True )
    current_count = current_count.reset_index().sort_values( ['count'], ascending=True )

    filter_count = filter_count[ np.in1d(filter_count['index'], prop_count['index'].values, invert=True) ]
    
    all_count = pd.concat([prop_count,filter_count])
    all_count = all_count.drop_duplicates( ['index'], keep='last' ).reset_index(drop=True)
    
    current_count = current_count[ np.in1d(current_count['index'], all_count['index'].values, invert=True) ]
    all_count = pd.concat([all_count,current_count]).reset_index(drop=True)
    print(all_count)
        
    prop_map = pd.Series( index=all_count['index'].values, data=all_count.index.values ).astype(int)
    prop_map.to_csv( PATH_PROCESSED + 'properties_map.csv' )

    refs = data[data.action_type == FILTER].reference.apply( map_list, mapper=prop_map )
    data.ix[refs.index, 'reference'] = refs.values
    
    data['current_filters'] = data.current_filters.apply( map_list, mapper=prop_map )
    data.ix[refs.index, 'reference'] = refs.values
    
    meta['properties_code'] = meta['properties'].apply( map_list, mapper=prop_map )
    
    print( ' -- properties mapping in {}'.format( (time.time() - tstart ) ) )
    
    return data, meta
        
#     #test to show weird sessions
#     def check( reference, name=None ):
#         if reference is None:
#             return False
#         filter_list = reference.split('|')
#         return any( name in s for s in filter_list)
#     
#     filtered = data[data.action_type == FILTER]
#     maks = filtered.reference.apply( check, name='Today' )
#     maks2 = filtered.reference.apply( check, name='Next Monday' )
#     print(filtered[maks | maks2][['platform','reference','user_id']])

def collect_prices( data, meta ):
    
    tstart = time.time()
    
    examples = data[data.action_type==CLICK][['session_id','impressions','prices']].copy()
    examples = expand(examples, columns=['impressions','prices'] )
    examples = examples.drop_duplicates( ['impressions','session_id'], keep='last' )
    examples['item_id'] = examples['impressions'].astype(int)
    del examples['impressions']
    
    print( 'expand prices in {}'.format( (time.time() - tstart) ) )
    
    group = examples.groupby('item_id')
    
    item_info = pd.DataFrame()
    item_info['price_last'] = group.prices.last()
    item_info['price_mean'] = group.prices.mean()
    item_info['price_min'] = group.prices.min()
    item_info['price_max'] = group.prices.max()
    item_info['price_size'] = group.prices.size()
    item_info['price_list'] = group.prices.apply( list )
    item_info = item_info.reset_index()
        
    meta = meta.merge(item_info, on='item_id',how='left')
    meta.ix[meta['price_list'].isnull(),'price_list'] = None
    
    print( 'all prices in {}'.format( (time.time() - tstart) ) )
    
    examples['price_session'] = examples['prices']
    del examples['prices']
        
    data = data.merge( examples, on=['session_id','item_id'], how='left' )
    
    print( 'sessions prices in {}'.format( (time.time() - tstart) ) )
    
    return data, meta
    

def split_list(data):
     
    tstart = time.time()
    data['impressions'] = data.impressions.apply( split_pipe )
    data['prices'] = data['prices'].apply( split_pipe ).values
    data['current_filters'] = data['current_filters'].apply( split_pipe, convert=str ).values
    print( 'convert lists in {}'.format( (time.time() - tstart ) ) )
     
    return data

# def transform_list(data):
#     
#     tstart = time.time()
#     data['impressions'] = data.impressions.apply( change_type, function=list )
#     data['prices'] = data['prices'].apply( change_type, function=list ).values
#     data['current_filters'] = data['current_filters'].apply( change_type, function=list ).values
#     print( 'convert lists in {}'.format( (time.time() - tstart ) ) )
#     
#     return data

if __name__ == '__main__':
    main()