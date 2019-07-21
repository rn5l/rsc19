'''
Created on May 24, 2019

@author: malte
'''
from pathlib import Path

from config.globals import BASE_PATH
from domain.action_types import CLICK, IMAGE, INFO, DEALS, RATING, SORT, FILTER
from helper.df_ops import copy_features, expand, reduce_mem_usage
from helper.loader import load_hdfs, write_hdfs, write_feather, load_feather
import numpy as np
import pandas as pd
import time
from featuregen.popularity import print_col_list
import numba


TEST_FOLDER = BASE_PATH + 'sample_mini/'

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    log = clean_test(log)
    position_features( TEST_FOLDER, log, examples, redo=True )

def clean_test(log):
    mask = (log.train == 0) & (log.hidden == 1)
    log.ix[mask,'item_id'] = np.nan
    return log

def position_features(base_path, log, examples, redo=False):
    
    name = 'position_features'
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( log, examples )
        examples = reduce_mem_usage(examples)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','prices','label','position'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list(cols)
        
    return examples

def create_features( log, examples ):
    
    tstart = time.time()
    print( 'create_features position' )
    
    cols_pre = examples.columns.values
    
    log = add_pos_to_log( log )
    
    mask_log = ~log.position_click.isnull()
    examples = pos_ratio(log, examples, mask_log, group=['position_click'], group_examples=['position'], key='click' )
    mask_log = mask_log & (log.hidden == 0)
    examples = pos_ratio(log, examples, mask_log, group=['position_click','device'], group_examples=['position','device'], key='click_device' )
    examples = pos_ratio(log, examples, mask_log, group=['position_click','platform'], group_examples=['position','platform'], key='click_platform' )
    examples = pos_ratio(log, examples, mask_log, group=['position_click','city'], group_examples=['position','city'], key='click_city' )
    examples = pos_ratio(log, examples, mask_log, group=['position_click','city','platform'], group_examples=['position','city','platform'], key='click_city_platform' )
    examples = inv_filter(log, examples, mask_log, group=['session_id'], key='invfilter' )
    
    examples = pos_by_group(log, examples, group=['impressions'], key='item' )
    mask_log = ~log.reference.isnull()
    examples = pos_by_group_log(log, examples, mask_log=mask_log, group=['device'], key='device', field='position_click' )
    examples = pos_by_group_log(log, examples, mask_log=mask_log, group=['platform'], key='platform', field='position_click' )
    examples = pos_by_group_log(log, examples, mask_log=mask_log, group=['city'], key='city', field='position_click' )
    examples = pos_by_group_log(log, examples, mask_log=mask_log, group=['city','platform'], key='city_platform', field='position_click' )
    
    examples = pos_by_group_log_session(log, examples, group=['session_id'], key='session' )
    
    mask_log = ~log.reference.isnull()
    examples = pos_by_group_log_session(log, examples, mask_log=mask_log, group=['session_id'], field='position_click', key='session_wrong' )
    
    mask_log = log.action_type == CLICK
    examples = pos_by_group_log_session(log, examples, mask_log=mask_log, group=['session_id'], key='session_click' )
    
    mask_log = (log.action_type == CLICK) & ~log.reference.isnull()
    examples = pos_by_group_log_session(log, examples, mask_log=mask_log, group=['session_id'], field='position_click', key='session_click_wrong' )
    #examples = pop_by_group(log, examples, mask_log, mask_examples, group=['device'], key='device')
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( 'create_features pop in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols

def pos_ratio( log, examples, mask_log, group=[], group_examples=None, key=None ):
    
    tstart = time.time()
    print( '\t pos_ratio {}'.format(key) )
    
    base_key = 'pos_'
    if not key is None:
        base_key += key + '_'
    
    if group_examples is None:
        group_examples = group
    
    pos = pd.DataFrame()
    pos[base_key + 'count'] = log[mask_log].groupby( group ).size()
    
    examples = examples.merge( pos, left_on=group_examples, right_index=True, how='left' )
    examples[base_key + 'count'].fillna( 0, inplace = True )
    
    if len(group) > 1:
        pos = pd.DataFrame()
        pos[base_key + 'basecount'] = log[mask_log].groupby( group[1:] ).size()
        
        examples = examples.merge( pos, left_on=group_examples[1:], right_index=True, how='left' )
        examples[base_key + 'basecount'].fillna( 1, inplace = True )
        examples[base_key + 'ratio'] = examples[base_key + 'count'] / examples[base_key + 'basecount']
        examples[base_key + 'ratio'].fillna( 0, inplace = True )
        del examples[base_key + 'basecount']
    else:
        examples[base_key + 'ratio'] = examples[base_key + 'count'] / len(log[mask_log])
        examples[base_key + 'ratio'].fillna( 0, inplace = True )
    
    print( '\t pos_ratio in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def inv_filter( log, examples, mask_log, group=[], group_examples=None, key=None ):
    
    tstart = time.time()
    print( '\t pos_ratio {}'.format(key) )
    
    base_key = 'pos_'
    if not key is None:
        base_key += key + '_'
    
    if group_examples is None:
        group_examples = group
    
    mask_log = mask_log & (log.position_invfilter == 1)
    
    pos = pd.DataFrame()
    pos[base_key + 'count'] = log[mask_log].groupby( group_examples ).size()
    
    examples = examples.merge( pos, left_on=group_examples, right_index=True, how='left' )
    examples[base_key + 'count'].fillna( 0, inplace = True )
        
    print( '\t pos_ratio in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def pos_by_group( log, examples, mask_examples=None, group=[], key=None ):
    
    tstart = time.time()
    print( '\t pos_by_group {}'.format(key) )
    
    base_key = 'pos_'
    if not key is None:
        base_key += key + '_'
        
    if mask_examples is None:
        mask_examples = examples.prices >= 0
        
    group_examples = group
    
    grouped = examples[mask_examples].groupby( group_examples )
    
    pos = pd.DataFrame()
    pos[base_key + 'mean'] = grouped['position'].mean()
    pos[base_key + 'min'] = grouped['position'].min()
    pos[base_key + 'max'] = grouped['position'].max()
    
    examples = examples.merge( pos, left_on=group_examples, right_index=True, how='left' )
    
    examples[base_key + 'dist'] = examples[base_key + 'mean'] - examples['position']
    
    examples[base_key + 'mean'].fillna( -1, inplace = True )
    examples[base_key + 'dist'].fillna( 0, inplace = True )
    examples[base_key + 'min'].fillna( -1, inplace = True )
    examples[base_key + 'max'].fillna( -1, inplace = True )
    
    print( '\t pop_by_group in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def pos_by_group_log_session( log, examples, mask_log=None, field='position', group=[], group_examples=None, key=None ):
    
    tstart = time.time()
    print( '\t pos_by_group_log {}'.format(key) )
    
    base_key = 'pos_'
    if not key is None:
        base_key += key + '_'
        
    if mask_log is None:
        mask_log = ~log[field].isnull()
    else:
        mask_log = mask_log & ~log[field].isnull()
    
    if group_examples is None:
        group_examples = group
    
    grouped = log[mask_log].groupby( group )
    
    pos = pd.DataFrame()
    pos[base_key + 'mean'] = grouped[field].mean()
    pos[base_key + 'min'] = grouped[field].min()
    pos[base_key + 'max'] = grouped[field].max()
    pos[base_key + 'last'] = grouped[field].last()
    pos[base_key + '2last'] = grouped[field].nth(-2)
    pos[base_key + 'altering'] = grouped[field].apply( is_sorted ) * 1
    
    examples = examples.merge( pos, left_on=group_examples, right_index=True, how='left' )
    
    examples[base_key + 'dist'] = examples['position'] - examples[base_key + 'last']
    examples[base_key + 'direction'] = examples[base_key + 'last'] - examples[base_key + '2last']
    
    examples[base_key + 'mean'].fillna( -1, inplace = True )
    examples[base_key + 'min'].fillna( -1, inplace = True )
    examples[base_key + 'max'].fillna( -1, inplace = True )
    examples[base_key + 'last'].fillna( -1, inplace = True )
    examples[base_key + '2last'].fillna( -1, inplace = True )
    examples[base_key + 'altering'].fillna( -1, inplace = True )
    examples[base_key + 'dist'].fillna( 0, inplace = True )
    examples[base_key + 'direction'].fillna( 0, inplace = True )
    
    print( '\t pos_by_group_log in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def pos_by_group_log( log, examples, mask_log=None, field='position', group=[], group_examples=None, key=None ):
    
    tstart = time.time()
    print( '\t pos_by_group_log {}'.format(key) )
    
    base_key = 'pos_'
    if not key is None:
        base_key += key + '_'
        
    if mask_log is None:
        mask_log = ~log[field].isnull()
    else:
        mask_log = mask_log & ~log[field].isnull()
    
    if group_examples is None:
        group_examples = group
    
    grouped = log[mask_log].groupby( group )
    
    pos = pd.DataFrame()
    pos[base_key + 'mean'] = grouped[field].mean()
    pos[base_key + 'min'] = grouped[field].min()
    pos[base_key + 'max'] = grouped[field].max()
    
    examples = examples.merge( pos, left_on=group_examples, right_index=True, how='left' )
    
    examples[base_key + 'mean_dist'] =  examples[base_key + 'mean'] - examples['position']
    
    examples[base_key + 'mean'].fillna( -1, inplace = True )
    examples[base_key + 'min'].fillna( -1, inplace = True )
    examples[base_key + 'max'].fillna( -1, inplace = True )
    examples[base_key + 'mean_dist'].fillna( -25, inplace = True )
    
    print( '\t pos_by_group_log in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def add_pos_to_log(log):
    
    tstart = time.time()
    
    current_pos = {}
    position = []
    position_click = []
    position_invfilter = []
    
    action_types = log.action_type.values
    references = log.reference.values
    item_ids = log.item_id.values
    sessions = log.session_id.values
    impressions = log.impressions.values
    cities = log.city.values
    
    current_session = -1
    current_city = -1
    session_done = False
    last_impressions = None
    
    for i in reversed(range(len(log))):
        
        action = action_types[i]
        reference = references[i]
        session = sessions[i]
        city = cities[i]
        item_id = item_ids[i]
        
        if current_session != session:
            current_session = session
            current_pos = {}
            session_done = False
            last_impressions = None
            
#         if current_city != city:
#             session_done = True
        
        if action == CLICK:
            impression = impressions[i]
            current_pos = { item_id: pos for pos,item_id in enumerate(impression) }
            
            if last_impressions is not None and not same(last_impressions, impression) or session_done:
                session_done = True
                position.append( np.nan )
                position_invfilter.append( True )
            else:
                if not np.isnan(reference) and reference in current_pos:
                    position.append( current_pos[reference] + 1 )
                else:
                    position.append( np.nan )
                position_invfilter.append( False )
            
            if not np.isnan(item_id) and item_id in current_pos:
                position_click.append( current_pos[item_id] + 1 )
            else:
                position_click.append( np.nan )
                
            last_impressions = impression
            
        elif action == SORT:
            current_pos = {}
            position.append( np.nan )
            position_click.append( np.nan )
            position_invfilter.append( False )
        elif action == FILTER:
            current_pos = {}
            position.append( np.nan )
            position_click.append( np.nan )
            position_invfilter.append( False )
        elif action in {IMAGE,INFO,RATING,DEALS}:
            if not np.isnan(reference) and reference in current_pos and not session_done:
                position.append( current_pos[reference] + 1 )
            else:
                position.append( np.nan )
                
            if not np.isnan(item_id) and item_id in current_pos:
                position_click.append( current_pos[reference] + 1 )
            else:
                position_click.append( np.nan )
                
            position_invfilter.append( False )
        else:
            position.append( np.nan )
            position_click.append( np.nan )
            position_invfilter.append( False )
    
    log['position'] = list(reversed(position))
    log['position_click'] = list(reversed(position_click))
    log['position_invfilter'] = list(reversed(position_invfilter))
    
    print( 'sessions position in {}'.format( (time.time() - tstart) ) )
    
    #log.to_csv(TEST_FOLDER +  'debug.csv', index=False)
    
    return log

@numba.jit
def same(l1,l2):
    if len(l1) != len(l2):
        return False
    for i, n in enumerate(l1):
        if l1[i] != l2[i]:
            return False
    return True

@numba.jit        
def is_sorted(l):
    for i in range(len(l) - 1):
        if l.values[i+1] < l.values[i]:
            return False
    return True

if __name__ == '__main__':
    main()
    