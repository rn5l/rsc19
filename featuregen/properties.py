'''
Created on May 24, 2019

@author: malte
'''
from pathlib import Path

from config.globals import BASE_PATH
from domain.action_types import CLICK, IMAGE, INFO, DEALS, RATING, FILTER
from helper.df_ops import copy_features, reduce_mem_usage, apply
from helper.loader import load_hdfs, write_feather, load_feather
import numpy as np
import pandas as pd
import time
from featuregen.popularity import print_col_list


TEST_FOLDER = BASE_PATH + 'competition/'
META_FOLDER = BASE_PATH + 'preprocessed/'

ACTION_MAP = {}
ACTION_MAP['all'] = [CLICK,IMAGE,INFO,DEALS,RATING]
ACTION_MAP['view'] = [IMAGE,INFO,DEALS,RATING]
ACTION_MAP['click'] = [CLICK]

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    properties_features( TEST_FOLDER, META_FOLDER, log, examples, redo=True )

def properties_features(base_path, meta_path, log, examples, redo=False):
    
    name = 'properties_features'
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( meta_path, log, examples )
        examples = reduce_mem_usage(examples)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','prices','label'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list(cols)
        
    return examples

def create_features( meta_path, log, examples, latent_prefix='d2v', latent_size=16  ):
    
    tstart = time.time()
    print( 'create_features properties' )
    
    cols_pre = examples.columns.values

    #pure meta
    use_from_meta = ['properties_set']
    
    #settify
    def settify(x):
        if type(x) == list:
            return set(x)
        elif type(x) == float and not np.isnan(x):
            return set([x])
        else:
            return set()
    meta = load_hdfs(meta_path + 'meta_extended.hd5')
    meta['properties_set'] = meta['properties_code'].apply( settify)
    meta = meta[['item_id']+use_from_meta]
    
    examples = add_from_file( meta, examples, to=['impressions'] )
    log_tmp = log.drop_duplicates(['session_id','reference','action_type'], keep='last')
    log_tmp = add_from_file( meta, log_tmp, to=['item_id'] )
    
    examples['properties_set'] = examples['properties_set'].apply( lambda x: set() if type(x) == float else x )
    log_tmp['properties_set'] = log_tmp['properties_set'].apply( lambda x: set() if type(x) == float else x )
    
    print( log_tmp['properties_set'] )
    
    #collect
    examples = add_group_props( log_tmp, examples, group=['session_id'], key='session' )
    examples = add_group_filter( log_tmp, examples, group=['session_id'], key='session' )
    examples = add_current_filter( log_tmp, examples )
    del log_tmp
    
    #settify
    
    examples['current_filters'] = examples['current_filters'].apply( settify )
    
    del_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( examples['properties_session_intersect'] )
    #compare
    examples['properties_match_session_intersect'] = ratio( examples, 'properties_session_intersect', 'properties_set' )
    examples['filter_match_session'] = ratio( examples, 'filter_session_all', 'properties_set' )
    examples['filter_current_match_session'] = ratio( examples, 'current_filters', 'properties_set' )
    examples['properties_match_session_union'] = ratio( examples, 'properties_session_union', 'properties_set' )
    examples['filter_count'] = examples['current_filters'].apply( len )
    
    for col in del_cols:
        del examples[col]
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    #examples = fill_na(examples, new_cols, 0)
    
    print( 'create_features pop in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols

def ratio( examples, filter_col, props_col ):
    def calc( a, b ):
        try:
            if len(a) == 0:
                return 1
            return len(a & b) / len(a)
        except Exception as e:
            print( e )
            print( a )
            print( b )
            exit()
    return apply(examples, [filter_col, props_col], lambda row: calc(row[0], row[1]), verbose=100000 )

def ratio2( examples, filter_col, props_col ):
    def calc( a, b ):
        try:
            if len(a) == 0:
                return 1
            return len(a & b) / len(a)
        except Exception as e:
            print( e )
            print( a )
            print( b )
            exit()
    return examples[[filter_col, props_col]].apply( lambda row: calc(row[filter_col], row[props_col]), axis=1 )

def add_group_props( log, examples, mask=None, after=False, group=[], group_examples=None, key=None ):
    
    tstart = time.time()
    print( '\t add_group_props {}'.format(key) )
    
    base_key = 'properties_'
    if key is not None:
        base_key += key + '_'
    
    if group_examples is None:
        group_examples = group
    group_log = group
    
    if mask is None:
        mask = log.train > -1
    if after:
        mask = mask & (log.exclude == 1)
    else:
        mask = mask & (log.exclude == 0)
    
    mask = mask & ~log.reference.isnull()
    length = log['properties_set'].apply( len )
    mask = mask & (length > 0)
    
    grouped = log[mask].groupby( group_log )
    
    collect = pd.DataFrame()
    collect[base_key + 'union'] = grouped['properties_set'].apply( lambda x: set.union(*x) )
    collect[base_key + 'intersect'] = grouped['properties_set'].apply( lambda x: set.intersection(*x) )

    examples = examples.merge( collect, left_on=group_examples, right_index=True, how='left' )
    
    examples.ix[examples[base_key + 'union'].isnull(), base_key + 'union'] = [set()] * sum( examples[base_key + 'union'].isnull() )
    examples.ix[examples[base_key + 'intersect'].isnull(), base_key + 'intersect'] = [set()] * sum( examples[base_key + 'intersect'].isnull() )
    
    
    print( '\t add_group_props in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def add_group_filter( log, examples, mask=None, after=False, group=[], group_examples=None, key=None ):

    tstart = time.time()
    print( '\t add_group_filter {}'.format(key) )
    
    base_key = 'filter_'
    if key is not None:
        base_key += key + '_'
    
    if group_examples is None:
        group_examples = group
    group_log = group
    
    if mask is None:
        mask = log.train > -1
    if after:
        mask = mask & (log.exclude == 1)
    else:
        mask = mask & (log.exclude == 0)
    
    mask = mask & log.action_type.isin( [FILTER] )
    mask = mask & ~log.reference.isnull()
    grouped = log[mask].groupby( group_log )
    
    collect = pd.DataFrame()
    collect[base_key + 'all'] = grouped['reference'].apply( set )
       
    examples = examples.merge( collect, left_on=group_examples, right_index=True, how='left' )
    
    examples[base_key + 'all'].fillna({}, inplace=True)
    examples.ix[examples[base_key + 'all'].isnull(), base_key + 'all'] = [set()] * sum( examples[base_key + 'all'].isnull() )
    
    print( '\t add_group_filter in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def add_current_filter( log, examples ):

    tstart = time.time()
    print( '\t add_current_filter' )
    
    if not 'current_filters' in examples:
    
        mask = log.hidden == 1          
        collect = pd.DataFrame()
        collect['current_filters'] = log[mask].groupby( 'session_id' )['current_filters'].first()
           
        examples = examples.merge( collect, left_on='session_id', right_index=True, how='left' )
    
    print( '\t add_current_filter in {}s'.format( (time.time() - tstart) ) )
    
    return examples
    
    
def add_from_file( data, examples, col=['item_id'], to=None, filter=None, rename=None, index=True ):
    
    tstart = time.time()
    print( '\t add_from_file {}'.format(data) )
    
    keep = False
    if to is None:
        to = col
        keep = True
    
    if type(data) == str:
        toadd = pd.read_csv( data, index_col=0 if index else None )
    else:
        toadd = data
        
    if filter is not None:
        toadd = toadd[col+filter]
    
    if rename is not None:
        for i,c in enumerate(filter):
            toadd[rename[i]] = toadd[filter[i]]
            del toadd[c]
    
    copy = False
    if col[0] in examples.columns:
        copy = True
    
    examples = examples.merge( toadd, left_on=to, right_on=col, how='left' )
    
    if col[0]+'_y' in examples:
        examples[col[0]] = examples[col[0]+'_y']
        del examples[col[0]+'_y']
    
    if not keep:
        del examples[col[0]]
    
    print( '\t add_from_file in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def fill_na( examples, cols, value ):
    for c in cols:
        examples[c] = examples[c].fillna(value)
    return examples

if __name__ == '__main__':
    main()
    