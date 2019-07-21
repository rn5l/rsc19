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


TEST_FOLDER = BASE_PATH + 'competition/'

ACTION_MAP = {}
ACTION_MAP['all'] = [CLICK,IMAGE,INFO,DEALS,RATING]
ACTION_MAP['view'] = [IMAGE,INFO,DEALS,RATING]
ACTION_MAP['deal'] = [DEALS]
ACTION_MAP['rating'] = [RATING]
ACTION_MAP['image'] = [IMAGE]
ACTION_MAP['click'] = [CLICK]

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    log = clean_test(log)
    pop_features( TEST_FOLDER, log, examples, hidden=False, redo=True )

def clean_test(log):
    mask = (log.train == 0) & (log.hidden == 1)
    log.ix[mask,'item_id'] = np.nan
    return log

def pop_features(base_path, log, examples, hidden=False, min_pop=None, train_only=False, redo=False):
    
    name = 'pop_features'
    if hidden:
        name += '_hidden'
    if min_pop is not None:
        name += '_mp' + str(min_pop)
    if train_only: 
        name += '_trainonly'
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( log, examples, hidden=hidden, min_pop=min_pop, train_only=train_only )
        examples = reduce_mem_usage(examples)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','prices','label'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list(cols)
        
    return examples

def create_features( log, examples, hidden=False, min_pop=0, train_only=False ):
    
    tstart = time.time()
    print( 'create_features pop' )
    
    cols_pre = examples.columns.values
    
    mask_log = log.train > -1
    mask_examples = examples.train > -1
    if train_only:
        mask_log = log.train == 1
        mask_examples = examples.train == 1
        
    if not hidden:
        mask_log = mask_log & (log.hidden == 0)
        
    examples = pop_by_group(log, examples, mask_log, mask_examples, group=[], key=None, hidden=hidden, min_pop=min_pop)
    examples = pop_by_group(log, examples, mask_log, mask_examples, group=['platform'], key='platform', hidden=hidden, min_pop=min_pop)
    #examples = pop_by_group(log, examples, mask_log, mask_examples, group=['device'], key='device', hidden=hidden, min_pop=min_pop)
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( 'create_features pop in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols
    
def pop_by_group( log, examples, mask_log, mask_examples, group=[], key=None, hidden=False, min_pop=None ):
    
    tstart = time.time()
    print( '\t pop_by_group {}'.format(key) )
    
    base_key = 'pop_'
    if not key is None:
        base_key += key + '_'
        
    group_examples = ['impressions'] + group
    group_log = ['item_id'] + group
    
    pop = pd.DataFrame()
    pop[base_key + 'impressions'] = examples[mask_examples].groupby( group_examples ).size()
    examples = examples.merge( pop, left_on=group_examples, right_index=True, how='left' )
    examples[base_key + 'impressions'].fillna( 0, inplace = True )
    
    print( '\t\t impressions in {}s'.format( (time.time() - tstart) ) )
    
    for name,actions in ACTION_MAP.items():
        
        print( '\t\t start {} in {}s'.format( name, (time.time() - tstart) ) )
        
        pop = pd.DataFrame()
        pop[base_key + name] = log[mask_log & log.action_type.isin(actions)].groupby( group_log ).size()            
        if min_pop is not None: 
            pop = pop[pop[base_key + name] > min_pop]
        
        print( '\t\t collect {} in {}s'.format( name, (time.time() - tstart) ) )
        
        examples = examples.merge( pop, left_on=group_examples, right_index=True, how='left' )
        del pop
        
        print( '\t\t merge {} in {}s'.format( name, (time.time() - tstart) ) )
        
        if hidden and CLICK in actions: #reduce by one on labels
            #examples.ix[ examples.label == 1, base_key + name] -= 1
            print( '\t\t corrected {} in {}s'.format( name, (time.time() - tstart) ) )
        
        examples[base_key + name] = examples[base_key + name].fillna(0)
        examples[base_key + name+'_per_impression'] = examples[base_key + name] / ( examples[base_key + 'impressions'] )
        examples[base_key + name+'_per_impression'] = examples[base_key + name+'_per_impression'].fillna(0)
            
    examples[base_key+'click_per_view'] = examples[base_key+'click'] / ( examples[base_key+'view'] + 20 )
    examples[base_key+'click_per_deal'] = examples[base_key+'deal'] / ( examples[base_key+'view'] + 20 )
    examples[base_key+'click_per_image'] = examples[base_key+'image'] / ( examples[base_key+'view'] + 20 )
    examples[base_key+'click_per_rating'] = examples[base_key+'rating'] / ( examples[base_key+'view'] + 20 )
    
    pop = pd.DataFrame()
    pop[base_key+'view_sessions'] = log[mask_log & log.action_type.isin(ACTION_MAP['view'])].drop_duplicates( ['session_id']+group_log, keep='last').groupby( group_log ).size()
    examples = examples.merge( pop, how='left', left_on=group_examples, right_index=True )
    
    pop = pd.DataFrame()
    pop[base_key+'click_sessions'] = log[mask_log & log.action_type.isin(ACTION_MAP['click'])].drop_duplicates( ['session_id']+group_log, keep='last').groupby( group_log ).size()
    examples = examples.merge( pop, how='left', left_on=group_examples, right_index=True )
    
    examples[base_key+'click_per_view_sessions'] = examples[base_key+'click_sessions'] / examples[base_key+'view_sessions']
    examples[base_key+'click_per_view_sessions'] = examples[base_key+'click_per_view_sessions'].replace( [np.inf, -np.inf], np.nan ).fillna( 0 )
    
    del examples[base_key+'click_sessions'], examples[base_key+'view_sessions']
    
    print( '\t pop_by_group in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def print_col_list( cols ):
    
    print( 'POP_FEATURES = [' )
    for name in cols:
        print( "    '"+name+"'," )
    print( ']' )

if __name__ == '__main__':
    main()
    