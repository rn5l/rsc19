'''
Created on May 24, 2019

@author: malte
'''
from pathlib import Path

from config.globals import BASE_PATH
from domain.action_types import CLICK, IMAGE, INFO, DEALS, RATING
from helper.df_ops import copy_features, expand, reduce_mem_usage
from helper.loader import load_hdfs, write_feather, load_feather, ensure_dir
import numpy as np
import pandas as pd
import time
import gc
from featuregen.popularity import print_col_list


TEST_FOLDER = BASE_PATH + 'competition/'

ACTION_MAP = {}
ACTION_MAP['all'] = [CLICK,IMAGE,INFO,DEALS,RATING]
ACTION_MAP['view'] = [IMAGE,INFO,DEALS,RATING]
ACTION_MAP['click'] = [CLICK]

MIN_OCCURENCES = None
FILLNA_MEAN = False
HIDDEN = False

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    log = clean_test(log)
    price_features( TEST_FOLDER, log, examples, min_occurences=MIN_OCCURENCES, hidden=HIDDEN, fillna_mean=FILLNA_MEAN, redo=True )

def clean_test(log):
    mask = (log.train == 0) & (log.hidden == 1)
    log.ix[mask,'item_id'] = np.nan
    log.ix[mask,'session_price'] = np.nan
    return log

def price_features(base_path, log, examples, min_occurences=None, hidden=False, train_only=False, fillna_mean=False, redo=False):
    
    name = 'price_features'
    if train_only: 
        name += '_trainonly'
    if hidden: 
        name += '_hidden'
    if min_occurences is not None: 
        name += '_min' + str(min_occurences)
    if fillna_mean: 
        name += '_fillmean'
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( base_path, log, examples, min_occurences=min_occurences, hidden=hidden, train_only=train_only, fillna_mean=fillna_mean )
        examples = reduce_mem_usage(examples, cols=cols)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','prices','city','platform','label'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list(cols)
    
    return examples
    
def create_features( base_path, log, examples, min_occurences=None, hidden=False, train_only=False, fillna_mean=False ):
    
    tstart = time.time()
    print( 'create_features price' )
    
    cols_pre = examples.columns.values
    
    mask_log = log.hidden > -1
    mask_examples = examples.train > -1
    if train_only:
        mask_log = mask_log & (log.train == 1)
        mask_examples = mask_examples & (examples.train == 1)
    if not hidden:
        mask_log = mask_log & (log.hidden < 1)
        
    clicks = log[log.action_type==CLICK][['train','session_id','prices','impressions','city','platform']].copy()
    clicks = expand(clicks, ['impressions','prices'])
    clicks = clicks.drop_duplicates(['session_id','impressions'], keep='last')
    mask_clicks = clicks.train > -1 if not train_only else clicks.train == 1
    
    ensure_dir( base_path + 'tmp/' )
    clicks.groupby('city').prices.mean().to_csv(base_path + 'tmp/' + 'city_price.csv')
    
    examples = price_by_group_imp(examples, clicks, mask_clicks, group=['impressions'], key='item', min_occurences=min_occurences, fillna_mean=fillna_mean)
    examples = price_by_group_imp(examples, clicks, mask_clicks, group=['city'], key='city', min_occurences=min_occurences, fillna_mean=fillna_mean)
    examples = price_by_group_imp(examples, clicks, mask_clicks, group=['platform'], key='platform', min_occurences=min_occurences, fillna_mean=fillna_mean)
    examples = price_by_group_imp(examples, clicks, mask_clicks, group=['city','platform'], key='city_platform', min_occurences=min_occurences, fillna_mean=fillna_mean)
    examples = price_by_group_imp(examples, examples, mask_examples, group=['session_id'], key='list', min_occurences=min_occurences, fillna_mean=fillna_mean)
    del clicks
    gc.collect()
    
#     clickprice = pd.DataFrame()
#     clickprice['prices_click'] = examples[(examples.train==1) & (examples.label==1)].groupby( 'session_id' ).prices.min()
#     examples = examples.merge( clickprice, left_on='session_id', right_index=True, how='left' )
#     del clickprice
    
    print(sum(mask_log))
    examples = price_by_group_action(log, examples, mask_log, group=['item_id'], group_examples=['impressions'], key='item', hidden=hidden, min_occurences=min_occurences, fillna_mean=fillna_mean)  
    examples = price_by_group_action(log, examples, mask_log, group=['platform'], key='platform', hidden=hidden, min_occurences=min_occurences, fillna_mean=fillna_mean)
    #examples = price_by_group_action(log, examples, mask_log, group=['device'], key='device', hidden=hidden, min_occurences=min_occurences, fillna_mean=fillna_mean)
    examples = price_by_group_action(log, examples, mask_log, group=['city'], key='city', hidden=hidden, min_occurences=min_occurences, fillna_mean=fillna_mean)
    examples = price_by_group_action(log, examples, mask_log, group=['city','platform'], key='city_platform', hidden=hidden, min_occurences=min_occurences, fillna_mean=fillna_mean)
     
#     del examples['prices_click']
    
    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( 'create_features price in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols

def price_by_group_imp( examples, agg_on, mask_agg, group=[], key=None, min_occurences=None, fillna_mean=False ):
    
    tstart = time.time()
    print( '\t price_by_group_imp {}'.format(key) )
    
    base_key = 'price_'
    if not key is None:
        base_key += key + '_'
    base_key += 'impressions_'
    
    grouped = agg_on[mask_agg].groupby( group )
    
    prices = pd.DataFrame()
    prices[base_key + 'min'] = grouped.prices.min()
    prices[base_key + 'max'] = grouped.prices.max()
    prices[base_key + 'mean'] = grouped.prices.mean()
    ##prices[base_key + 'std'] = grouped.prices.std()
#     prices[base_key + 'count'] = grouped.prices.size()
#     
#     if min_occurences is not None:
#         prices = prices[prices[base_key + 'count'] >= min_occurences]
    
    #del prices[base_key + 'count']
    
    examples = examples.merge( prices, left_on=group, right_index=True, how='left' )
    del prices
    
    examples = norm_and_dist( examples, base_key )
    examples = fillna( examples, base_key, fillna_mean=fillna_mean )
    
    print( '\t price_by_group_imp in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def price_by_group_action( log, examples, mask_log, group=[], group_examples=None, key=None, hidden=False, min_occurences=None, fillna_mean=False ):
    
    tstart = time.time()
    print( '\t price_by_group_action {}'.format(key) )
    
    base_key = 'price_'
    if not key is None:
        base_key += key + '_'
    
    if group_examples is None:
        group_examples = group
    group_log = group
    
#     correct = examples[examples.label == 1][ ['session_id'] + group_examples ].copy()
#     correct['correct'] = 1
#     
#     examples = examples.merge( correct, on=['session_id'] + group_examples, how='left' )
#     del correct
    
    for name, actions in ACTION_MAP.items():
        
        print( '\t\t start {} in {}s'.format( name, (time.time() - tstart) ) )
        
        base_key_name = base_key + name + '_'
        
        mask = mask_log & log.action_type.isin(actions) & ~log.price_session.isnull()
        grouped = log[mask].drop_duplicates(group+['session_id','action_type','price_session'], keep='last').groupby( group_log )
        #print( log[mask & (log.city == 4726)] )
        
        print( '\t\t grouped {} in {}s'.format( name, (time.time() - tstart) ) )
        
        pop = pd.DataFrame()
         
        pop[base_key_name + 'min' ] = grouped['price_session'].min()         
        pop[base_key_name + 'max' ] = grouped['price_session'].max()          
        pop[base_key_name + 'mean' ] = grouped['price_session'].mean()
        #pop[base_key_name + 'count' ] = grouped['price_session'].size()
        #pop[base_key_name + 'tmp' ] = grouped['price_session'].apply(list)
        #pop[base_key_name + 'std' ] = log[mask_log & log.action_type.isin(actions)].groupby( group_log ).std()
                
#         if min_occurences is not None:
#             pop = pop[pop[base_key_name + 'count' ] >= min_occurences]
        
#         if pop[base_key_name + 'count' ].min() >= 5:
#             hidden = False
        
        print( '\t\t collect {} in {}s'.format( name, (time.time() - tstart) ) )
        
        examples = examples.merge( pop, left_on=group_examples, right_index=True, how='left' )
        #del pop
        
        #print(examples[[base_key_name + 'min', base_key_name + 'min2', base_key_name + 'max', base_key_name + 'max2']])
        
#         examples[ base_key_name + 'mean'] = examples[base_key_name + 'sum' ] / examples[base_key_name + 'count' ]
        
        print( '\t\t merged {} in {}s'.format( name, (time.time() - tstart) ) )
        
#         if hidden and CLICK in actions: #reduce by price on labels
#             
#             correct = ( examples.correct == 1 ) & ~examples['prices_click'].isnull()
#             
#             examples.ix[correct, base_key_name + 'mean'] = ( examples.ix[correct, base_key_name + 'sum' ] - examples['prices_click'] ) / ( examples.ix[correct, base_key_name + 'count' ] - 1 )
#             
#             if sum( examples[ base_key_name + 'mean'] == 0 ) > 0:
#                 print( examples[examples[ base_key_name + 'mean'] == 0][['session_id','prices_click', 'city', base_key_name + 'tmp']] )
#                 #print( pop.ix[4726] )
#                 exit()
#                 
#             ismin = correct & (examples['prices_click'] == examples[ base_key_name + 'min' ])
#             ismax = correct & (examples['prices_click'] == examples[ base_key_name + 'max' ])
#             single = correct & (examples[ base_key_name + 'count' ] == 1)
#             examples.ix[ ismin, base_key_name + 'min'] = examples.ix[ ismin, base_key_name + 'min2']
#             examples.ix[ ismax, base_key_name + 'max'] = examples.ix[ ismax, base_key_name + 'max2']
#             examples.ix[ single, base_key_name + 'mean'] = np.nan
#             
#             print('ismin ',sum(ismin) )
#             print('ismax ',sum(ismax) )
#             print('single ',sum(single) )
#             print('swap ',sum(examples[base_key_name + 'min'] > examples[base_key_name + 'max']) )
#             
#             print( '\t\t corrected {} in {}s'.format( name, (time.time() - tstart) ) )
        
        
        same = examples[ base_key_name + 'min' ] == examples[ base_key_name + 'max' ]
        examples.ix[ same, base_key_name + 'min'] = np.nan
        examples.ix[ same, base_key_name + 'max'] = np.nan
        
#         del examples[ base_key_name + 'min2' ], examples[ base_key_name + 'max2' ]
#         del examples[ base_key_name + 'sum' ]#, examples[ base_key_name + 'count' ]
        
        examples = norm_and_dist(examples, base_key_name)
        examples = fillna( examples, base_key_name, fillna_mean=fillna_mean )
    
#     del examples['correct']
    
    print( '\t price_by_group_action in {}s'.format( (time.time() - tstart) ) )
    
    return examples

def norm_and_dist( examples, base_key ):
    
    examples[base_key + 'minmax'] = ( examples['prices'] - examples[base_key + 'min'] ) / ( examples[base_key + 'max'] - examples[base_key + 'min'] )
    examples[base_key + 'permean'] = examples['prices'] / examples[base_key + 'mean']
    examples[base_key + 'dist'] = examples[base_key + 'mean'] - examples['prices']
    #examples[base_key + 'range'] = examples[base_key + 'max'] - examples[base_key + 'min']
    
    examples[base_key + 'minmax_dist'] = ( examples[base_key + 'mean'] - examples[base_key + 'min'] ) / ( examples[base_key + 'max'] - examples[base_key + 'min'] )
    examples[base_key + 'minmax_dist'] = examples[base_key + 'minmax_dist'] - examples[base_key + 'minmax']
    
    del examples[base_key + 'min'], examples[base_key + 'max'], examples[base_key + 'minmax']
    
    return examples

def fillna( examples, base_key, fillna_mean=False ):
    
    if not fillna_mean:
        examples[base_key + 'dist'].fillna(0, inplace=True)
        examples[base_key + 'mean'].fillna(-1, inplace=True)
        #examples[base_key + 'minmax'].fillna(-1, inplace=True)
        examples[base_key + 'minmax_dist'].fillna(0, inplace=True)
        examples[base_key + 'permean'].fillna(-1, inplace=True)
        #examples[base_key + 'min'].fillna( -1, inplace=True )
        #examples[base_key + 'max'].fillna( -1, inplace=True )
        #examples[base_key + 'range'].fillna( -1, inplace=True )
        #examples[base_key + 'count'].fillna( 0, inplace=True )
        pass
    else:
        examples[base_key + 'mean'].fillna( examples[base_key + 'mean'].mean(), inplace=True )
        examples[base_key + 'dist'] = examples[base_key + 'mean'] - examples['prices']
        #examples[base_key + 'minmax'].fillna( examples[base_key + 'minmax'].mean(), inplace=True )
        examples[base_key + 'minmax_dist'].fillna( examples[base_key + 'minmax_dist'].mean(), inplace=True )
        examples[base_key + 'permean'].fillna( examples[base_key + 'permean'].mean(), inplace=True )
        #examples[base_key + 'min'].fillna( examples[base_key + 'min'].mean(), inplace=True )
        #examples[base_key + 'max'].fillna( examples[base_key + 'max'].mean(), inplace=True )
        #examples[base_key + 'range'].fillna( examples[base_key + 'range'].mean(), inplace=True )
        #examples[base_key + 'count'].fillna( 0, inplace=True )
        
    return examples

# def norm_and_dist_old( examples, base_key ):
#     
#     examples[base_key + 'norm'] = ( examples['prices'] - examples[base_key + 'min'] ) / ( examples[base_key + 'max'] - examples[base_key + 'min'] )
#     examples[base_key + 'mean_norm'] = ( examples[base_key + 'mean'] - examples[base_key + 'min'] ) / ( examples[base_key + 'max'] - examples[base_key + 'min'] )
#     
#     inf = np.isinf(examples[base_key + 'norm'])
#     examples.ix[inf, base_key + 'norm'] = examples['prices'] / examples[base_key + 'max']
#     examples.ix[inf, base_key + 'mean_norm'] = examples[base_key + 'mean'] / examples[base_key + 'max']
#     
#     examples[base_key + 'dist'] = examples[base_key + 'mean'] - examples['prices']
#     examples[base_key + 'dist_abs'] = examples[base_key + 'dist'].apply( abs )
#     
#     examples[base_key + 'dist'] = examples[base_key + 'dist'].fillna( -1 - examples['prices'] )
#     examples[base_key + 'dist_abs'] = examples[base_key + 'dist_abs'].fillna( -1 )
#     examples[base_key + 'mean'] = examples[base_key + 'mean'].fillna( -1 )
#     
#     examples[base_key + 'normdist'] = examples[base_key + 'mean_norm'] - examples[base_key + 'norm']
#     examples[base_key + 'normdist_abs'] = examples[base_key + 'normdist'].apply( abs )
#     
#     examples[base_key + 'normdist'] = examples[base_key + 'normdist'].fillna( examples[base_key + 'normdist'].mean() )
#     examples[base_key + 'normdist_abs'] = examples[base_key + 'normdist_abs'].fillna( -1 )
#     examples[base_key + 'norm'] = examples[base_key + 'norm'].fillna(-1)
#     examples[base_key + 'mean_norm'] = examples[base_key + 'mean_norm'].fillna(-1)
#     
#     del examples[base_key + 'min'], examples[base_key + 'max'], examples[base_key + 'mean_norm']
#     
#     return examples

if __name__ == '__main__':
    main()
    