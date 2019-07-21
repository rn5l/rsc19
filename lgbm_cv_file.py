'''
Created on Mar 19, 2019

@author: malte
'''

import gc
from pathlib import Path
import time

from config.globals import BASE_PATH
from domain.features import FEATURES
from evaluate import evaluate
from featuregen.create_set import create_set
from helper.df_ops import train_test_cv
from helper.loader import ensure_dir
import lightgbm as lgbm
import numpy as np
import pandas as pd


RAW = 'raw/'
SET = 'competition/'

CONF = {
    'train_only': False,
    
    'pop_hidden': False,
    'path_pop': BASE_PATH + SET,
    'min_pop': None,
    
    'price_hidden': False,
    'path_price': BASE_PATH + SET,
    'min_occurences': None,
    'fillna_mean': False,
    
    'path_session': BASE_PATH + SET,
    'path_poi': BASE_PATH + SET,
        
    'path_crawl': BASE_PATH + 'crawled/',
    
    'path_meta': BASE_PATH + 'preprocessed/',
    'meta_latent': 'd2v',
    
    'path_latent': BASE_PATH + 'competition/',
}

#ALGO 
SHUFFLE = False
SPLITS = 5
STOPPING = 500
MAX_EPOCHS = 10000

#KEYS
DSKEY = 'dataset'
ALGKEY = 'lgbmcv_{}_{}_split{}-es{}_dart0.1'.format( 'ltr', 'shfl' if SHUFFLE else 'noshfl', SPLITS, STOPPING )
#STACK = False

def main():
    
    train = create_set( base_path=BASE_PATH + SET, conf=CONF, key=DSKEY, redo=False )
    len_test = ( len(train[train.train == 0]) )
    train.query('train == 1', inplace=True)
    
    indices = train_test_cv( train, train.label, splits=SPLITS, shuffle=SHUFFLE )
    
    del train
    gc.collect()
    
    score = np.zeros( (SPLITS, len_test) )
    i = 0
    
    for train_idx, val_idx in indices:
        
        dataset_file_key = DSKEY + 'SPLIT' + str(i) + 'of' + str(SPLITS) + 'SHFL' + str(SHUFFLE)
        ensure_dir( BASE_PATH + SET + 'tmp/' )
        file = BASE_PATH + SET + 'tmp/' + dataset_file_key + '_train.lgbm'
        
        if not Path( file ).is_file():
    
            tstart = time.time()
            
            train = create_set( base_path=BASE_PATH + SET, conf=CONF, key=DSKEY, redo=False )
            
            print( 'loaded in {}'.format( (time.time() - tstart) ) )
            tstart = time.time()

            train.query( 'train == 1', inplace=True )
                
            file = BASE_PATH + SET + 'tmp/'+dataset_file_key+'_train.csv'
            #dump_svmlight_file( X_train[KEEP_FEATURES], y_train, file, zero_based=True, multilabel=False )
            train.loc[ train_idx, ['label']+FEATURES].to_csv( file, index=False, header=False )
            #num_cols = len(X_train.columns)
            q_train =  train.loc[train_idx].groupby('session_id').size().values
            
            d_train = lgbm.Dataset( file )#, categorical_feature=CAT_FEATURES )
            d_train.set_group( q_train )
            d_train.set_label( train.loc[ train_idx, 'label'] )
            d_train.set_feature_name( FEATURES )
            
            #del q_train
            gc.collect()
            
            
            file = BASE_PATH + SET + 'tmp/'+dataset_file_key+'_valid.csv'
            #dump_svmlight_file( X_valid[KEEP_FEATURES], y_valid, file, zero_based=True, multilabel=False )
            train.loc[ val_idx, ['label']+FEATURES].to_csv( file, index=False, header=False)
            
            q_valid =  train.loc[ val_idx ].groupby('session_id').size().values
            y_val = train.loc[ val_idx, 'label' ]
            del train
            gc.collect()
            
            d_valid = d_train.create_valid( file )#, categorical_feature=CAT_FEATURES )
            file = BASE_PATH + SET + 'tmp/'+dataset_file_key+'_valid.lgbm'
            d_valid.set_group( q_valid )
            d_valid.set_label( y_val  )
            d_valid.set_feature_name( FEATURES )
            d_valid.save_binary(file)
            file = BASE_PATH + SET + 'tmp/'+dataset_file_key+'_train.lgbm'
            d_train.save_binary(file)
            #del q_valid, d_valid, d_train
            gc.collect()
            
            file = BASE_PATH + SET + 'tmp/'+dataset_file_key+'_train.lgbm'
            d_train = lgbm.Dataset( file )
            file = BASE_PATH + SET + 'tmp/'+dataset_file_key+'_valid.lgbm'
            d_valid = lgbm.Dataset( file )
            
        else:
            
            tstart = time.time()
            
            print('load binary lgbm sets')
            file = BASE_PATH + SET + 'tmp/'+dataset_file_key+'_train.lgbm'
            d_train = lgbm.Dataset( file )
            file = BASE_PATH + SET + 'tmp/'+dataset_file_key+'_valid.lgbm'
            d_valid = lgbm.Dataset( file )
            
            print( 'loaded sets in {}'.format( (time.time() - tstart) ) )
        
        watchlist = [d_train, d_valid]
    
        params = {}
        params['boosting'] = 'dart'
        params['learning_rate'] = 0.1
        params['application'] = 'lambdarank'
        params['metric'] = 'ndcg'
        params['eval_at'] = '30'
        #params['max_depth'] = -1
        #params['num_leaves'] = 64
        #params['max_bin'] = 512
        params['feature_fraction'] = 0.5
        params['bagging_fraction'] = 0.5
        #params['min_data_in_leaf'] = 20
        #params['verbosity'] = 0
        
        evals_result = {}
        model = lgbm.train( params, train_set=d_train, num_boost_round=MAX_EPOCHS, valid_sets=watchlist, early_stopping_rounds=STOPPING, evals_result=evals_result, verbose_eval=10 )
        
        ensure_dir( BASE_PATH + SET + 'lgbm/' )
        model.save_model( BASE_PATH + SET + 'lgbm/' + ALGKEY + '.' + str(i) + '.txt' , num_iteration=model.best_iteration )
        
        del params, watchlist, d_train, d_valid,  evals_result
        gc.collect()
        
        
        test = create_set( base_path=BASE_PATH + SET, conf=CONF, key=DSKEY, redo=False )
        test.query('train == 0', inplace=True)
        X_test = test[ FEATURES ].values.astype(np.float32)
        del test
        gc.collect()
        
        y_test = model.predict(X_test, num_iteration=model.best_iteration )
        score[i] = y_test
        i+=1
        
        del y_test, model, X_test
        gc.collect()
    
    
    test = create_set( base_path=BASE_PATH + SET, conf=CONF, key=DSKEY, redo=False )
    test.query('train == 0', inplace=True)
    
    test['prob_norm'] = 0
    test['prob_direct'] = 0
    for i in range(SPLITS):
        test['prob_direct_'+str(i)] = score[i]
        test['prob_norm'+str(i)] = ( test['prob_direct_'+str(i)] - test['prob_direct_'+str(i)].min() ) / ( test['prob_direct_'+str(i)].max() - test['prob_direct_'+str(i)].min() ) 
        test['prob_direct'] += test['prob_direct_'+str(i)]
        test['prob_norm'] += test['prob_norm'+str(i)]
        
    test['prob_norm'] = test['prob_norm'] / SPLITS
    test['prob_direct'] = test['prob_direct'] / SPLITS
    
    #truth = pd.read_csv( self.folder + 'truth.csv' )
    #truth['label2'] = 1
    #test = test.merge( truth[['session_id','reference','label2']], left_on=['session_id','impressions'], right_on=['session_id','reference'], how='left' )
    #test['label'] =  test['label2'].fillna(0)
    #del test['label2']
    
    test = test.sort_values(['session_id','prob_norm'], ascending=False)
    #test.to_csv( BASE_PATH + SET + 'test_debugcv.csv' )
    
    solution = pd.DataFrame()
    solution['recommendations'] = test.groupby( 'session_id' ).impressions.apply( list )
    solution['confidences'] = test.groupby( 'session_id' ).prob_norm.apply( list )
    solution.reset_index(drop=True)
    solution = solution.merge( test[['session_id', 'user_id', 'timestamp', 'step']].drop_duplicates(keep='last'), on='session_id', how='inner' )    
    solution.to_csv( BASE_PATH + '/' + SET + '/solution_' + ALGKEY + '_norm.csv' )
    
    test = test.sort_values(['session_id','prob_direct'], ascending=False)
    solution = pd.DataFrame()
    solution['recommendations'] = test.groupby( 'session_id' ).impressions.apply( list )
    solution['confidences'] = test.groupby( 'session_id' ).prob_direct.apply( list )
    solution.reset_index(drop=True)
    solution = solution.merge( test[['session_id', 'user_id', 'timestamp', 'step']].drop_duplicates(keep='last'), on='session_id', how='inner' )    
    solution.to_csv( BASE_PATH + '/' + SET + '/solution_' + ALGKEY + '_direct.csv' )
    
    result = evaluate( solution, base=BASE_PATH, dataset=SET )
    print( result.T )

if __name__ == '__main__':
    main()
    