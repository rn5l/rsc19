'''
Created on Mar 19, 2019

@author: malte
'''

import gc
from pathlib import Path

from config.globals import BASE_PATH
from domain.features import FEATURES
from evaluate import evaluate
from featuregen.create_set import create_set
from helper.df_ops import train_test_cv
from helper.loader import load_feather, ensure_dir
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
LTR = True
SHUFFLE = False
SPLITS = 5
STOPPING = 500
MAX_EPOCHS = 10000

#KEYS
DSKEY = 'dataset'
ALGKEY = 'lgbmcv_{}_{}_split{}-es{}_dart0.1'.format( 'ltr' if LTR else 'bin', 'shfl' if SHUFFLE else 'noshfl', SPLITS, STOPPING )
STACK = False

def main():
    
    train = create_set( base_path=BASE_PATH + SET, conf=CONF, key=DSKEY, redo=False )
    test = train.query('train == 0')
    
    test_file_key = DSKEY
    ensure_dir(BASE_PATH + SET + 'tmp/')
    test_file = BASE_PATH + SET + 'tmp/' + test_file_key + '_test.fthr'
    
    if not Path(test_file).is_file():
        test = test.reset_index(drop=True)
        test.to_feather( test_file )
    
    test_len = len(test)
    del test
    gc.collect()
    
    train.query('train == 1', inplace=True)
    
    X = train[ FEATURES + ['session_id'] ]
    y = train[ 'label' ]
    
    del train
    gc.collect()
    
    score = np.zeros( (SPLITS, test_len) )
    i = 0
    
    for train_idx, val_idx in train_test_cv( X, y, splits=SPLITS, shuffle=SHUFFLE ):
        
        X_train = X.loc[train_idx]
        X_valid = X.loc[val_idx]
        y_train = y.loc[train_idx]
        y_valid = y.loc[val_idx]
        
        if LTR:
            q_train =  X_train.groupby( ['session_id'] ).size().values.astype(np.float32)
            q_valid = X_valid.groupby( ['session_id'] ).size().values.astype(np.float32)
            xtrain = X_train[FEATURES].values.astype(np.float32)
            ytrain = y_train.values.astype(np.float32)
            del X_train, y_train
            gc.collect()
            d_train = lgbm.Dataset( xtrain, label=ytrain, group=q_train, feature_name=FEATURES)#, categorical_feature=CAT_FEATURES )
            del q_train
            gc.collect()
            xval = X_valid[FEATURES].values.astype(np.float32)
            yval = y_valid.values.astype(np.float32)
            del X_valid, y_valid
            gc.collect()
            d_valid = lgbm.Dataset( xval, label=yval, group=q_valid, feature_name=FEATURES)#, categorical_feature=CAT_FEATURES )
            del q_valid
            gc.collect()
        else:
            d_train = lgbm.Dataset( X_train[FEATURES], label=y_train, feature_name=FEATURES )#+ ['session_id'])#, categorical_feature=CAT_FEATURES )
            d_valid = lgbm.Dataset( X_valid[FEATURES], label=y_valid, feature_name=FEATURES )#+ ['session_id'])#, categorical_feature=CAT_FEATURES )
        
        watchlist = [d_train, d_valid]
    
        params = {}
        params['boosting'] = 'dart'
        params['learning_rate'] = 0.1
        if LTR:
            params['application'] = 'lambdarank'
            params['metric'] = 'ndcg'
            params['eval_at'] = '30'
        else:
            params['application'] = 'binary'
            params['metric'] = 'binary_logloss'
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
        model.save_model( BASE_PATH + SET + 'lgbm/'+ALGKEY+'.'+str(i)+'.txt' , num_iteration=model.best_iteration, )
        
        del params, watchlist, d_train, d_valid,  evals_result
        gc.collect()
    
        test = load_feather(test_file)
        
        X_test = test[ FEATURES ].values.astype(np.float32)
        
        y_test = model.predict(X_test, num_iteration=model.best_iteration )
        score[i] = y_test
        i+=1
        
        del y_test, model, X_test, test
        gc.collect()
    
    test = load_feather(test_file)
    
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
    