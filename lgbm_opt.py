'''
Created on Mar 19, 2019

@author: malte
'''

import gc
import pickle

from hyperopt import tpe, hp
from hyperopt.base import Trials
from hyperopt.fmin import fmin

from config.globals import BASE_PATH
from domain.features import FEATURES
from evaluate import evaluate
from featuregen.create_set import create_set
from helper.df_ops import train_test_split_idx
import lightgbm as lgbm
import numpy as np
import pandas as pd

#PATH
RAW = 'raw/'
SET = 'sample/'

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
        
    'path_crawl': BASE_PATH + 'crawled/',
    'path_poi': BASE_PATH + SET,
    
    'path_meta': BASE_PATH + 'preprocessed/',
    'meta_latent': 'd2v',
    
    'path_latent': BASE_PATH + 'competition/',
}

#KEYS
DSKEY = 'dataset'
TRAILKEY = 'trails-lgbm'

def objective( params ):
        
        train = create_set( base_path=BASE_PATH + SET, conf=CONF, key=DSKEY, redo=False )
            
        test = train.query('train == 0')
        train.query('train == 1', inplace=True)
    
        X = train[ FEATURES + ['session_id'] ]
        y = train[ 'label' ]
        
        del train
        gc.collect()
        
        X_train, X_valid = train_test_split_idx( X, y, test_size=0.1, shuffle=params['shuffle'] )
        print( 'shuffled sample ',params['shuffle'] )
        
        if params['ltr']:
            params['application'] = 'lambdarank'
            params['metric'] = 'ndcg'
            params['eval_at'] = '30'
        else:
            params['application'] = 'binary'
            params['metric'] = 'binary_logloss'
            
        if params['ltr']:
            q_train =  X.loc[X_train].groupby( ['session_id'] ).size().values.astype(np.float32)
            q_valid = X.loc[X_valid].groupby( ['session_id'] ).size().values.astype(np.float32)
            xtrain = X.loc[X_train][FEATURES].values.astype(np.float32)
            ytrain = y.loc[X_train].values.astype(np.float32)
            del X_train
            gc.collect()
            d_train = lgbm.Dataset( xtrain, label=ytrain, group=q_train, feature_name=FEATURES)#, categorical_feature=CAT_FEATURES )
            del q_train, xtrain, ytrain
            gc.collect()
            xval = X.loc[X_valid][FEATURES].values.astype(np.float32)
            yval = y.loc[X_valid].values.astype(np.float32)
            del X_valid
            gc.collect()
            d_valid = lgbm.Dataset( xval, label=yval, group=q_valid, feature_name=FEATURES)#, categorical_feature=CAT_FEATURES )
            del q_valid, xval, yval
            gc.collect()
        else:
            xtrain = X.loc[X_train][FEATURES].values.astype(np.float32)
            ytrain = y.loc[X_train].values.astype(np.float32)
            d_train = lgbm.Dataset( xtrain, label=ytrain, feature_name=FEATURES )#+ ['session_id'])#, categorical_feature=CAT_FEATURES )
            del xtrain, xtrain, X_train
            gc.collect()
            
            xval = X[X_valid][FEATURES].values.astype(np.float32)
            yval = y[X_valid].values.astype(np.float32)
            d_valid = lgbm.Dataset( xval, label=yval, feature_name=FEATURES )#+ ['session_id'])#, categorical_feature=CAT_FEATURES )
            del xval, yval, X_valid
            gc.collect()
        watchlist = [d_train, d_valid]
        
        evals_result = {}
        model = lgbm.train( params, train_set=d_train, num_boost_round=10000, valid_sets=watchlist, early_stopping_rounds=int(params['early_stopping']), evals_result=evals_result, verbose_eval=10 )
        
        X_test = test[ FEATURES ]
        y_test = model.predict(X_test, num_iteration=model.best_iteration )
        
        test['prob'] = y_test
        test = test.sort_values(['session_id','prob'], ascending=False)
        solution = pd.DataFrame()
        solution['recommendations'] = test.groupby( 'session_id' ).impressions.apply( list )
        solution['confidences'] = test.groupby( 'session_id' ).prob.apply( list )
        solution.reset_index(drop=True)
        solution = solution.merge( test[['session_id', 'user_id', 'timestamp', 'step']].drop_duplicates(keep='last'), on='session_id', how='inner' )    
        #solution.to_csv( BASE_PATH + '/' + SET + '/solution_' + ALGKEY + '.csv' )
        
        result = evaluate( solution, base=BASE_PATH, dataset=SET )
        print( result.T )
        
        del solution,test,X_test,y_test,d_train, d_valid, watchlist
        gc.collect()
        
        return -1 * result['mrr@A'].values[0]


def main():
    
    space = {
        'ltr': hp.choice('ltr', [True]),
        'shuffle': hp.choice('shuffle', [False]),
        'num_leaves': hp.choice('num_leaves', list(np.arange(8, 256, 2, dtype=int) )),
        'max_depth': hp.choice('max_depth', list(np.arange(4, 64, 2, dtype=int) )),
        'max_bin': hp.choice('max_bin', list(np.arange(255, 255*4, 5, dtype=int) )),
        'min_data_in_leaf': hp.choice('min_data_in_leaf', list(np.arange(5, 100, 5, dtype=int) )),
        'learning_rate': hp.uniform('learning_rate', 0.01, 0.3),
        'bagging_fraction': hp.uniform('bagging_fraction', 0.2, 1.0),
        'feature_fraction': hp.uniform('feature_fraction', 0.2, 1.0),
        'early_stopping': hp.uniform('test_size', 100, 1000),
    }

    trials_step = 1  # how many additional trials to do after loading saved trials. 1 = save after iteration
    max_trials = 1  # initial max_trials. put something small to not have to wait

    
    try:  # try to load an already saved trials object, and increase the max
        trials = pickle.load(open( BASE_PATH + SET + TRAILKEY + '.hyperopt', "rb"))
        print("Found saved Trials! Loading...")
        max_trials = len(trials.trials) + trials_step
        print("Rerunning from {} trials to {} (+{}) trials".format(len(trials.trials), max_trials, trials_step))
    except:  # create a new trials object and start searching
        trials = Trials()

    best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            trials=trials,
            max_evals=max_trials)
    
    print("Best:", best)
    print("Num:", max_trials)
    
    # save the trials object
    with open(BASE_PATH + SET + TRAILKEY + ".hyperopt", "wb") as f:
        pickle.dump(trials, f)


if __name__ == '__main__':
    while True:
        main()
        