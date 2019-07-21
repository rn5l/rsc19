'''
Created on May 15, 2019

@author: malte
'''
import pickle
import numpy as np
import pandas as pd
from config.globals import BASE_PATH

SET = 'sample_small/'
FILE = 'trails-lgbm-lr0.05.hyperopt'
ALGO = 'lgbm'

space = {
'ltr':[True,False],
'shuffle': [True,False],
'num_leaves': list(np.arange(8, 256, 2, dtype=int) ),
'max_depth': list(np.arange(4, 64, 2, dtype=int) ),
'max_bin': list(np.arange(255, 255*4, 5, dtype=int) ),
'min_data_in_leaf': list(np.arange(5, 100, 5, dtype=int) ),
#'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
#'learning_rate': 0.01,
#'bagging_fraction': hp.uniform('bagging_fraction', 0.3, 1.0),
#'feature_fraction': hp.uniform('feature_fraction', 0.3, 1.0)
}

space_keras = {
'shuffle': [True,False],
'half': [True,False],
'opt':['adam','sgd','adadelta'],
'num_layers': list(np.arange(2, 20, 2, dtype=int) ),
'layer_size': list(np.arange(200, 4000, 200, dtype=int) ),
'batch': list(np.arange(500, 10000, 100, dtype=int) ),
#'colsample_bytree': hp.uniform('colsample_bytree', 0.3, 1.0),
#'learning_rate': 0.01,
#'bagging_fraction': hp.uniform('bagging_fraction', 0.3, 1.0),
#'feature_fraction': hp.uniform('feature_fraction', 0.3, 1.0)
}

def main():
    if ALGO == 'lgbm':
        main_lgbm()
    if ALGO == 'keras':
        main_keras()

def main_lgbm():

    trials = pickle.load( open( BASE_PATH + SET + FILE, 'rb' ) )
    res = {}
    res['ltr'] = []
    res['shuffle'] = []
    res['num_leaves'] = []
    res['max_depth'] = []
    res['max_bin'] = []
    res['min_data_in_leaf'] = []
    res['bagging_fraction'] = []
    res['feature_fraction'] = []
    res['test_size'] = []
    res['loss'] = []
    
    for entry in trials: 
        print( entry )
        for k,v in entry['misc']['vals'].items():
            if k in space:
                res[k].append( space[k][ v[0] ] )
            else:
                res[k].append( v[0] )
        
        res['loss'].append( entry['result']['loss'] )
    
    print( pd.DataFrame( res ).sort_values( ['loss'], ascending=True ).head(5).T )

def main_keras():

    trials = pickle.load( open( BASE_PATH + SET + FILE, 'rb' ) )
    res = {}
    res['half'] = []
    res['opt'] = []
    res['learning_rate'] = []
    res['num_layers'] = []
    res['layer_size'] = []
    res['batch'] = []
    res['drop'] = []
    res['drop_final'] = []
    res['momentum'] = []
    res['loss'] = []
    
    for entry in trials: 
        print( entry )
        for k,v in entry['misc']['vals'].items():
            if k in space_keras:
                val = space_keras[k][ v[0] ]
                res[k].append( val )
            else:
                res[k].append( v[0] )
        
        res['loss'].append( entry['result']['loss'] )
    
    for k,v in res.items(): 
        print(k)
        print(len(v))
    
    print( pd.DataFrame( res ).sort_values( ['loss'], ascending=True ).head(5).T )

if __name__ == '__main__':
    main()