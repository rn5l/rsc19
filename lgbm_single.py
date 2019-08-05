'''
Created on Mar 19, 2019

@author: malte
'''

from collections import Counter
import gc
from itertools import chain
import time

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection.univariate_selection import f_classif

from config.globals import BASE_PATH
from domain.features import FEATURES, CAT_FEATURES, WO_CRAWL
from evaluate import evaluate
from featuregen.create_set import create_set
from helper.df_ops import train_test_split, check_cols
import lightgbm as lgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from helper.loader import ensure_dir

RAW = 'raw/'
SET = 'sample_test/'

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
VALID = 0.1
STOPPING = 500
FS = None
FS_IMP = None

#KEYS
DSKEY = 'dataset'
ALGKEY = 'lgbm_{}_{}_val{}-{}_{}{}_dart0.1'.format( 'ltr' if LTR else 'bin', 'shfl' if SHUFFLE else 'noshfl', VALID, STOPPING, 'FS'+str(FS)+'_' if FS is not None else '', 'FSIMP'+str(FS_IMP)+'_' if FS_IMP is not None else '' )
STACK = False

def main():
    
    tstart = time.time()
    
    train = create_set( base_path=BASE_PATH + SET, conf=CONF, key=DSKEY, redo=False )
    #train = resolve_na(train)
    
    print( 'loaded in {}'.format( (time.time() - tstart) ) )
    tstart = time.time()
        
    #test = train.query('train == 0')
    train.query('train == 1', inplace=True)
        
    print( 'split in {}'.format( (time.time() - tstart) ) )
    tstart = time.time()
    
    if FS_IMP is not None:
        FEATURES_IMP = get_features_by_importance(FS_IMP)
    else:
        FEATURES_IMP = WO_CRAWL
    
    print( [item for item, count in Counter(FEATURES).items() if count > 1] )
    
    y = train[ 'label' ]
    X = train[ FEATURES_IMP + ['session_id'] ] 
            
    #input("Press Enter to continue...")
    
    print( 'FEATURES in in {}'.format( (time.time() - tstart) ) )
    tstart = time.time()
    
    if STACK:
        train_stack = train[['user_id','session_id','step','timestamp','impressions']].copy()
    del train
    gc.collect()
    
    print( 'gc collect in in {}'.format( (time.time() - tstart) ) )
    tstart = time.time()
    
    if FS != None:
        check_cols( X )
        keep = feature_selection(X[FEATURES_IMP], y ,FS)
        KEEP_FEATURES = [ FEATURES_IMP[i] for i in keep ]
    else:
        KEEP_FEATURES = FEATURES_IMP
    
    X_train, X_valid, y_train, y_valid = train_test_split( X, y, test_size=VALID, shuffle=SHUFFLE )
    
    print( 'split in in {}'.format( (time.time() - tstart) ) )
    tstart = time.time()
        
    if LTR:
        q_train =  X_train.groupby( ['session_id'] ).size().values.astype(np.float32)
        q_valid = X_valid.groupby( ['session_id'] ).size().values.astype(np.float32)
        xtrain = X_train[KEEP_FEATURES].values.astype(np.float32)
        ytrain = y_train.values.astype(np.float32)
        del X_train, y_train
        gc.collect()
        d_train = lgbm.Dataset( xtrain, label=ytrain, group=q_train, feature_name=KEEP_FEATURES)#, categorical_feature=CAT_FEATURES )
        del q_train
        gc.collect()
        xval = X_valid[KEEP_FEATURES].values.astype(np.float32)
        yval = y_valid.values.astype(np.float32)
        del X_valid, y_valid
        gc.collect()
        d_valid = lgbm.Dataset( xval, label=yval, group=q_valid, feature_name=KEEP_FEATURES)#, categorical_feature=CAT_FEATURES )
        del q_valid
        gc.collect()
    else:
        xtrain = X_train[KEEP_FEATURES].values.astype(np.float32)
        ytrain = y_train.values.astype(np.float32)
        del X_train, y_train
        gc.collect()
        d_train = lgbm.Dataset( xtrain, label=ytrain, feature_name=KEEP_FEATURES )#+ ['session_id'])#, categorical_feature=CAT_FEATURES )
        del xtrain, ytrain
        gc.collect()
        xval = X_valid[KEEP_FEATURES].values.astype(np.float32)
        yval = y_valid.values.astype(np.float32)
        del X_valid, y_valid
        gc.collect()
        d_valid = lgbm.Dataset( xval, label=yval, feature_name=KEEP_FEATURES )#+ ['session_id'])#, categorical_feature=CAT_FEATURES )
        del xval, yval
        gc.collect()
        
    print( 'create sets in {}'.format( (time.time() - tstart) ) )
    tstart = time.time()
        
    watchlist = [d_train, d_valid]

    params = {}
    params['boosting'] = 'dart'
    params['learning_rate'] = 0.1
    if LTR:
        params['application'] = 'lambdarank'
        params['metric'] = 'ndcg'
        params['eval_at'] = '30'
        #params['group_column'] = 'name:session_id'
    else:
        params['application'] = 'binary'
        params['metric'] = 'binary_logloss'
#     params['max_depth'] = 34
#     params['num_leaves'] = 234
#     params['max_bin'] = 485
#     params['feature_fraction'] = 0.202505
#     params['bagging_fraction'] = 0.823505
#     params['min_data_in_leaf'] = 15
    params['feature_fraction'] = 0.5
    params['bagging_fraction'] = 0.5
    #params['bagging_freq'] = 5
    #params['verbosity'] = 0
    
    evals_result = {}
    model = lgbm.train( params, train_set=d_train, num_boost_round=10000, valid_sets=watchlist, early_stopping_rounds=STOPPING, evals_result=evals_result, verbose_eval=10 ) #, feval=mrr )

    print( 'train in in {}'.format( (time.time() - tstart) ) )
    tstart = time.time()
     
#     ax = lgbm.plot_metric(evals_result, metric='auc')
#     plt.show()
    
    export_importance( model, KEEP_FEATURES, export=FS is None and FS_IMP is None )
    
    ensure_dir(BASE_PATH + SET + 'lgbm/')
    model.save_model( BASE_PATH + SET + 'lgbm/'+ALGKEY+'.txt' , num_iteration=model.best_iteration)
    
    test = create_set( base_path=BASE_PATH + SET, conf=CONF, key=DSKEY, redo=False )
    test.query('train == 0', inplace=True)
    
    
    X_test = test[ KEEP_FEATURES ]
    y_test = model.predict(X_test, num_iteration=model.best_iteration )
    
    print( 'predict in {}'.format( (time.time() - tstart) ) )
    tstart = time.time()
    
    test['prob'] = y_test
    
    if STACK:
        test[['user_id','session_id','step','timestamp','impressions','prob']].to_csv( BASE_PATH + '/' + SET + '/stacking/teprobs_' + ALGKEY + '.csv' )
        
        y_pred = model.predict( X[ KEEP_FEATURES ] )
        train_stack['prob'] = y_pred
        train_stack[['user_id','session_id','step','timestamp','impressions','prob']].to_csv( BASE_PATH + '/' + SET + '/stacking/trprobs_' + ALGKEY + '.csv' )
     
    
#     truth = pd.read_csv( BASE_PATH + SET + 'truth.csv' )
#     truth['label2'] = 1
#     test = test.merge( truth[['session_id','reference','label2']], left_on=['session_id','impressions'], right_on=['session_id','reference'], how='left' )
#     test['label'] =  test['label2'].fillna(0)
#     del test['label2']
    
    test = test.sort_values(['session_id','prob'], ascending=False)
    
#     test.to_csv( BASE_PATH + SET + '/test_examine.csv' )
    
    solution = pd.DataFrame()
    solution['recommendations'] = test.groupby( 'session_id' ).impressions.apply( list )
    solution['confidences'] = test.groupby( 'session_id' ).prob.apply( list )
    solution.reset_index(drop=True)
    solution = solution.merge( test[['session_id', 'user_id', 'timestamp', 'step']].drop_duplicates(keep='last'), on='session_id', how='inner' )    
    solution.to_csv( BASE_PATH + '/' + SET + '/solution_' + ALGKEY + '.csv' )
    
    result = evaluate( solution, base=BASE_PATH, dataset=SET )
    print( result.T )


def mrr( y_hat, train ):
    res = 0
    correct = train.get_label()
    groups = train.get_group()
    gids = list( chain.from_iterable( [ [idx] * num for num,idx in zip( groups, range(len(groups)) ) ] ) )
    pos = list( chain.from_iterable( [ range(1,num+1) for num,idx in zip( groups, range(len(groups)) ) ] ) )
    a = pd.DataFrame( {'y': y_hat, 'correct': correct, 'group': gids } )
    a = a.sort_values( ['group','y'], ascending=[True,False] )
    a['rank'] = pos
    res = np.mean( 1 / a[ a.correct == 1 ]['rank'].values )
    return 'mrr', res, True

def feature_selection( X, y, k=10 ):

    print('feature_selection ',k)
#     var = VarianceThreshold(threshold=(.8 * (1 - .8)))
#     var.fit_transform(X)
#     sup_var = var.get_support(indices=True)
    sup_cat = [ FEATURES.index( f ) for f in CAT_FEATURES ]
    
#     clf = ExtraTreesClassifier(n_estimators=5)
#     selector = SelectFromModel(clf, threshold=0, max_features=k)    
    selector = SelectKBest(f_classif, k=k)
    selector.fit(X, y)
    # Get columns to keep
    sup_kbest = selector.get_support(indices=True)
    # Create new dataframe with only desired columns, or overwrite existing
#     print( sup_var )
    print( sup_kbest )
    print( len(FEATURES) )
#     sup_kbest = list( filter( lambda x: x in sup_var or x in sup_cat, sup_kbest) )
    
    tmp = [ FEATURES[i] for i in list(set(sup_cat) | set(sup_kbest)) ]
    print( tmp )
    
    return list(set(sup_cat) | set(sup_kbest))

def get_features_by_importance( FS_IMP ):
    
    importance = pd.read_csv( BASE_PATH + SET + 'lgbm_importance.csv' )
    features = list( importance.sort_values('split').tail(FS_IMP).feature.values )
    print( 'select {} of {} features'.format( len(features), len(FEATURES) ) )
    return features
 
def export_importance(model, features, plot=False, export=True):
    
    if export:
        importance_split = model.feature_importance()
        importance_gain = model.feature_importance( importance_type='gain' )
        importance = pd.DataFrame({ 'feature': features, 'split': importance_split, 'gain': importance_gain })
        importance.to_csv( BASE_PATH + SET + 'lgbm_importance.csv', index=False )
    if plot:
        ax = lgbm.plot_importance(model, max_num_features=50)
        plt.show()
      
if __name__ == '__main__':
    main()
    