'''
Created on Apr 17, 2019

@author: malte
'''
import pandas as pd
import time
from scipy import sparse
from sklearn import decomposition as dc
import numpy as np
from domain.action_types import CLICK, IMAGE, INFO, RATING, DEALS, IMPRESSION
from config.globals import BASE_PATH
from helper.loader import load_hdfs, ensure_dir
from helper.df_ops import expand
import gc
import implicit

# ACTIONS = [CLICK, IMAGE, INFO, RATING, DEALS, IMPRESSION]
# VALUE = [5, 3, 3, 3, 3, 1]
# KEY = 'all'

ACTIONS_ITEM = [CLICK, IMAGE, INFO, RATING, DEALS]
VALUE_ITEM = [5, 3, 3, 3, 3]
KEY_ITEM = 'item'

ACTIONS_CLICK = [CLICK]
VALUE_CLICK = [1]
KEY_CLICK = 'click'

SIZEA = 32
SIZEB = 128
METHOD = 'bpr'

DATA_FOLDER = BASE_PATH + 'competition/'

def main():
    log = load_hdfs( DATA_FOLDER + 'data_log.hd5' )
    #examples = load_hdfs( DATA_FOLDER + 'data_examples.hd5' )
    create_latent_factors( log, size=SIZEA, actions=ACTIONS_ITEM, values=VALUE_ITEM, key=KEY_ITEM, method=METHOD )
    gc.collect()
    create_latent_factors( log, size=SIZEA, actions=ACTIONS_ITEM, values=VALUE_ITEM, key=KEY_ITEM, method=METHOD )
    gc.collect()
    create_latent_factors( log, size=SIZEB, actions=ACTIONS_CLICK, values=VALUE_CLICK, key=KEY_CLICK, method=METHOD )
    gc.collect()
    create_latent_factors( log, size=SIZEB, actions=ACTIONS_CLICK, values=VALUE_CLICK, key=KEY_CLICK, method=METHOD )
    gc.collect()
    
def create_latent_factors( full, size=32, actions=None, values=None, key='all', method='bpr' ):
    
    start = time.time()
    
    full = full[full.action_type.isin(actions)]
    
    if IMPRESSION in actions:
        full = extend_clicks( full )
    
    full = full.drop_duplicates( ['session_id','reference','action_type'], keep='last' )
    full = full[~full.reference.isnull() & (full.exclude == 0)]
        
    items = full['reference'].unique()
    item_map = pd.Series( index=items, data=range(len(items)) )
    full['item_idx'] = full['reference'].map( item_map )
    
    sessions = full['session_id'].unique()
    session_map = pd.Series( index=sessions, data=range(len(sessions)) )
    full['session_idx'] = full['session_id'].map( session_map )
    
    full['value'] = 1
    for i,action in enumerate(actions):
        full.ix[full.action_type == action, 'value'] = values[i]
    
    full['value'] = full.groupby( ['session_id','reference'] ).value.transform(max)
    full = full.drop_duplicates( ['session_id','reference','action_type'], keep='last' )
    
    SPM = sparse.csr_matrix(( full['value'].tolist(), (full.item_idx, full.session_idx)), shape=( full.item_idx.nunique(), full.session_idx.nunique() ))
    
    print( 'created user features in ',(time.time() - start) )
    
    start = time.time()
    
    start = time.time()
    
    if method == 'bpr':
        model = implicit.bpr.BayesianPersonalizedRanking( factors=size-1, iterations=200, use_gpu=False )
        model.fit(SPM)
        If = model.item_factors
        Sf =  model.user_factors
    elif method == 'als':
        model = implicit.als.AlternatingLeastSquares( factors=size, iterations=200, use_gpu=False, calculate_training_loss=True )
        model.fit(SPM)
        If = model.item_factors
        Sf =  model.user_factors
    elif method == 'nmf':
        nmf = dc.NMF(n_components=size, init='random', random_state=0, max_iter=500, verbose=1)
        If = nmf.fit_transform( SPM )
        Sf = nmf.components_.T

    # train the model on a sparse matrix of item/user/confidence weights

    IF = ['if_'+str(i) for i in range(size)]
    SF = ['sf_'+str(i) for i in range(size)]
    
    Sf = pd.DataFrame( Sf, index=full.session_idx.unique() )
    Sf.columns = SF
    Sf['session_id'] = session_map.index
    If = pd.DataFrame( If, index=full.item_idx.unique() )
    If.columns = IF
    If['item_id'] = item_map.index
    
    item_emb = If.sort_values('item_id')[IF].values
    session_emb = Sf.sort_values('session_id')[SF].values
    
    ensure_dir( DATA_FOLDER + 'latent/' )
    If.to_csv( DATA_FOLDER + 'latent/' + method + 'c_'+key+'_item_features.'+str(size)+'.csv', index=False)
    Sf.to_csv( DATA_FOLDER + 'latent/' + method + 'c_'+key+'_session_features.'+str(size)+'.csv', index=False)
    
    print('created latent features in ',(time.time() - start))
    
    res = []
    
    for row in full.itertuples():
        session = session_emb[int(row.session_idx)]
        item = item_emb[row.item_idx]
        
        res.append( np.dot( item, session.T ) )
        
    full['reconst'] = res
    
    print( full[['item_idx','session_idx','value','reconst']] )
    
def extend_clicks(full):
    
    tstart = time.time()
    
    click_actions = full[full.action_type == CLICK].copy()
    
    click_actions = expand(click_actions, columns=['impressions','prices'])
    click_actions['action_type'] = IMPRESSION
    click_actions['reference'] = click_actions['impressions']
    
    full = pd.concat([full,click_actions])
    full = full.sort_values( ['session_id','timestamp'] )
    
    del click_actions
    gc.collect()
    
    print( 'extended clicks in {}'.format( (time.time() - tstart) ) )
    
    
    return full
    
    
    

if __name__ == '__main__':
    main()