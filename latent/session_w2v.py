'''
Created on Apr 17, 2019

@author: malte
'''
import pandas as pd
import time
from scipy import sparse
from sklearn import decomposition as dc
import numpy as np
from random import shuffle
from itertools import chain
import gensim
from gensim.models.doc2vec import TaggedDocument, Doc2Vec
from helper.loader import load_hdfs, ensure_dir
from config.globals import BASE_PATH
from domain.action_types import CLICK, IMAGE, INFO, RATING, DEALS, IMPRESSION

# ACTIONS = [CLICK, IMAGE, INFO, RATING, DEALS, IMPRESSION]
# KEY = 'all'

ACTIONS_ITEM = [CLICK, IMAGE, INFO, RATING, DEALS]
KEY_ITEM = 'item'

ACTIONS_CLICK = [CLICK]
KEY_CLICK = 'click'

SIZE = 32
ITERATIONS = 20

DATA_FOLDER = BASE_PATH + 'competition/'

def main():
    log = load_hdfs( DATA_FOLDER + 'data_log.hd5' )
    #examples = load_hdfs( DATA_FOLDER + 'data_examples.hd5' )
    create_latent_factors( log, size=SIZE, actions=ACTIONS_ITEM, key=KEY_ITEM )
    create_latent_factors( log, size=SIZE, actions=ACTIONS_CLICK, key=KEY_CLICK )
    
def create_latent_factors( full, size=32, actions=ACTIONS_CLICK, key=KEY_CLICK ):
    
    start = time.time()
    
    full = full[full.action_type.isin(actions)]
    
    full = full.drop_duplicates( ['session_id','reference','action_type'], keep='last' )
    full = full[~full.reference.isnull() & (full.exclude == 0)]
    
    items = set( full.reference.unique() )
    print( len(items) )
    
    lists = pd.DataFrame()
    lists['session_id'] = full.groupby('session_id').session_id.min()
    lists['sequence'] = full.groupby('session_id').reference.apply( list )
    del full
    
    sequences = []
    
    for row in lists.itertuples():
        props = [str(i) for i in row.sequence]
        sequences.append( TaggedDocument(words=props, tags=[str(row.session_id)]) )
    
    print( 'created sequences in ',(time.time() - start) )
    
    start = time.time()
    
    print('ITEM2VEC FEATURES')
    start = time.time()
    
    model = Doc2Vec(vector_size=size, window=5, min_count=1, workers=4)
    model.build_vocab(sequences)
    print('vocab build')
    
    for i in range(ITERATIONS):
        model.train(sequences, epochs=1, total_examples=model.corpus_count)
        print('trained {} in {}'.format( i, ( time.time() - start ) ))
    
    d = {}  
    for item in lists.session_id.values:
        d[str(item)] = model[str(item)]
    
    frame = pd.DataFrame( d )
    frame = frame.T
    frame.columns = ['sf_'+str(i) for i in range(size)]
    frame['session_id'] = pd.to_numeric( frame.index ).astype(np.int32)
    
    ensure_dir( DATA_FOLDER + 'latent/' )
    frame.to_csv( DATA_FOLDER + 'latent/' + 'd2v_'+key+'_session_features.'+str(size)+'.csv', index=False)
    
    d = {}  
    for item in items:
        d[str(item)] = model.wv[str(item)]
    
    frame = pd.DataFrame( d )
    frame = frame.T
    frame.columns = ['if_'+str(i) for i in range(size)]
    frame['item_id'] = pd.to_numeric( frame.index ).astype(np.int32)
    
    frame.to_csv( DATA_FOLDER + 'latent/' + 'd2v_'+key+'_item_features.'+str(size)+'.csv', index=False)
    
    print('created latent features in ',(time.time() - start))

if __name__ == '__main__':
    main()