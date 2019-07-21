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
from helper.loader import load_hdfs
from config.globals import BASE_PATH

PREP = 'preprocessed/'
#RAW = 'sample_test/'

ITERATIONS = 20

def main():
    meta = load_hdfs( BASE_PATH + PREP + 'meta_extended.hd5' )
    keep = meta[['item_id','properties_code']]
    
    create_latent_factors(keep)
    
def create_latent_factors( meta, size=16 ):
    
    start = time.time()
    
    items = set( chain.from_iterable( meta.properties_code ) )
    print( len(items) )
    sequences = []
    
    for row in meta.itertuples():
        props = [str(i) for i in row.properties_code]
        shuffle( props )
        sequences.append( TaggedDocument(words=props, tags=[str(row.item_id)]) )
    
    print( 'created sequences in ',(time.time() - start) )
    
    start = time.time()
    
    print('ITEM2VEC FEATURES')
    start = time.time()
    
    model = Doc2Vec(vector_size=size, window=len(items), min_count=1, workers=4)
    model.build_vocab(sequences)
    print('vocab build')
    
    for i in range(ITERATIONS):
        shuffle(sequences)
        model.train(sequences, epochs=2, total_examples=model.corpus_count)
        print('trained {} in {}'.format( i, ( time.time() - start ) ))
    
    d = {}  
    for item in meta.item_id.values:
        d[str(item)] = model[str(item)]
    
    frame = pd.DataFrame( d )
    frame = frame.T
    frame.columns = ['if_'+str(i) for i in range(size)]
    frame['item_id'] = pd.to_numeric( frame.index ).astype(np.int32)
    
    frame.to_csv( BASE_PATH + PREP +'d2v_item_features.'+str(size)+'.csv', index=False)
    
    print('created latent features in ',(time.time() - start))

if __name__ == '__main__':
    main()