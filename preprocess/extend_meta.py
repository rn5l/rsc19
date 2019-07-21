'''
Created on Apr 15, 2019

@author: malte
'''

import pandas as pd
import numpy as np
from helper.apply_ops import split_pipe
from config.globals import BASE_PATH
from helper.loader import ensure_dir

RAW = BASE_PATH + 'raw/'
PREP = BASE_PATH + 'preprocessed/'

def main():
    
    meta = pd.read_csv( RAW + 'item_metadata.csv' )
    meta = featurize( meta, 'properties' )
    meta['rating'] = count_rating( meta )
    meta['stars'] = count_stars( meta )
    meta['features'] = meta['list'].apply( len )
    
    ensure_dir( PREP )
    meta.to_csv( PREP + 'item_metadata.csv' )
    
def featurize( meta, col ):
    
    meta['list'] = meta[col].apply( split_pipe, convert=str )
    meta['set'] = meta['list'].apply( lambda x: frozenset(x) )
    for prop in frozenset.union(*meta['set']):
        print( 'added ', prop )
        meta[prop] = meta['set'].apply(lambda x: int(prop in x))
    
    del meta['set']
    
    return meta

def count_rating( meta ):
    
    rating = np.zeros( len(meta) )
    rating += meta['Satisfactory Rating']
    rating += meta['Good Rating']
    rating += meta['Very Good Rating']
    rating += meta['Excellent Rating']
    return rating

def count_stars( meta ):
    
    stars = np.zeros( len(meta) )
    for i in range(1,6):
        stars += meta[str(i)+' Star'] * i
    return stars

if __name__ == '__main__':
    main()