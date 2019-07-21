'''
Created on May 24, 2019

@author: malte
'''
from featuregen.popularity import pop_features, print_col_list
from featuregen.price import price_features
from config.globals import BASE_PATH
from helper.loader import load_hdfs, load_feather, write_feather
import gc
from featuregen.session import session_features
from featuregen.user import user_features
from pathlib import Path
from featuregen.crawl import crawl_features
from featuregen.geo import geo_features
from featuregen.rank import rank_features
from featuregen.meta import meta_features
from featuregen.position import position_features
from featuregen.properties import properties_features
from featuregen.latent_sim import latent_sim_features
from featuregen.combine import combine_features
from helper.df_ops import reduce_mem_usage
from featuregen.time import time_features
from featuregen.stars import stars_features
from featuregen.list_context import list_conext, list_context_features

SET = BASE_PATH + 'competition/'

#pop features conf
CONF = {
    'train_only': False,
    
    'pop_hidden': False,
    'path_pop': SET,
    'min_pop': None,
    
    'price_hidden': False,
    'path_price': SET,
    'min_occurences': None,
    'fillna_mean': False,
    
    'path_session': SET,
    'path_poi': SET,
    
    'path_crawl': BASE_PATH + 'crawled/',
    
    'path_meta': BASE_PATH + 'preprocessed/',
    'meta_latent': 'd2v',
    
    'path_latent': BASE_PATH + 'competition/',
}



def main():

    examples = create_set( SET, conf=None, redo=True )
    print_col_list( examples )
    

def create_set(base_path=SET, key='dataset', conf={}, redo=False): 
    
    name = key
    
    path = Path( base_path + 'sets/' + name + '.fthr' )
    if path.is_file() and not redo:
        print( 'loaded' )
        examples = load_feather( path )
        gc.collect()
    else:
        print( 'create' )
        log = load_hdfs( base_path + 'data_log.hd5' )
        examples = load_hdfs( base_path + 'data_examples.hd5' )
        if 'current_filters' in set(examples.columns):
            print( 'current_filters' )
            del examples['current_filters']
        if 'session_id_pre' in set(examples.columns):
            print( 'session_id_pre' )
            del examples['session_id_pre'] 
            
        examples = pop_features(conf['path_pop'], log, examples, hidden=conf['pop_hidden'], min_pop=conf['min_pop'], train_only=conf['train_only'], redo=redo)
        examples = price_features(conf['path_price'], log, examples, min_occurences=conf['min_occurences'], hidden=conf['price_hidden'], train_only=conf['train_only'], fillna_mean=conf['fillna_mean'], redo=redo)
        examples = session_features(conf['path_session'], log, examples,crawl_path=conf['path_crawl'],  redo=redo)
        examples = crawl_features(base_path, conf['path_crawl'], log, examples, redo=redo)
        examples = geo_features(base_path, conf['path_crawl'], log, examples, redo=redo)
        examples = meta_features(base_path, conf['path_meta'], log, examples, latent=conf['meta_latent'], redo=redo)
        examples = user_features(conf['path_session'], log, examples,crawl_path=conf['path_crawl'], poi_path=conf['path_poi'], redo=redo)
        examples = position_features(base_path, log, examples, redo=redo)
        examples = properties_features(base_path, conf['path_meta'], log, examples, redo=redo)
        #examples = latent_features(base_path, log, examples, latent_path=conf['path_latent'], redo=redo)
        examples = latent_sim_features(base_path, log, examples, latent_path=conf['path_latent'], redo=redo)
        examples = combine_features(base_path, log, examples, redo=redo)
        examples = rank_features(base_path, log, examples, redo=redo)
        examples = time_features(base_path, log, examples, redo=redo)
        examples = list_context_features(base_path, log, examples, redo=redo)
        examples = stars_features(base_path, conf['path_meta'], log, examples, redo=redo)
        #examples = prediction_features(base_path, log, examples, redo=redo)
        
        #examples.to_csv( base_path + 'sets/' + name + '.csv' )
        write_feather(examples, path)
        
        del log
        gc.collect()
    
    #print_col_list( examples.columns )
    #examples = reduce_mem_usage(examples)
    return examples

def print_col_list( examples, name='ALLF' ):
    
    print( name +' = [' )
    for name in examples:
        print( "   '"+name+"'," )
    print( ']' )
    
def resolve_na(train):
    
    cols = []
    for col in train.columns: 
        if sum( train[col].isnull() ) > 0:
            #print('na ',col)
            cols.append( col )
    print_col_list(cols, name='NA_COLS')
    exit()
     
if __name__ == '__main__':
    main()