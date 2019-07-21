'''
Created on May 24, 2019

@author: malte
'''
from pathlib import Path

from config.globals import BASE_PATH
from domain.action_types import CLICK, IMAGE, INFO, DEALS, RATING, POI
from helper.df_ops import copy_features, reduce_mem_usage
from helper.loader import load_hdfs, write_feather, load_feather
import numpy as np
import pandas as pd
import time
from featuregen import session
from featuregen.popularity import print_col_list


TEST_FOLDER = BASE_PATH + 'competition/'
CRAWL_FOLDER = BASE_PATH + 'crawled/'

ACTION_MAP = {}
ACTION_MAP['all'] = [CLICK,IMAGE,INFO,DEALS,RATING]
ACTION_MAP['view'] = [IMAGE,INFO,DEALS,RATING]
ACTION_MAP['click'] = [CLICK]

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    geo_features( TEST_FOLDER, CRAWL_FOLDER, log, examples, redo=True )

def geo_features(base_path, crawl_path, log, examples, redo=False):
    
    name = 'geo_features'
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( crawl_path, log, examples )
        examples = reduce_mem_usage(examples, cols=cols)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','prices','label'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list(cols)
        
    return examples

def create_features( crawl_path, log, examples ):
    
    tstart = time.time()
    print( 'create_features geo' )
    
    cols_pre = examples.columns.values
    
    rm = False
    if 'session_last_poi' not in examples.columns:
        rm = True
        examples = session.last_for_reference(log, examples, action=POI, group=['session_id','city'], key='poi' )
        
    if 'ci_lat' not in examples.columns:  
        examples = add_from_file( crawl_path + 'item_info/crawl_ci.csv', examples, to=['impressions'], filter=['ci_lat','ci_lng','ci_rating_percentage'] )
    examples = add_from_file( crawl_path + 'poi/poi_latlng.csv', examples, col=['poi'], to=['session_last_poi'], filter=['poi_lat','poi_lng'] )
    examples = add_from_file( crawl_path + 'city/city_latlng.csv', examples, col=['city'], filter=['city_lat','city_lng'] )
    
    print(examples)
    
    examples['distance_city'] = haversine(examples.ci_lat, examples.ci_lng, examples.city_lat, examples.city_lng)
    examples['distance_poi'] = haversine(examples.ci_lat, examples.ci_lng, examples.poi_lat, examples.poi_lng)
    
    examples['distance_last'] = examples['distance_city']
    examples.ix[~examples['distance_poi'].isnull(), 'distance_last'] = examples.ix[~examples['distance_poi'].isnull()]['distance_poi']
    
    examples = minmax_norm(examples, 'distance_last', na=1)
    examples = minmax_norm(examples, 'distance_city', na=1)
    
    if rm:
        del examples['session_last_poi']
        
    #del examples['ci_lat'], examples['ci_lng']
    del examples['poi_lng'], examples['poi_lat']
    del examples['city_lng'], examples['city_lat']
    
    del examples['distance_poi']
    
    
    if not 'price_list_impressions_minmax' in examples.columns:
        examples = minmax_norm(examples, 'prices')
    else:
        examples['prices_norm'] = examples['price_list_impressions_minmax']
    
    examples['distance_last_per_price'] = examples['distance_last'] * examples['prices']
    examples['distance_last_per_price_norm'] = examples['distance_last_norm'] * examples['prices_norm']
    if 'ci_rating_percentage' in examples.columns:
        examples['distance_last_per_rating'] = examples['distance_last'] / examples['ci_rating_percentage']
    else:
        examples['distance_last_per_rating'] = examples['distance_last'] / examples['ri_rating_percentage']
        
#     examples['distance_last_per_price'] = examples['distance_last_per_price'].replace([np.inf],np.nan).fillna(-1)
#     examples['distance_last_per_price_norm'] = examples['distance_last_per_price_norm'].replace([np.inf],np.nan).fillna(-1)
#     examples['distance_last_per_rating'] = examples['distance_last_per_rating'].replace([np.inf],np.nan).fillna(-1)
    
    del examples['prices_norm']
    if 'ci_rating_percentage' in examples.columns:
        del examples['ci_rating_percentage']
        
#     examples['distance_city'] = examples['distance_city'].replace([np.inf],np.nan).fillna( -1 )
#     examples['distance_last'] = examples['distance_last'].replace([np.inf],np.nan).fillna( -1 )

    new_cols = np.setdiff1d(examples.columns.values, cols_pre)
    
    print( 'create_features geo in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols

def minmax_norm( examples, col, na=-1 ):
    examples[col+'_min'] = examples.groupby( 'session_id' )[col].transform(min)
    examples[col+'_max'] = examples.groupby( 'session_id' )[col].transform(min)
    examples[col+'_norm'] = ( examples[col] - examples[col+'_min'] ) / (examples[col+'_max'] - examples[col+'_min'] )
    examples[col+'_norm'] = examples[col+'_norm'].replace([np.inf],np.nan).fillna( na )
    del examples[col+'_min'], examples[col+'_max']
    
    return examples
    
# vectorized haversine function
def haversine(lat1, lon1, lat2, lon2, to_radians=True, earth_radius=6371):
    """
    slightly modified version: of http://stackoverflow.com/a/29546836/2901002

    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees or in radians)

    All (lat, lon) coordinates must have numeric dtypes and be of equal length.

    """
    if to_radians:
        lat1, lon1, lat2, lon2 = np.radians([lat1, lon1, lat2, lon2])

    a = np.sin((lat2-lat1)/2.0)**2 + \
        np.cos(lat1) * np.cos(lat2) * np.sin((lon2-lon1)/2.0)**2

    return earth_radius * 2 * np.arcsin(np.sqrt(a))
  
def add_from_file( file, examples, col=['item_id'], to=None, filter=None ):
    
    tstart = time.time()
    print( '\t add_from_file {}'.format(file) )
    
    keep = False
    if to is None:
        to = col
        keep = True
    
    toadd = pd.read_csv( file, index_col=0 )
    if filter is not None:
        toadd = toadd[col+filter]
    
    copy = False
    if col[0] in examples.columns:
        copy = True
    
    examples = examples.merge( toadd, left_on=to, right_on=col, how='left' )
    
    if copy and not keep: 
        examples[col[0]] = examples[col[0]+'_y']
        del examples[col[0]+'_y']
    elif not copy and not keep:
        del examples[col[0]]
    
    print( '\t add_from_file in {}s'.format( (time.time() - tstart) ) )
    
    return examples
        

if __name__ == '__main__':
    main()
    