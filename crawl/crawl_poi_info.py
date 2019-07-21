'''
Created on May 15, 2019

@author: malte
'''

import datetime
from glob import glob
import pickle
import sys
import time
import traceback

from geopy.geocoders.googlev3 import GoogleV3
import requests

from config.globals import BASE_PATH
from domain.action_types import POI
import pandas as pd


CRAWL_FOLDER = 'crawled/'
PREP_FOLDER = 'preprocessed/'

EXPORT_FOLDER = 'poi/'

RETRY = 5
DUMP_AFTER = 10000

COLLECT_FOLDERS = ['poi/']

def main():
    crawl()
    collect()
    
def collect():
    
    itemset = set()
    
    res = {}
    res['poi'] = []
    res['poi_lat'] = []
    res['poi_lng'] = []
    
    for folder in COLLECT_FOLDERS:
        cities = get_processed( BASE_PATH + CRAWL_FOLDER + folder )
        for k,v in cities.items():
            #pprint.pprint( items_done[k] )
            if k not in itemset:
                res['poi'].append( k )
                res['poi_lat'].append( v[0] )
                res['poi_lng'].append( v[1] )
                itemset.add(k)
        
        print( 'size after ',folder,': ',len(itemset) )
            
    res = pd.DataFrame(res)
    res.to_csv( BASE_PATH + CRAWL_FOLDER + 'poi/poi_latlng.csv' )

def crawl():
    
    geolocator = GoogleV3(api_key='***REMOVED***')
    #geolocator = Nominatim(user_agent="tu-dortmund-research")
#     print( get_poi('Consulate General of the Unites States, Recife, Brazil', gmaps=geolocator) )
#     exit()
    
    pois = get_pois( BASE_PATH + PREP_FOLDER )
    pois_done = get_processed( BASE_PATH + CRAWL_FOLDER + EXPORT_FOLDER )
    
    pois = pois[~pois.poi.isin(pois_done)]
    
    #test
    pois = pois[:10]
    
    result_map = {}
    
    first = pois.poi.values[0]
    
    togo = len(pois)
    tstart = time.time()
        
    try:
        
        retries = RETRY
        fuck = False
        dump = DUMP_AFTER
        
        i = 0
        
        while not fuck:
            
            if i == len(pois):
                break
            
            try:
            
                item_str = pois.poi_str.values[i]
                item = pois.poi.values[i]
                
                res = get_poi( item_str, gmaps=geolocator )
                if res is not None:
                    result_map[item] = res
                #time.sleep(1)
                i += 1
                retries = RETRY
                togo -= 1 
                dump -= 1
                
                if dump == 0:
                    pickle.dump( result_map, open( BASE_PATH + CRAWL_FOLDER + EXPORT_FOLDER + 'pfrom_'+str(first)+'.pkl', 'wb' ) )
                    dump = DUMP_AFTER
                    first = item
                    result_map = {}
                
                if togo % 100 == 0:
                    spent = time.time() - tstart
                    done = i + 1 
                    each = spent / done
                    left = each * togo
                    eta = datetime.timedelta( seconds=left )
                    spent = datetime.timedelta( seconds=spent )
                    
                    print( 'done {} of {} in {}, {} left'.format( done, len(pois), spent, eta ) )
                
                
            except Exception:
                retries -= 1
                print( 'retries ', retries )
                if retries <= 0: 
                    raise
                wait = RETRY - retries + 1
                time.sleep(pow(2, wait))
                
            
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_tb:")
        traceback.print_tb(exc_traceback)
        pickle.dump( result_map, open( BASE_PATH + CRAWL_FOLDER + EXPORT_FOLDER + 'pfrom_'+str(first)+'.pkl', 'wb' ) )
        
    pickle.dump( result_map, open( BASE_PATH + CRAWL_FOLDER + EXPORT_FOLDER + 'pfrom_'+str(first)+'.pkl', 'wb' ) )

def get_poi( item, gmaps=None ):
    
    if gmaps is not None:
        geocode_result = gmaps.geocode(item)
        #time.sleep(0.1)
        if geocode_result == None:
            print( item )
            return None
    else:
        geocode_result = requests.get( 'http://www.datasciencetoolkit.org/maps/api/geocode/json?sensor=false&address='+item )
        
    return geocode_result.latitude, geocode_result.longitude, geocode_result.raw
    
def get_processed( folder ):
    
    items = {}
    
    for file_name in glob( folder + '*.pkl' ):
        res_part = pickle.load( open( file_name, 'rb' ) )
        for k,v in res_part.items():
            items[k] = v
    
    return items


def get_pois( folder ):
        
    all3 = pd.read_hdf( folder + 'joined_final.hd5', key='table' )
    poicity = pd.DataFrame()
    poicity['city'] = all3[all3.action_type == POI].groupby( 'reference' ).city.min()
    poicity['poi'] = poicity.index
    del all3
    
    cities_map = pd.read_csv( folder + 'city_map.csv', header=None, names=['city_str', 'city'], dtype={0:str,1:int} ) 
    poi_map = pd.read_csv( folder + 'poi_map.csv', header=None, names=['poi_str', 'poi'], dtype={0:str,1:int} ) 
    
    poi_map = poi_map.merge( poicity, on="poi", how="left" )
    poi_map = poi_map.merge( cities_map, on="city", how="left" )
    poi_map['poi_str'] = poi_map['poi_str'] + ', ' + poi_map['city_str']
    del poi_map['city_str'], poi_map['city']
    print(poi_map)
    
    return poi_map
    
if __name__ == '__main__':
    main()
    
    