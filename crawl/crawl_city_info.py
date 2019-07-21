'''
Created on May 15, 2019

@author: malte
'''

import requests
import pickle
import pandas as pd
import numpy as np
from helper.apply_ops import split_pipe
from glob import glob
import time
from tqdm import tqdm
import sys
import traceback
import datetime
from geopy.geocoders.osm import Nominatim
from geopy.geocoders.here import Here
from geopy.geocoders.mapbox import MapBox
from geopy.geocoders.googlev3 import GoogleV3
from config.globals import BASE_PATH


CRAWL_FOLDER = 'crawled/city/'
PREP_FOLDER = 'preprocessed/'

EXPORT_FOLDER = CRAWL_FOLDER + 'google/'

RETRY = 5
DUMP_AFTER = 1000

KEY = '***REMOVED***'

COLLECT_FOLDERS = [CRAWL_FOLDER+'google/', CRAWL_FOLDER+'here/', CRAWL_FOLDER+'mapbox/']

def main():
    crawl()
    #collect()
    
def collect():
    
    itemset = set()
    
    res = {}
    res['city'] = []
    res['city_lat'] = []
    res['city_lng'] = []
    
    for folder in COLLECT_FOLDERS:
        cities = get_processed( BASE_PATH + folder )
        for k,v in cities.items():
            #pprint.pprint( items_done[k] )
            if k not in itemset and v is not None:
                res['city'].append( k )
                res['city_lat'].append( v[0] )
                res['city_lng'].append( v[1] )
                itemset.add(k)
        
        print( 'size after ',folder,': ',len(itemset) )
            
    res = pd.DataFrame(res)
    res.to_csv( BASE_PATH + CRAWL_FOLDER + 'city_latlng.csv' )
    
def crawl():
    
    geolocator = GoogleV3(api_key=KEY)
    #geolocator = Nominatim(user_agent="tu-dortmund-research2")
    #geolocator = Here(app_id='***REMOVED***', app_code='***REMOVED***')
    #geolocator = MapBox(api_key='***REMOVED***')
#     print( get_city('Recife, Brazil', gmaps=geolocator) )
#     exit()
    
    cities = get_cities( BASE_PATH + PREP_FOLDER )
    cities_done = get_processed( BASE_PATH + EXPORT_FOLDER )
        
    cities = cities[~cities.city.isin(cities_done)]
    
    result_map = {}
    
    first = cities.city.values[0]
    
    togo = len(cities)
    tstart = time.time()
        
    try:
        
        retries = RETRY
        fuck = False
        dump = DUMP_AFTER
        
        i = 0
        
        while not fuck:
            
            if i == len(cities):
                break
            
            try:
            
                item = cities.city.values[i]
                item_str = cities.city_str.values[i].replace('/',' ')
                
                res = get_city( item_str, gmaps=geolocator )
                if res is not None:
                    result_map[item] = res
                    
                #time.sleep(1)
                i += 1
                retries = RETRY
                togo -= 1 
                dump -= 1
                
                if dump == 0:
                    pickle.dump( result_map, open( BASE_PATH + EXPORT_FOLDER + 'cfrom_'+str(first)+'.pkl', 'wb' ) )
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
                    
                    print( 'done {} of {} in {}, {} left'.format( done, len(cities), spent, eta ) )
                
                
            except Exception as e:
                retries -= 1
                print( 'retries ', retries )
                print( 'city ', item_str )
                exc_type, exc_value, exc_traceback = sys.exc_info()
                print("*** print_tb:")
                traceback.print_tb(exc_traceback)
                print(e)
                if retries <= 0: 
                    raise
                wait = RETRY - retries + 1
                time.sleep(pow(2, wait))
                
            
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        print("*** print_tb:")
        traceback.print_tb(exc_traceback)
        pickle.dump( result_map, open( BASE_PATH + EXPORT_FOLDER + 'cfrom_'+str(first)+'.pkl', 'wb' ) )
        
    pickle.dump( result_map, open( BASE_PATH + EXPORT_FOLDER + 'cfrom_'+str(first)+'.pkl', 'wb' ) )

def get_city( city, gmaps=None ):
    
    if gmaps is not None:
        geocode_result = gmaps.geocode(city)
        if geocode_result == None:
            print( city )
            return None
    else:
        geocode_result = requests.get( 'http://www.datasciencetoolkit.org/maps/api/geocode/json?sensor=false&address='+city )
        
    return geocode_result.latitude, geocode_result.longitude, geocode_result.raw
    
def get_processed( folder ):
    
    items = {}
    
    for file_name in glob( folder + '*.pkl' ):
        res_part = pickle.load( open( file_name, 'rb' ) )
        for k,v in res_part.items():
            items[k] = v
    
    return items


def get_cities( folder ):
        
    cities_map = pd.read_csv( folder + 'city_map.csv', header=None, names=['city_str', 'city'], dtype={0:str,1:int} ) 
    
    return cities_map
    
if __name__ == '__main__':
    main()
    
    