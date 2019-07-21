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
import sys
import traceback
import datetime
import pprint
from config.globals import BASE_PATH
from helper.loader import ensure_dir

RAW_FOLDER = 'raw/'
PREP_FOLDER = 'preprocessed/'

INFO_FOLDER = 'crawled/item_info/'
INFO_URL = 'https://www.trivago.de/api/v1/accommodation/:ID:/complete-info.json'

RATING_FOLDER = 'crawled/item_rating/'
RATING_URL = 'https://www.trivago.de/api/v1/accommodation/:ID:/ratings.json'

IMAGES_FOLDER = 'crawled/item_images/'
IMAGES_URL = 'https://www.trivago.de/api/v1/accommodation/:ID:/gallery.json'

RETRY = 5
DUMP_AFTER = 10000

def main():
    crawl(BASE_PATH + INFO_FOLDER, INFO_URL)
    #crawl(BASE_PATH + RATING_FOLDER, RATING_URL)
    #crawl(BASE_PATH + IMAGES_FOLDER, IMAGES_URL)
    
    collect_info()
    #collect_rating()
    #collect_images()
    
def collect_info():
    items_done = get_processed( BASE_PATH + INFO_FOLDER )
    
    titles = ['WLAN in Lobby','WLAN im Zimmer','Pool','Wellness','Parkplätze','Haustiere erlaubt','Klimaanlage','Restaurant','Hotelbar','Fitnessraum']
    keys = ['wifi_lobby','wifi_room','pool','spa','parking','pets','aircondition','restaurant','bar','gym']
    
    res = {}
    res['item_id'] = []
    res['ci_rating_index'] = []
    res['ci_rating_percentage'] = []
    res['ci_lat'] = []
    res['ci_lng'] = []
    res['ci_stars'] = []
    res['ci_superior'] = []
    res['ci_partner'] = []
    res['ci_itemgroup'] = []
    for entry in keys:
        res['ci_'+entry] = []
    
    for k,v in items_done.items():
        
        #pprint.pprint( items_done[k] )
        
        res['item_id'].append( k )
        
        res['ci_rating_index'].append( v['accommodation']['reviewRating']['index'] )
        res['ci_rating_percentage'].append( v['accommodation']['reviewRating']['percentage'] )
        res['ci_lat'].append( v['accommodation']['geocode']['lat'] )
        res['ci_lng'].append( v['accommodation']['geocode']['lng'] )
        res['ci_stars'].append( v['accommodation']['hotelStarRating']['starCount'] )
        res['ci_superior'].append( v['accommodation']['hotelStarRating']['isSuperior'] )
        res['ci_partner'].append( v['accommodation']['isPremiumPartner'] )
        res['ci_itemgroup'].append( v['accommodation']['itemGroup'] )
        
        for e in v['amenities']['topFeatures']:
            idx = titles.index(e['title'])
            score = 0
            if e['isAvailable']:
                score = 1
            if e['isFreeOfCharge']:
                score = 2
                
            res['ci_'+keys[idx]].append( score )
    
    res = pd.DataFrame(res)
    res.to_csv( BASE_PATH + INFO_FOLDER + 'crawl_ci.csv' )

def collect_rating():
    items_done = get_processed( BASE_PATH + RATING_FOLDER )
    
    aspects = {'service', 'cleanliness', 'hotel_condition', 'room', 'location', 'comfort', 'facilities', 'value_for_money', 'food_and_drinks', 'breakfast'}
    
    res = {}
    res['item_id'] = []
    res['ri_rating_index'] = []
    res['ri_rating_percentage'] = []
    res['ri_rating_count'] = []
    res['ri_rating_advertiser'] = []
    res['ri_tested'] = []
    res['ri_awards'] = []
    for a in aspects:
        res['ri_'+a] = []
    
#     all_aspects = set()
    
    for k,v in items_done.items():
        
        res['item_id'].append( k )
        res['ri_rating_index'].append(  v['ratingSummary']['reviewRating']['index'] )
        res['ri_rating_percentage'].append(  v['ratingSummary']['reviewRating']['percentage'] )
        res['ri_rating_count'].append(  v['ratingSummary']['partnerReviewCount'] )
        res['ri_rating_advertiser'].append(  v['ratingSummary']['hasAdvertiserRatings'] )
        res['ri_tested'].append(  v['ratingSummary']['hasQualityTests'] )
        res['ri_awards'].append(  len( v['ratingSummary']['hotelAwards'] ) )
        aspectsdone = set()
        
        for e in v['ratingSummary']['ratingAspects']:
            aspectsdone.add( e['type'])
            res['ri_'+e['type']].append( e['percentage'] )
            
        for aspect in ( aspects - aspectsdone ):
            res['ri_'+aspect].append( np.nan )
        
#         if  len(v['ratingSummary']['ratingAspects']) != 0:
#             #pprint.pprint( items_done[k] )     
#             if len( v['ratingSummary']['ratingAspects'] ) == len(aspects):
#                 pprint.pprint( v )
#             for e in v['ratingSummary']['ratingAspects']:
#                 all_aspects.add( e['type'])
#                 
#     print( all_aspects )
    
    res = pd.DataFrame(res)
    res.to_csv( BASE_PATH + RATING_FOLDER + 'crawl_ri.csv' )

def collect_images():
    items_done = get_processed( BASE_PATH + IMAGES_FOLDER )
        
    res = {}
    res['item_id'] = []
    res['ii_count'] = []
    res['ii_partner_count'] = []
    res['ii_partner_set'] = []
    res['ii_partner_max'] = []
    
#     all_aspects = set()
    
    partnerset = set()
    
    for k,v in items_done.items():
        
        res['item_id'].append( k )
        res['ii_count'].append( len( v['images'] ) )
        
        partners = {}
        partners_count = 0
        max_count = 0
        max_partner = None
        for img in v['images']:
            if img['partnerName'] != '':
                if not img['partnerName'] in partners:
                    partners[ img['partnerName'] ] = 0
                partners[ img['partnerName'] ] += 1
                
                if partners[ img['partnerName'] ] > max_count:
                    max_partner = img['partnerName']
                    max_count = partners[ img['partnerName'] ]
                
                partners_count += 1
                partnerset.add( img['partnerName'] )
        
        res['ii_partner_count'].append( partners_count )
        res['ii_partner_set'].append( partners )
        res['ii_partner_max'].append( max_partner )
    
    res = pd.DataFrame(res)
    res.to_hdf( BASE_PATH + IMAGES_FOLDER + 'crawl_ii.hd5', key='table' )
    
    del res['ii_partner_set']
    res['ii_partner_max'] = res['ii_partner_max'].astype( 'category' ).cat.codes
    
    res.to_csv( BASE_PATH + IMAGES_FOLDER + 'crawl_ii.csv' )
    
def crawl( folder, url ):
    
    items = get_items( BASE_PATH + RAW_FOLDER )
    items_done = get_processed( folder )
    
    items = list( filter( lambda x: x not in items_done, items ) )
    
    result_map = {}
    
    first = items[0]
    
    togo = len(items)
    tstart = time.time()
        
    try:
        
        retries = RETRY
        problem = False
        dump = DUMP_AFTER
        
        i = 0
        
        while not problem:
            
            if i == len(items):
                break
            
            try:
            
                item = int(items[i])
                result_map[item] = get_item( url, item )
                #time.sleep(1)
                i += 1
                retries = RETRY
                togo -= 1 
                dump -= 1
                
                if dump == 0:
                    ensure_dir(folder)
                    pickle.dump( result_map, open( folder + 'from_'+str(first)+'.pkl', 'wb' ) )
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
                    
                    print( 'done {} of {} in {}, {} left'.format( done, len(items), spent, eta ) )
                
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
        ensure_dir(folder)
        pickle.dump( result_map, open( folder + 'from_'+str(first)+'.pkl', 'wb' ) )
    
    ensure_dir(folder)
    pickle.dump( result_map, open( folder + 'from_'+str(first)+'.pkl', 'wb' ) )

def get_item( url, idx ):
    
    result = requests.get( url.replace( ':ID:', str(int(idx)) ) )

    resultr = None
        
    if result.status_code == 200:
        resultr = result.json()

    return resultr
    

def get_processed( folder ):
    
    items = dict()
        
    for file_name in glob( folder + '*.pkl' ): 
        res_part = pickle.load( open( file_name, 'rb' ) )
        for k,v in res_part.items():
            items[k] = v

    return items

def get_items( folder ):
    
    tstart = time.time()
    
    train = pd.read_csv( folder + 'train.csv' )
    train['item_id'] = pd.to_numeric( train.reference, errors='coerce' )
    train_items_interact = set(train[~train.item_id.isnull()].item_id.unique())
    train_items_view = set()
    train[train.action_type=='clickout item'].impressions.apply( split_pipe, collect_unique=train_items_view, convert=int )

    train_items = train_items_interact | train_items_view
    
    print( 'items train ', len( train_items ) )
    print( 'time ', (time.time() - tstart))
    
    del train
    
    test = pd.read_csv( folder + 'test.csv' )
    test['item_id'] = pd.to_numeric( test.reference, errors='coerce' )
    test_items_interact = set(test[~test.item_id.isnull()].item_id.unique())
    test_items_view = set()
    test[test.action_type=='clickout item'].impressions.apply( split_pipe, collect_unique=test_items_view, convert=int )
    
    test_items = test_items_interact | test_items_view
    
    print( 'items test ',len( test_items ) )
    print( ' -- new test ', len( test_items - train_items ) )
    print( 'time ', (time.time() - tstart))
    
    del test
    
    items = test_items | train_items
    
    meta = pd.read_csv( folder + 'item_metadata.csv' )
    meta_items = set(meta[~meta.item_id.isnull()].item_id.unique())
    
    print( 'items meta ',len( meta_items ) )
    print( ' -- new meta ', len( meta_items - items ) )
    print( ' -- not meta ', len( items - meta_items ) )
    
    items = items | meta_items
    items = list(sorted(items))
    
    return items[:50]
    
if __name__ == '__main__':
    main()
    
    