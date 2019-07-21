'''
Created on May 24, 2019

@author: malte
'''
from pathlib import Path

from config.globals import BASE_PATH
from domain.action_types import CLICK, IMAGE, INFO, DEALS, RATING
from helper.df_ops import copy_features, reduce_mem_usage
from helper.loader import load_hdfs, write_feather, load_feather
import numpy as np
import pandas as pd
import time
from featuregen.popularity import print_col_list


TEST_FOLDER = BASE_PATH + 'competition/'
META_FOLDER = BASE_PATH + 'preprocessed/'

ACTION_MAP = {}
ACTION_MAP['all'] = [CLICK,IMAGE,INFO,DEALS,RATING]
ACTION_MAP['view'] = [IMAGE,INFO,DEALS,RATING]
ACTION_MAP['click'] = [CLICK]

def main():
    log = load_hdfs( TEST_FOLDER + 'data_log.hd5' )
    examples = load_hdfs( TEST_FOLDER + 'data_examples.hd5' )
    meta_features( TEST_FOLDER, META_FOLDER, log, examples, latent=None, redo=False )

def meta_features(base_path, meta_path, log, examples, latent='d2v', latent_size=16, redo=False):
    
    name = 'meta_features'
    if latent == None:
        name += '_all'
    else:
        name += '_' +str(latent_size)
    
    path = Path( base_path + 'features/' + name + '.fthr' )
    if path.is_file() and not redo:
        features = load_feather( path )
        features = features[features.session_id.isin( examples.session_id.unique() )]
        examples = copy_features( examples, features )
    else:
        examples, cols = create_features( meta_path, log, examples, latent_prefix=latent, latent_size=latent_size )
        examples = reduce_mem_usage(examples)
        write_feather( examples[['session_id','impressions'] + list(cols)], path )
        #examples[['session_id','impressions','prices','label'] + list(cols)].to_csv( base_path + 'features/' + name + '.csv' )
        print_col_list(cols)
        
    return examples

def create_features( meta_path, log, examples, latent_prefix='d2v', latent_size=32  ):
    
    tstart = time.time()
    print( 'create_features crawl' )
    
    cols_pre = examples.columns.values
    
    if latent_prefix is not None:
        #pure meta
        use_from_meta = PROPERTIES_MOST_FILTERED + ['features']
        examples = add_from_file( meta_path + 'item_metadata.csv', examples, to=['impressions'], filter=use_from_meta )
        
        #latent d2v
        LATENT_ITEM = [ 'if_'+str(i) for i in range(latent_size) ]
        LATENT_NEW = [ latent_prefix+'_'+x for x in LATENT_ITEM ]    
        path = meta_path + latent_prefix + '_item_features.' + str(latent_size) + '.csv'
        examples = add_from_file( path, examples, to=['impressions'], filter=LATENT_ITEM, rename=LATENT_NEW, index=False )
        
        new_cols = np.setdiff1d(examples.columns.values, cols_pre)
        
        examples = fill_na(examples, new_cols, 0)
    else:
        #pure meta
        use_from_meta = PROPERTIES_ALL + ['features']
        examples = add_from_file( meta_path + 'item_metadata.csv', examples, to=['impressions'], filter=use_from_meta )
        
        new_cols = np.setdiff1d(examples.columns.values, cols_pre)
        
        examples = fill_na(examples, new_cols, 0)
        
    print( 'create_features pop in {}s'.format( (time.time() - tstart) ) )
    
    return examples, new_cols
    
def add_from_file( file, examples, col=['item_id'], to=None, filter=None, rename=None, index=True ):
    
    tstart = time.time()
    print( '\t add_from_file {}'.format(file) )
    
    keep = False
    if to is None:
        to = col
        keep = True
    
    toadd = pd.read_csv( file, index_col=0 if index else None )
    if filter is not None:
        toadd = toadd[col+filter]
    
    if rename is not None:
        for i,c in enumerate(filter):
            toadd[rename[i]] = toadd[filter[i]]
            del toadd[c]
    
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

def fill_na( examples, cols, value ):
    for c in cols:
        examples[c] = examples[c].fillna(value)
    return examples

PROPERTIES_MOST_FILTERED = [
    'Hotel', 
    '5 Star', 
    'Resort', 
    '4 Star', 
    'Hostal (ES)', 
    'Motel', 
    '3 Star', 
    'Free WiFi (Combined)', 
    'Very Good Rating', 
    'Excellent Rating', 
    'House / Apartment', 
    'Swimming Pool (Combined Filter)', 
    'Car Park', 
    'Good Rating', 
    'Serviced Apartment', 
    'Air Conditioning', 
    'Spa (Wellness Facility)', 
    '2 Star', 
    'Pet Friendly', 
    'Satisfactory Rating', 
    'All Inclusive (Upon Inquiry)', 
    'Restaurant', 
    '1 Star', 
    'Reception (24/7)', 
    'Bed & Breakfast', 
    'Guest House', 
    'Beach', 
    'Jacuzzi (Hotel)', 
    'Swimming Pool (Indoor)', 
    'Pousada (BR)', 
    'Hostel', 
    'Non-Smoking Rooms', 
    'Airport Shuttle', 
    'Bathtub', 
    'Balcony', 
    'Free WiFi (Rooms)', 
    'Swimming Pool (Outdoor)', 
    'Spa Hotel', 
    'Sauna', 
    'Luxury Hotel', 
    'Gym', 
    'Direct beach access', 
    'Room Service', 
    'Health Retreat', 
    'Romantic', 
    'Accessible Parking', 
    'Shower', 
    'Gay-friendly', 
    'Family Friendly', 
    'Wheelchair Accessible', 
    'Self Catering', 
    'Airport Hotel', 
    'Terrace (Hotel)', 
    'Camping Site', 
    'Playground', 
    'Accessible Hotel', 
    'Bungalows', 
    'Casa Rural (ES)', 
    'Hotel Bar'
    ]

PROPERTIES_ALL = [
    'Country Hotel',
    'Theme Hotel',
    'Hostal (ES)',
    'Openable Windows',
    'Design Hotel',
    'Camping Site',
    'Swimming Pool (Indoor)',
    '4 Star',
    'Doctor On-Site',
    'Self Catering',
    'Farmstay',
    'Swimming Pool (Bar)',
    'Radio',
    'Body Treatments',
    'Spa Hotel',
    'Skiing',
    'Casa Rural (ES)',
    'Cot',
    'Reception (24/7)',
    'From 3 Stars',
    'Free WiFi (Rooms)',
    'From 2 Stars',
    'Balcony',
    'Eco-Friendly hotel',
    'Cosmetic Mirror',
    'Large Groups',
    'Room Service (24/7)',
    'Pool Table',
    'Cable TV',
    'Minigolf',
    'Organised Activities',
    'Bathtub',
    'Excellent Rating',
    'Family Friendly',
    'WiFi (Rooms)',
    'Horse Riding',
    'Shower',
    'On-Site Boutique Shopping',
    'Terrace (Hotel)',
    'Massage',
    'Playground',
    'Car Park',
    'Airport Shuttle',
    'Hairdresser',
    'Good Rating',
    'Boutique Hotel',
    'Pousada (BR)',
    'Television',
    'Jacuzzi (Hotel)',
    '1 Star',
    'Direct beach access',
    'Accessible Parking',
    'Bowling',
    'Gay-friendly',
    'Bungalows',
    'Steam Room',
    'Restaurant',
    'Convenience Store',
    'Ironing Board',
    'Hiking Trail',
    'Halal Food',
    'Laundry Service',
    'Non-Smoking Rooms',
    'Swimming Pool (Outdoor)',
    'Tennis Court',
    'Lift',
    'Nightclub',
    'Volleyball',
    'House / Apartment',
    'Very Good Rating',
    'Hotel Bar',
    'Swimming Pool (Combined Filter)',
    'Convention Hotel',
    'Spa (Wellness Facility)',
    'Business Hotel',
    'Table Tennis',
    'Fridge',
    'Hypoallergenic Bedding',
    'Childcare',
    'Romantic',
    'Beach',
    'Wheelchair Accessible',
    'Hammam',
    'Airport Hotel',
    'Concierge',
    'Motel',
    '5 Star',
    'Business Centre',
    'Ski Resort',
    'Desk',
    'Hot Stone Massage',
    'Towels',
    'Kosher Food',
    'Hairdryer',
    'Beauty Salon',
    'Sun Umbrellas',
    'Hostel',
    'Porter',
    'Guest House',
    'Fitness',
    'From 4 Stars',
    'Fan',
    'Accessible Hotel',
    'Luxury Hotel',
    'Free WiFi (Combined)',
    'Surfing',
    'Diving',
    'Club Hotel',
    'Satisfactory Rating',
    'Resort',
    '3 Star',
    'Golf Course',
    'Computer with Internet',
    'Sailing',
    'Room Service',
    'Szep Kartya',
    'Washing Machine',
    'Bed & Breakfast',
    'Water Slide',
    'Conference Rooms',
    'Pet Friendly',
    'Honeymoon',
    'Sauna',
    'Solarium',
    'Adults Only',
    'Free WiFi (Public Areas)',
    'Hypoallergenic Rooms',
    'Casino (Hotel)',
    'Health Retreat',
    '2 Star',
    "Kids' Club",
    'Teleprinter',
    'Electric Kettle',
    'Safe (Hotel)',
    'Central Heating',
    'Shooting Sports',
    'Senior Travellers',
    'Satellite TV',
    'Deck Chairs',
    'Safe (Rooms)',
    'Beach Bar',
    'Hydrotherapy',
    'Boat Rental',
    'Air Conditioning',
    'Bike Rental',
    'WiFi (Public Areas)',
    'Telephone',
    'Microwave',
    'Sitting Area (Rooms)',
    'Hotel',
    'Tennis Court (Indoor)',
    'Express Check-In / Check-Out',
    'Singles',
    'Flatscreen TV',
    'Serviced Apartment',
    'Gym',
    'All Inclusive (Upon Inquiry)',
    ]

if __name__ == '__main__':
    main()
    