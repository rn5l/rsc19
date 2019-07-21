'''
Created on May 15, 2019

@author: malte
'''
from helper.loader import load_maps, load_csv, ensure_dir
from config.globals import BASE_PATH

#PATH
RAW = 'raw/'
PREPROCESS = 'preprocessed/'
SET = 'competition/'
OUT = 'final/'

KEY = 'lgbmcv_ltr_noshfl_split5-es500_dart0.1'

def main():
    
    keep = ['user_id','session_id','timestamp','step','item_recommendations']
    
    solution = load_csv( BASE_PATH + SET + 'solution_' + KEY + '.csv' ).sort_values(['session_id'])
    solution['recommendations'] = solution['recommendations'].apply( lambda x: eval(x.replace('\n','').replace('\r','').replace(' ',',').replace(',,',',').replace(',,',',').replace(',,',',').replace('[,','[')) )
    solution['confidences'] = solution['confidences'].apply( lambda x: eval(x.replace('\n','').replace('\r','').replace(' ',',').replace(',,',',').replace(',,',',').replace(',,',',').replace(',,',',').replace('[,','[')) )

    example = load_csv( BASE_PATH + RAW + 'submission_popular.csv' )
    umap, smap = load_maps( BASE_PATH + PREPROCESS )
    
    solution['item_recommendations'] = solution['recommendations'] #.apply( lambda x: eval(x.replace('\n','').replace('\r','').replace(' ',',').replace(',,',',').replace(',,',',').replace(',,',',').replace('[,','[')) )
    del solution['recommendations']
    
    def okay(s):
        if type(s) == int:
            print(s)
        if type(s) == str:
            print(s)
        return ' '.join( map( str, s ) )
    print(solution['item_recommendations'])
    solution['item_recommendations'] = solution['item_recommendations'].apply( okay )
    print(solution['item_recommendations'])
    solution['user_id'] = solution['user_id'].apply( lambda x: umap[x] )
    solution['session_id'] = solution['session_id'].apply( lambda x: smap[x] )
    solution = solution[keep]
    
    if check( solution, example ):
        ensure_dir( BASE_PATH + OUT )
        solution.to_csv( BASE_PATH + OUT + 'sub_'+KEY+'.csv' )
    
def check( solution, example ):
    
    res = True
    
    if len( solution) != len(example):
        print( 'different length...' )
        res = False
    
    ssess =set( solution['user_id'].unique() )
    susers = set( solution['session_id'].unique() )
    
    esess =set( example['user_id'].unique() )
    eusers = set( example['session_id'].unique() )
    
    if len( eusers - susers ) > 0:
        print( 'not enough users...' )
        print( eusers - susers )
        res = False
        
    if len( esess - ssess ) > 0:
        print( 'not enough sess...' )
        print( esess - ssess )
        res = False
    
    return res

if __name__ == '__main__':
    main()

