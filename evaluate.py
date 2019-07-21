'''
Created on Mar 19, 2019

@author: malte
'''

from config.globals import BASE_PATH
from helper.loader import load_csv, load_truth
import numpy as np
import pandas as pd


SET = 'sample/'

def main():
    
    algs = {}
    
    algs['lgbm_ltr_noshfl_val0.1-500_new'] = []
    
    #load data
    truth = load_truth( BASE_PATH + SET )
    
    all = pd.DataFrame()
    mrrs = pd.DataFrame()
    
    for key, alg in algs.items():
        
        mrr = evaluate(key, base=BASE_PATH, dataset=SET, truth=truth, all=all)
        mrrs = pd.concat( [mrrs,mrr] )
    
    tmp = pd.DataFrame()
    tmp['mrr@A'] = [ len(all) ]
    tmp['mrr@5'] = [ len(all) ]
    tmp['mrr@10'] = [ len(all) ]
    tmp['mrr@step<=1'] = [ len( all[all.step <= 1]) ]
    tmp['mrr@step<=2'] = [ len( all[all.step <= 2] ) ]
    tmp['mrr@step<=3'] = [ len( all[all.step <= 3] ) ]
    tmp['mrr@step<=5'] = [ len( all[all.step <= 5] ) ]
    tmp['mrr@step>5'] = [ len( all[all.step > 5] ) ]
    tmp['mrr@step>10'] = [ len( all[all.step > 10] ) ]
    tmp['mrr@step>20'] = [ len( all[all.step > 20] ) ]
    if 'with_item' in all.columns:
        tmp['mrr@wi'] = [ len( all[all.with_item] ) ]
        tmp['mrr@woi'] = [ len( all[~all.with_item] ) ]
    
    mrrs = pd.concat( [mrrs,tmp], sort=False )
    mrrs.to_csv('eval.csv')
    
#     print( mrrs.T )
#     print( all[all['ffnn2000f4_noshfl_adam_optadam'] > all['lgbmcv_bin_noshfl_val5-100__mean']][[key for key, algs in algs.items()] ] )
#     print( len(all['ffnn2000f4_noshfl_adam_optadam'] > all['lgbmcv_bin_noshfl_val5-100__mean']) )
#     print( len(all) )
#     print( all[['lgbmcv_bin_noshfl_val5-100__mean','ffnn2000f4_noshfl_adam_optadam']].max(axis=1).mean() )


def evaluate( key, base=BASE_PATH, dataset=SET, truth=None, all=None ):
    
    if truth is None:
        truth = load_csv( base + dataset + 'truth.csv' ).sort_values(['session_id'])
    
    if type(key) is pd.DataFrame:
        solution = key
    else:
        solution = load_csv( base + dataset + 'solution_' + key + '.csv' ).sort_values(['session_id'])
        solution['recommendations'] = solution['recommendations'].apply( lambda x: eval(x.replace('\n','').replace('\r','').replace(' ',',').replace(',,',',').replace(',,',',').replace(',,',',').replace('[,','[')) )
        solution['confidences'] = solution['confidences'].apply( lambda x: eval(x.replace('\n','').replace('\r','').replace(' ',',').replace(',,',',').replace(',,',',').replace(',,',',').replace(',,',',').replace('[,','[')) )

    both = truth.merge( solution[['session_id','step','recommendations','confidences']], on=['session_id','step'], how='left' )
    both['score'] = both.apply(get_reciprocal_ranks, axis=1)
    
    if all is not None:
        all['session_id'] = truth['session_id']
        all['step'] = truth['step']
        all['reference'] = truth['reference']
        if 'with_item' in truth.columns:
            all['with_item'] = truth['with_item']
        all[key] = both['score']
        all[key + '_conf'] = both['confidences'].apply( lambda x: x[0] )
    
    res = dict()
    res['algorithm'] = [key]
    
    res['mrr@A'] = [ both.score.mean() ]
    res['mrr@5'] = [ both[both.score >= 0.2 ].score.sum() / len(both) ]
    res['mrr@10'] = [ both[both.score >= 0.1 ].score.sum() / len(both) ]
    res['mrr@step<=1'] = [ both[both.step <= 1].score.mean() ]
    res['mrr@step<=2'] = [ both[both.step <= 2].score.mean() ]
    res['mrr@step<=3'] = [ both[both.step <= 3].score.mean() ]
    res['mrr@step<=5'] = [ both[both.step <= 5].score.mean() ]
    res['mrr@step>5'] = [ both[both.step > 5].score.mean() ]
    res['mrr@step>10'] = [ both[both.step > 10].score.mean() ]
    res['mrr@step>20'] = [ both[both.step > 20].score.mean() ]
    if 'with_item' in both.columns:
        res['mrr@wi'] = [ both[both.with_item].score.mean() ]
        res['mrr@woi'] = [ both[~both.with_item].score.mean() ]
    
    res = pd.DataFrame(res)

    return res

def get_reciprocal_ranks(ps):
    """Calculate reciprocal ranks for recommendations."""
    mask = ps.reference == np.array(ps.recommendations)

    if mask.sum() == 1:
        rranks = generate_rranks_range(0, len(ps.recommendations))
        return np.array(rranks)[mask].min()
    else:
        return 0.0

def generate_rranks_range(start, end):
    """Generate reciprocal ranks for a given list length."""

    return 1.0 / (np.arange(start, end) + 1)

if __name__ == '__main__':
    main()