'''
Created on Apr 8, 2019

@author: malte
'''
import numpy as np

def split_pipe(val, collect_unique=None, collect_list=None, convert=int):
    if val is not None:
        result = [convert(x) for x in val.split('|')] if type(val) is str else val
        if collect_unique is not None:
            for i in result:
                collect_unique.add(i)
        if collect_list is not None:
            for i in result:
                collect_list.append(i)
        return result
    return None
    
def join_pipe(val):
    if val is not None:
        result = '|'.join([str(x) for x in val])
        return result
    return None

def map_list(val, mapper=None):
    if val is not None:
        #print(val)
        #print(type(val))
        if type(val) is list:
            return [mapper[x] for x in val]
        return mapper[val]
    return None

def change_type( val, function=list):
    if val is not None:
        return function(val)
    return None


