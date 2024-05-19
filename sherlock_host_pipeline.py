
# import imp
import lasair
import pandas as pd 
import os
import re
import json



token = 'XXXXXXXXXXXXXXX'

def get_potential_host(obj, ra, dec, ori_df_path):
    print(obj)
    L = lasair.lasair_client(token)
    c = L.sherlock_position(ra, dec)
    crossmatches = c["crossmatches"]
    if crossmatches is not None:
        print(crossmatches)
        top_host = crossmatches[0]
        top_host['object_id'] = obj
        top = pd.DataFrame.from_dict(top_host)
        original_df = pd.read_csv(ori_df_path)
        if obj not in original_df['object_id'].tolist():
            df = pd.concat([original_df, top], ignore_index=True)
            df.to_csv('test.csv')
        top_ra, top_dec = top_host['raDeg'], top_host['decDeg']
    else:
        top_ra, top_dec = None, None

    return top_ra, top_dec

def get_multiple_hosts(table, ori_df_path):
    df = pd.read_csv(ori_df_path)
    L = lasair.lasair_client(token)
    for ztf_id, ra, dec in zip(table['ztf_id'], table['ra'], table['dec']):
        if ztf_id in df['object_id'].tolist():
            print('object %s\'s host is already recorded!\n'% ztf_id)
            continue
        else:
            c = L.sherlock_position(ra, dec)  
            crossmatches = c["crossmatches"]
            if len(crossmatches) >= 1:
                top_host = crossmatches[0]
                top_host['object_id'] = ztf_id
                top = pd.DataFrame.from_dict([top_host])
                df = pd.concat([df, top], ignore_index=True)
                print('object %s\'s host is added!\n'% ztf_id)
            else:
                print('object %s\'s host is not found!\n'% ztf_id)
    df.to_csv(ori_df_path)



 
    