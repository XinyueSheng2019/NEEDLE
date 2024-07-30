
# import imp
import lasair
import pandas as pd 
import os
import re
import json
from ztf_mag_pipeline import get_json 



token = 'XXXXX'

def get_potential_host_from_json(obj, mag_path):
    obj_mag_path = mag_path + '/' + str(obj) + '.json'
    if os.path.exists(obj_mag_path):
        f = open(obj_mag_path)
        om = json.load(f)
        f.close()
    else:
        om = get_json(obj, mag_path)

    if 'sherlock' in om.keys():
        raDeg, decDeg = om['sherlock']['raDeg'], om['sherlock']['decDeg']
        return raDeg, decDeg
    else:
        return None, None
        


    

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




if __name__ == '__main__':
    full_obj_list = '../../../data/full_obj_info.csv'
    full_objs = pd.read_csv(full_obj_list)
    sherlock_csv = '../../../data/ztf_sherlock_matches/ztf_sherlock_host.csv'
    get_multiple_hosts(full_objs, sherlock_csv)



    # get_potential_host(obj = 'ZTF20aauwjla', ra = 16.851866, dec = 34.53307, ori_df_path = sherlock_csv)

    
    