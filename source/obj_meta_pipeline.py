'''
Make a csv file for each object in g and r band
filename: ztfid_meta.csv
header:
'candi_id', 
'candi_mag', 'candi_magerr',  'prev_delta_mag', 'ps_r_mag' , 'ps_r_magerr', 'ps_delta_mag',
'disc_mjd', 'obs_mjd', 'delta_t', 'prev_delta_t'

Further consideration:
'image_credit'
'disc_mag', 'disc_magerr': the discovery date might belong to other bands but r, the mag could be tricky to substract.


method:
1. image_meta.json: 
    get 'dis_mjd' among all bands
    get 'candi_id' in r band
2. go to each candi folder to check the mag_info.json:
    get 'obs_mjd'; 'candi_mag'; 'candi_magerr'; 
3. go to Panstarr folder to get the txt file for this object:
    get 'ps_r_mag' , 'ps_r_magerr'
4. build a dataframe for existed info
5. substract to get 'prev_delta_mag', 'ps_delta_mag', 'delta_t', 'prev_delta_t'
'''


import pandas as pd 
import numpy as np 
import json
import os
import re
from cal_extinction import ext




def get_host_mag(host_path, obj, band):
    if os.path.exists(host_path+'/'+obj+'.csv'):
        host = pd.read_csv(host_path+'/'+obj+'.csv')
        row = host.iloc[0]
        if band == 'f1':
            return row['gAp']
        else:
            return row['rAp']
    else:
        return np.nan
    
def collect_meta(obj, objs_path, host_path, add_ext = True):
    '''
    
    meta_df = pd.DataFrame(columns = ['disc_mjd', 'disc_mjd_r', 'disc_mag_r', 
                                    'ps_r_mag' , 'ps_r_magerr', 
                                    'candi_id', 'candi_mjd', 'candi_mag', 'candi_magerr',  '
                                    'ps_delta_mag', 'delta_t', 'prev_delta_t', 'prev_delta_mag'])

    '''
    
    meta_path = os.path.join(objs_path, obj + '/mag_with_img.json')
    # get discovery date and candidate with mag data
    j = open(meta_path, 'r')
    image_meta = json.loads(j.read())

    candi_file_num = []
    candi_idx_l = []
    candi_mjd_l = []
    candi_mag_l = []
    candi_magerr_l = []
    ps_delta_mag_l = []
    delta_t_l = []
    prev_delta_t_l = []
    prev_delta_mag_l = []
    delta_mag_l = []
    band_l = []
    host_mag_l = []
    disc_mjd_l = []
    disc_mag_l = []
    disc_magerr_l = []
    ra, dec = image_meta['ra'], image_meta['dec']
    
    disdate = image_meta['disdate'] - 2400000.5
    delta_t_disc_l = []
    if add_ext:
        exts = ext(ra, dec)

    for f in ['f1', 'f2']:
        fn = 'g' if f == 'f1' else 'r'
        candi_list = image_meta["candidates_with_image"][f]
        candi_list = [x for x in candi_list if 'candid' in x.keys()]
        disc_mag, disc_magerr = None, None
        disc_mjd = None # for each band


        if len(candi_list)>=1:  
            host_mag = get_host_mag(host_path, obj, f)

            candi_list = sorted(candi_list, key = lambda d:d['mjd'])
            flag = False
            # print(candi_list)

            for cl in candi_list:
                band_l.append(f)
                candi_file_num.append(cl['filefracday'])
                candi_idx_l.append(cl['candid'])
                host_mag_l.append(host_mag)
                if add_ext:
                    obs_mjd, candi_mag, candi_magerr = cl['mjd'], cl['magpsf'] - exts[f'ZTF_{fn}'], cl['sigmapsf']
                else:
                    obs_mjd, candi_mag, candi_magerr = cl['mjd'], cl['magpsf'], cl['sigmapsf']

                if flag == False: # first candidate
                    prev_delta_t_l.append(0.0)
                    prev_delta_mag_l.append(0.0)
                    disc_mag = candi_mag
                    disc_magerr = candi_magerr
                    disc_mjd = obs_mjd
                    flag = True
                else:
                    if obs_mjd - candi_mjd_l[-1]>= 1:
                        prev_delta_t_l.append(round(obs_mjd - candi_mjd_l[-1], 5))
                        prev_delta_mag_l.append(round(candi_mag - candi_mag_l[-1], 5))
                    else:
                        if len(candi_mjd_l) >= 2 and obs_mjd - candi_mjd_l[-2] >= 1:
                            prev_delta_t_l.append(round(obs_mjd - candi_mjd_l[-2], 5))
                            prev_delta_mag_l.append(round(candi_mag - candi_mag_l[-2], 5))
                        elif len(candi_mjd_l) >= 3 and obs_mjd - candi_mjd_l[-3] >= 1:
                            prev_delta_t_l.append(round(obs_mjd - candi_mjd_l[-3], 5))
                            prev_delta_mag_l.append(round(candi_mag - candi_mag_l[-3], 5))
                        else:
                            prev_delta_t_l.append(round(obs_mjd - disc_mjd, 5))
                            prev_delta_mag_l.append(round(candi_mag - disc_mag, 5))

                candi_mjd_l.append(obs_mjd)
                candi_mag_l.append(candi_mag)
                candi_magerr_l.append(candi_magerr)
                delta_t_l.append(round(obs_mjd - disc_mjd, 5))
                delta_mag_l.append(round(candi_mag - disc_mag, 5))
                disc_mjd_l.append(disc_mjd)
                disc_mag_l.append(disc_mag)
                disc_magerr_l.append(disc_magerr)
                delta_t_disc_l.append(round(obs_mjd - disdate, 5))

                if host_mag is not None:
                    ps_delta_mag_l.append(round(candi_mag - host_mag, 5))
                else:
                    ps_delta_mag_l.append(None) 

    obj_dict = {
        'filefracday': candi_file_num,
        'candid': candi_idx_l,
        'filter': band_l,
        'candi_mjd': candi_mjd_l,
        'candi_mag': candi_mag_l,
        'candi_magerr': candi_magerr_l,
        'disc_mjd_band': disc_mjd_l,
        'disc_mag': disc_mag_l,
        'disc_magerr': disc_magerr_l,
        'host_mag': host_mag_l,
        'delta_host_mag': ps_delta_mag_l, 
        'delta_t_discovery_band': delta_t_l, 
        'delta_t_discovery': delta_t_disc_l,
        'delta_t_recent': prev_delta_t_l, 
        'delta_mag_discovery': delta_mag_l,
        'delta_mag_recent': prev_delta_mag_l
    }

    df = pd.DataFrame(obj_dict)
    df = df.drop_duplicates(subset = ['candi_mjd'])
    if add_ext:
        df.to_csv(objs_path + '/'+obj + '/obj_meta4ML_ext.csv')
    else:
        df.to_csv(objs_path + '/'+obj + '/obj_meta4ML.csv')
    


if __name__ == '__main__':

    obj_re = re.compile('ZTF')
    add_ext = True
    
    objs_path = '/Users/xinyuesheng/Documents/astro_projects/data/untouch_testset/images'
    host_path = '/Users/xinyuesheng/Documents/astro_projects/data/untouch_testset/hosts'
    if add_ext:
        host_path += '_ext'
    

    file_names = os.listdir(objs_path) 
    file_names = list(filter(obj_re.match, file_names))
    for obj in file_names:
        collect_meta(obj, objs_path, host_path, add_ext)


