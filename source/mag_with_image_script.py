import json

import os
import re


image_path = '../../data/image_sets_v3'
obj_re = re.compile('ZTF')
file_names = os.listdir(image_path) 
file_names = list(filter(obj_re.match, file_names))


for obj in file_names:
    obj_path = os.path.join(image_path, obj)
    j = open(obj_path+'/image_meta.json', 'r')
    meta = json.loads(j.read())
    mag_with_img_dict = {}
    mag_with_img_dict['id'] = meta['id']
    mag_with_img_dict['label'] = meta['label']
    mag_with_img_dict['ra'] = meta['ra']
    mag_with_img_dict['dec'] = meta['dec']
    mag_with_img_dict['disdate'] = meta['disdate']
    mag_with_img_dict['candidates_with_image'] = {'f1':[], 'f2':[], 'f3':[]}
    
    for f in ['1','2','3']:
        if 'withMag' in meta['f'+f].keys():
            mag_records = meta['f'+f]['withMag']
            for mr in mag_records:
                obs_with_mag = obj_path+'/'+f+'/'+mr+'/mag_info.json'
                m = open(obs_with_mag,'r')
                mag_info = json.loads(m.read())
                mag_info['filefracday'] = mr
                mag_with_img_dict['candidates_with_image']['f'+f].append(mag_info)
                m.close()
    j.close()


    with open(obj_path+ '/mag_with_img.json','w') as outfile:
            json.dump(mag_with_img_dict, outfile, indent=4)



    


