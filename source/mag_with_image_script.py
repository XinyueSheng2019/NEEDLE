import json
import os
import re

image_path = '../../data/image_sets_v3'
obj_re = re.compile('ZTF')
file_names = [file for file in os.listdir(image_path) if obj_re.match(file)]

for obj in file_names:
    obj_path = os.path.join(image_path, obj)
    with open(os.path.join(obj_path, 'image_meta.json'), 'r') as j:
        meta = json.load(j)
    
    mag_with_img_dict = {
        'id': meta['id'],
        'label': meta['label'],
        'ra': meta['ra'],
        'dec': meta['dec'],
        'disdate': meta['disdate'],
        'candidates_with_image': {'f1': [], 'f2': [], 'f3': []}
    }
    
    for f in ['1', '2', '3']:
        if 'withMag' in meta[f].keys():
            mag_records = meta[f]['withMag']
            for mr in mag_records:
                obs_with_mag = os.path.join(obj_path, f, mr, 'mag_info.json')
                with open(obs_with_mag, 'r') as m:
                    mag_info = json.load(m)
                    mag_info['filefracday'] = mr
                    mag_with_img_dict['candidates_with_image']['f' + f].append(mag_info)

    with open(os.path.join(obj_path, 'mag_with_img.json'), 'w') as outfile:
        json.dump(mag_with_img_dict, outfile, indent=4)
