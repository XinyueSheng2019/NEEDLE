#! /usr/bin/env python

from logging import raiseExceptions
import pandas as pd 
import numpy as np
import os
import json
from multiprocessing import Pool,cpu_count
import lasair


token = 'XXXXXXXXXXXX'

def get_json(ztf_id, path):
	save_path = path + '/' + str(ztf_id) + '.json'
	if not os.path.exists(save_path):
		L = lasair.lasair_client(token)
		c = L.objects([ztf_id])[0]
		try: # remove non-detections
			temp_list = []
			for cd in c['candidates']:
				if 'candid' in cd.keys():
					temp_list.append(cd)
			c['candidates'] = temp_list
		except:
			print(c)
		
		json_object = json.dumps(c, indent=4)

		outfile = open(save_path, "w") # Writing to sample.json
		outfile.write(json_object)	
		outfile.close()

	# return c

# if __name__ == '__main__':


# 	image_path = '/Users/xinyuesheng/Documents/astro_projects/data/image_sets_v3'
# 	img_files = os.listdir(image_path)
# 	img_files.remove('.DS_Store')
# 	img_files.remove('readme.md')
	
# 	path = '/Users/xinyuesheng/Documents/astro_projects/data/mag_sets_v4'
# 	if not os.path.exists(path):
# 		os.makedirs(path)
# 	for obj in img_files:
# 		print(obj)
# 		get_json(obj, path)
	




# https://lasair-ztf.lsst.ac.uk/api/objects/?objectIds=ZTF22aadghqe&token=f7b4b64c53168512a4bcba06827c6c0015e9c9f6&format=json

