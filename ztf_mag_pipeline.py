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

