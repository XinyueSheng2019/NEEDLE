'''
This script is for updating the training and test sets with new ZTF objects.
Download their images corresponding to their magnitude information on Lasair/ZTF.
Build a .JSON meta file for each object.
'''

import os
import sys
import re
import subprocess
import errno
import csv
import json
from datetime import datetime as dt
from multiprocessing import Pool,cpu_count
from itertools import repeat
import pandas as pd 
import numpy as np 
import lasair


from astropy.time import Time
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs.wcs import WCS
from astropy.utils.data import get_pkg_data_filename
from astropy.modeling.rotations import Rotation2D
from ztf_mag_pipeline import get_json




def convert2jd(obs_date):
    if type(obs_date) == str:
        t = Time(obs_date)
        t.format = 'jd'
        return t.value
    elif type(obs_date) is float:
        if obs_date <= 2400000.5:
            return obs_date + 2400000.5
        else:
            return obs_date
    else:
        jds = []
        for d in obs_date.tolist():
            t = Time(d)
            t.format = 'jd'
            jds.append(t.value)
        return jds
  


def convert_ztf(ztf_name, obj_file):
    return obj_file[(obj_file.ZTFID == ztf_name)]['n_RA'], obj_file[(obj_file.ZTFID == ztf_name)]['n_Dec']


# def read_multi_objs(filepath, outdir, size = 1, duration = 100):  # multithreading available
#     # return ztf_name, pos, discover date.
#     obj_table = pd.read_csv(filepath)

#     outdir = os.getcwd() + '/' + outdir
#     path_safe(outdir)

#     print(f'starting computations on {cpu_count()} cores')

#     obj_table_list = [obj_table.iloc[[i]] for i in np.arange(len(obj_table))]

#     with Pool() as pool:
#         pool.starmap(collect_image, zip(repeat(size), repeat(duration), obj_table_list, repeat(outdir)))
    

    # single core: 

    # for i in np.arange(len(obj_table)):
    #     collect_image(size, duration, obj_table.iloc[[i]], outdir)


def test_valid(path):
    if os.path.exists(path):
        if os.path.getsize(path) < 800:
            os.remove(path)
            return 0
        else:
            return 1
    else:
        return 0
   
    


def test_valid_and_flip(path, rotation = False, flip = False):
    if os.path.getsize(path) < 800:
        os.remove(path)
        return 0
    else:
        flag = 1
        if rotation is True:
            rotate_image(path)
        if flip is True:
            flag = flip_image(path)
        if flag == 1:
            return 1
        else:
            return 0
        



def path_safe(path):
    original_umask = os.umask(0)
    try:
        os.makedirs(path, mode=0o775)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    os.umask(original_umask)


def create_path(path, new_folder):
    new_path = os.path.join(path, str(new_folder))
    if os.path.isdir(new_path) == False:
        os.mkdir(new_path)
    return new_path


def cutout_ref(filename, ra, dec, size = 60):
    # cutout the reference image to proposed size
    fn = get_pkg_data_filename(filename)
    f = fits.open(fn, ignore_missing_end=True)
    header = fits.getheader(filename)
    pixels = WCS(header=header).all_world2pix(ra,dec,1)
    cutout = Cutout2D(f[0].data, position=pixels, size=size)

    # Put the cutout image in the FITS HDU
    f[0].data = cutout.data

    # Write the cutout to a new FITS file
    cutout_filename = filename
    f[0].writeto(cutout_filename, overwrite=True)


def cutout_img(data, header, ra, dec, size = 60):
    # cutout the image to proposed size
    pixels = WCS(header=header).all_world2pix(ra,dec,1)
    pixels = [int(x) for x in pixels]
    cutout = Cutout2D(data, position=pixels, size=size)
    return cutout.data


def flip_image(filename):
    # Used for the sci and diff images in order to let North up and East left.
    if filename[-2:]!='fz':
        fn = get_pkg_data_filename(filename)
        f = fits.open(fn, ignore_missing_end=True)[0]
        f.data = np.flip(f.data, 0)
        f.writeto(filename, overwrite=True)
        return 1
    else:
        f = fits.open(filename,ignore_missing_end=True)
        f.verify('fix')
        if f[0].data is not None:
            f[0].data = np.flip(f[0].data, 0)
            f.writeto(filename, overwrite=True)
            return 1
        elif len(f) > 1 and f[1].data is not None:
            f[1].data = np.flip(f[1].data, 0)
            f.writeto(filename, overwrite=True)
            return 1
        else:
            os.remove(filename)
            return 0



def rotate_image(filename):
    # Used for ref image.
    # fn = get_pkg_data_filename(filename)
    f = fits.open(filename,ignore_missing_end=True)[0]
    f.data = np.rot90(f.data, 2)
    f.writeto(filename, overwrite=True)
  


def check_complete(path):
    # check if the downloaded obj images are completed. If not, return False.
    if os.path.exists(path+'/image_meta.json'):
        return 1
    else:
        return 0



def read_table(table, size, duration, outdir, magdir, parrallel = False):
    ''' read a list of ZTF object'''

    if parrallel:
        with Pool() as pool:
            pool.starmap(collect_image_from_irsa, zip(table['object_id'], table['disdate'], table['type'], repeat(size), repeat(duration), repeat(outdir), repeat(magdir)))
    else:
        for _, row in table.iterrows():
            collect_image_from_irsa(row['object_id'], row['disdate'], row['type'], size, duration, outdir, magdir)



def collect_image_from_irsa(ztf_id, disdate, type, size, duration, outdir, magdir):
    ''' download all existed images of a ZTF object, from irsa API.'''

    def add_flag(disdate, start, end):
        '''
        Add a flag to the meta:
            0: discover date within the start and end dates
            1: discover date before the start date
            2: discover date after the end date
            3: no valid dates
        '''
        if disdate >= start and disdate < end:
            return 0
        elif disdate < start:
            return 1
        elif disdate >= end:
            return 2
        else:
            return 3


    jfile = get_json(ztf_id, magdir)

    ra, dec = jfile["objectData"]["ramean"], jfile["objectData"]["decmean"]


    if disdate is None:
        disdate = jfile['candidates'][-1]['jd']
    else:
        disdate = convert2jd(disdate)
    
    if type is None:
        if 'TNS' in jfile.keys() and 'type' in jfile['TNS']:
            label = jfile['TNS']['type']
            print('TNS has its classification as ', jfile['TNS']['type'])
        else:
            label = None
    else:
        label = type
    
    obj_name = ztf_id

    mag_cand = jfile['candidates']
    jd_list = np.array([float(x['jd']) for x in mag_cand])
    fid_list = np.array([x['fid'] for x in mag_cand])
    obj_dir = outdir + '/' + obj_name 

    if os.path.exists(obj_dir) and check_complete(obj_dir):
        print('ALREADY EXIST AND COMPLETE: ', obj_name)
        return 1
    else:
        obj_dir = create_path(outdir, obj_name)
        # convert the input dates to MJD
        print('COLLECTING: ', obj_name)

        # generate a .JSON meta file
        meta_dict = {}
        meta_dict['id'] = obj_name
        meta_dict['label'] = label
        meta_dict['ra'], meta_dict['dec'] = ra, dec 
        meta_dict['size'] = 1
        meta_dict['disdate'] = disdate

        # generate a .JSON file for magnitudes with images provided
        mag_with_img_dict = {}
        mag_with_img_dict['id'] = obj_name
        mag_with_img_dict['label'] = label
        mag_with_img_dict['ra'] = ra
        mag_with_img_dict['dec'] = dec
        mag_with_img_dict['disdate'] = disdate
        mag_with_img_dict['candidates_with_image'] = {'f1':[], 'f2':[], 'f3':[]}
            

        f_subsets=[1,2,3]
        # f_subsets = {'g':1, 'r':2, 'i':3}
        for f in f_subsets: # multithreading available

            filter_ind = np.array([d for d, x in enumerate(fid_list) if x == f])
            filter_jd = np.array([jd_list[d] for d, x in enumerate(fid_list) if x == f])
    
            meta_dict['f'+str(f)] = {}
            path = create_path(obj_dir, f)
            irsa_call = '\"https://irsa.ipac.caltech.edu/ibe/search/ztf/products/sci?POS=%s,%s&ct=csv&where=fid=%s\"' % (ra, dec, f)

            print(irsa_call)

            filename = path + '/ztf_' + str(f) + '.csv'
            os.system("curl -o %s %s" % (filename, irsa_call))

            df = pd.read_csv(filename)

            if len(df) >= 1: # if the file has observations
                earliest = min(df['obsjd'])
                latest = max(df['obsjd'])

                start_jd = disdate
                end_jd = start_jd + duration

                if start_jd < earliest and end_jd >= earliest:
                    start_jd = earliest
                elif end_jd < earliest:
                    start_jd = earliest
                    end_jd = earliest + duration
                elif latest < start_jd:
                    start_jd = None
                    end_jd = None
                
                meta_dict['f'+str(f)]['start'] = start_jd
                meta_dict['f'+str(f)]['end'] = end_jd
                if start_jd is None:
                    continue
                
                meta_dict['f'+str(f)]['obsnum'] = 0
                meta_dict['f'+str(f)]['flag'] = add_flag(disdate, start_jd, end_jd)
                columns_name = df.columns
                del df

                meta_dict['f'+str(f)]['multiple_ref'] = False
                meta_dict['f'+str(f)]['obj_with_no_ref'] = False
                meta_dict['f'+str(f)]['obs_with_no_diff'] = []
                meta_dict['f'+str(f)]['obs_with_no_sci'] = []
                meta_dict['f'+str(f)]['bogus'] = [] # leave the space for further operations
                meta_dict['f'+str(f)]['withMag'] = []

                # create a .csv file to record required observation dates' info
                meta_csv = open(path+'/'+'obs_info.csv', 'w')
                writer_csv = csv.writer(meta_csv)
                writer_csv.writerow(columns_name)


                with open(filename) as csvfile:
                    reader = csv.DictReader(csvfile)
                    count = 0
                    for line in reader: # multithreading available

                        obsjd = float(line['obsjd'])
                        filefracday = line['filefracday']
                        yyyy = filefracday[0:4]
                        mmdd = filefracday[4:8]
                        fracday = filefracday[8:14]
                        
                        filtercode = line['filtercode']
                        # obsjd = line['obsjd']

                        ccdid = line['ccdid']
                        pad_num = 2 - len(str(ccdid)) # pad the ccdid to two digits
                        ccdid = '0'*pad_num + str(ccdid)

                        field = int(line['field'])
                        pad_num = 6 - len(str(field))
                        field = '0'*pad_num + str(field)

                        imgtypecode = line['imgtypecode']
                        qid = line['qid']


                        if obsjd >= start_jd and obsjd <= end_jd:  
                            count += 1

                            obsjd_path = create_path(path, filefracday)

                            # science image:
                            
                            irsa_url = "%4s/%4s/%6s/ztf_%14s_%s_%s_c%s_%s_q%s_sciimg.fits" % (yyyy, mmdd, fracday, filefracday, field, filtercode, ccdid, imgtypecode, qid)
                            irsa_url = '\"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/%s?center=%s,%s&size=%sarcmin&gzip=false\"' % (irsa_url, ra, dec, size)
                            sci_fname = "sci_ztf_%14s_%s_%s_c%s_%s_q%s_sciimg.fits.fz" % (filefracday, field, filtercode, ccdid, imgtypecode, qid)
                            sci_filename = obsjd_path + '/' + sci_fname
                            print('science image url: ', irsa_url)

                            if not os.path.exists(sci_filename):
                                os.system("curl -o %s %s" % (sci_filename, irsa_url))
                                test_re = test_valid_and_flip(sci_filename, flip = True)
                                if test_re == 0:
                                    meta_dict['f'+str(f)]['obs_with_no_sci'].append(filefracday) 
                                else:
                                    if len(filter_jd)>=1 and np.min(np.abs(filter_jd-obsjd)) <= 0.1:  
                                        find_idx = np.argmin(np.abs(filter_jd-obsjd))
                                        obs_mag_info = mag_cand[filter_ind[find_idx]]
                                        with open(obsjd_path + "/mag_info.json", "w") as outfile:
                                            json.dump(obs_mag_info, outfile, indent=4)
                                        meta_dict['f'+str(f)]['withMag'].append(filefracday)
                                        obs_mag_info['filefracday'] = filefracday
                                        mag_with_img_dict['candidates_with_image']['f'+str(f)].append(obs_mag_info)
                            else:
                                if os.path.exists(obsjd_path + "/mag_info.json"):
                                    meta_dict['f'+str(f)]['withMag'].append(filefracday)
                                    m = open(obsjd_path + "/mag_info.json",'r')
                                    mag_info = json.loads(m.read())
                                    mag_info['filefracday'] = filefracday
                                    mag_with_img_dict['candidates_with_image']['f'+str(f)].append(mag_info)

                            
                            # difference image:
                            diff_url = "%4s/%4s/%6s/ztf_%14s_%s_%s_c%s_%s_q%s_scimrefdiffimg.fits.fz" % (yyyy, mmdd, fracday, filefracday, field, filtercode, ccdid, imgtypecode, qid)
                            diff_url = '\"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/sci/%s?center=%s,%s&size=%sarcmin&gzip=true\"' % (diff_url, ra, dec, size)
                            diff_frame = "diff_ztf_%14s_%s_%s_c%s_%s_q%s_scimrefdiffimg.fits.fz" % (filefracday, field, filtercode, ccdid, imgtypecode, qid)
                            diff_filename = obsjd_path + '/' + diff_frame
                            print('diff image url: ', diff_url)
                            

                            if not os.path.exists(diff_filename):
                                os.system("curl -o %s %s" % (diff_filename, diff_url))

                                test_re = test_valid_and_flip(diff_filename, flip = True)
                                if test_re == 0:
                                    meta_dict['f'+str(f)]['obs_with_no_diff'].append(filefracday)
     
                            if not os.listdir(obsjd_path):
                                os.rmdir(obsjd_path)
   
                            writer_csv.writerow(line.values())

                        else:
                            continue
                    
                    # reference image: this is flipped by irsa database.
                    fieldprefix = field[:3]
                    ref_url = '\"https://irsa.ipac.caltech.edu/ibe/data/ztf/products/ref/%s/field%s/%s/ccd%s/q%s/ztf_%s_%s_c%s_q%s_refimg.fits?center=%s,%s&size=%sarcmin&gzip=false\"' % (fieldprefix, field, filtercode, ccdid, qid, field, filtercode, ccdid, qid, ra, dec, size)
                    ref_frame = 'ref_ztf_'+field+'_'+filtercode+'_c'+ccdid+'_q'+qid+'_refimg.fits'
                    ref_filename = path + '/' + ref_frame
                    print('ref image url: ', ref_url)
                    if not os.path.exists(ref_filename):
                        os.system("curl -o %s %s" % (ref_filename, ref_url))
                        print(ref_filename)
                        test_re = test_valid_and_flip(ref_filename, rotation = True)
                        if test_re == 0:
                            meta_dict['f'+str(f)]['obj_with_no_ref'] = True

                meta_dict['f'+str(f)]['obsnum'] = count
                meta_csv.close()

            else:
                meta_dict['f'+str(f)]['start'] = None
                meta_dict['f'+str(f)]['end'] = None
                meta_dict['f'+str(f)]['obsnum'] = 0
                continue

    # If any .gz file appears to be ASCII, just delete it

            filelist = os.listdir(path)
            for file in filelist:
                if (file.endswith('.gz')):
                    filefull = os.path.join(path, file)
                    if (re.search(r':.* text',subprocess.Popen(["file",filefull],stdout=subprocess.PIPE).stdout.read()) is not None):
                        print('No image found, deleting: ', filefull)
                        os.remove(filefull)

        with open(obj_dir + "/image_meta.json", "w") as outfile:
            json.dump(meta_dict, outfile, indent=4)
        
        with open(obj_dir+ '/mag_with_img.json','w') as outfile:
            json.dump(mag_with_img_dict, outfile, indent=4)
        print ('Done: ', obj_name)


        return 0
    

def collect_image_from_lasair(ztf_id, disdate, type, size, duration, outdir, magdir):
    ''' download all existed images of a ZTF object, from Lasair mag JSON file.'''

    def add_flag(disdate, start, end):
        '''
        Add a flag to the meta:
            0: discover date within the start and end dates
            1: discover date before the start date
            2: discover date after the end date
            3: no valid dates
        '''
        if disdate >= start and disdate < end:
            return 0
        elif disdate < start:
            return 1
        elif disdate >= end:
            return 2
        else:
            return 3

    print('Collecting ZTF object: ', ztf_id)
    jfile = get_json(ztf_id, magdir)

    # remove non-detection
    temp_list = []
    for cd in jfile['candidates']:
        if 'candid' in cd.keys():
            temp_list.append(cd)
    jfile['candidates'] = temp_list
 

    ra, dec = jfile["objectData"]["ramean"], jfile["objectData"]["decmean"]

    disdate = jfile['objectData']['discMjd']
    discFilter = jfile['objectData']['discFilter']

    
    if type is None:
        if 'TNS' in jfile.keys() and 'type' in jfile['TNS']:
            label = jfile['TNS']['type']
            print('TNS has its classification as ', jfile['TNS']['type'])
        else:
            label = None
    else:
        label = type

    if label is None:
        print('Classification is no found.\n')

    
    obj_name = ztf_id

    mag_cand = jfile['candidates']
    obj_dir = outdir + '/' + obj_name 

    if os.path.exists(obj_dir) and check_complete(obj_dir):
        print('ALREADY EXIST AND COMPLETE: ', obj_name)
        return 1
    else:
        obj_dir = create_path(outdir, obj_name)
        print('Collecting images: ', obj_name)

        # generate a .JSON meta file
        meta_dict = {}
        meta_dict['id'] = obj_name
        meta_dict['label'] = label
        meta_dict['ra'], meta_dict['dec'] = ra, dec 
        meta_dict['size'] = 1
        meta_dict['disdate'] = disdate

        # generate a .JSON file for magnitudes with images provided
        mag_with_img_dict = {}
        mag_with_img_dict['id'] = obj_name
        mag_with_img_dict['label'] = label
        mag_with_img_dict['ra'] = ra
        mag_with_img_dict['dec'] = dec
        mag_with_img_dict['disdate'] = disdate
        mag_with_img_dict['candidates_with_image'] = {'f1':[], 'f2':[], 'f3':[]}
            

        f_subsets=[1,2,3]
        # f_subsets = {'g':1, 'r':2, 'i':3}

        for f in f_subsets: # multithreading available
            meta_dict['f'+str(f)] = {}
            path = create_path(obj_dir, f)

            candidates = [x for x in mag_cand if 'image_urls' in x.keys() and x['fid'] == f]
            candidates_mjd = [x['mjd'] for x in candidates]

            if len(candidates) >= 1: # if the file has observations
                earliest = min(candidates_mjd)
                latest = max(candidates_mjd)

                start_mjd = disdate
                end_mjd = start_mjd + duration

                if start_mjd < earliest and end_mjd >= earliest:
                    start_mjd = earliest
                elif end_mjd < earliest:
                    start_mjd = earliest
                    end_mjd = earliest + duration
                elif latest < start_mjd:
                    start_mjd = None
                    end_mjd = None
                
                meta_dict['f'+str(f)]['start'] = start_mjd
                meta_dict['f'+str(f)]['end'] = end_mjd
                if start_mjd is None:
                    continue
                
                meta_dict['f'+str(f)]['obsnum'] = 0
                meta_dict['f'+str(f)]['flag'] = add_flag(disdate, start_mjd, end_mjd)
                # columns_name = df.columns

                meta_dict['f'+str(f)]['multiple_ref'] = False
                meta_dict['f'+str(f)]['obj_with_no_ref'] = False
                meta_dict['f'+str(f)]['obs_with_no_diff'] = []
                meta_dict['f'+str(f)]['obs_with_no_sci'] = []
                meta_dict['f'+str(f)]['bogus'] = [] # leave the space for further operations
                meta_dict['f'+str(f)]['withMag'] = []

                count = 0
                if len(candidates) >= 1:
                    for c in candidates:
                        if c['mjd'] >= start_mjd and c['mjd'] <= end_mjd: 
                            obsjd_path = create_path(path, c["candid"])
                            science_url  = c['image_urls']['Science']
                            template_url = c['image_urls']['Template']
                            difference_url = c['image_urls']['Difference']
                            sci_name = 'sci_' + str(c['candid']) + '.fits'
                            ref_name = 'ref_' + str(c['candid']) + '.fits'
                            diff_name = 'diff_'+ str(c['candid']) + '.fits'
                            sci_filename = obsjd_path + '/' + sci_name 
                            diff_filename = obsjd_path + '/' + diff_name
                            ref_filename = path + '/' + ref_name
                            if not os.path.exists(sci_filename):
                                    os.system("curl -o %s %s" % (sci_filename, science_url))
                                    os.system("curl -o %s %s" % (ref_filename, template_url))
                                    os.system("curl -o %s %s" % (diff_filename, difference_url))
                            # test if images have been downloaded correctly
                            if test_valid(sci_filename):
                                meta_dict['f'+str(f)]['withMag'].append(c['candid'])  
                                mag_with_img_dict['candidates_with_image']['f'+ str(f)].append(c)
                                count += 1 
                            else:
                                meta_dict['f'+str(f)]['obs_with_no_sci'].append(c['candid'])
                            if not test_valid(diff_filename):
                                meta_dict['f'+str(f)]['obs_with_no_diff'].append(c['candid'])
                               
                            test_valid(ref_filename)
                            if not os.path.exists(sci_filename) and not os.path.exists(diff_filename):
                                os.rmdir(obsjd_path)

                        else:
                            continue

                meta_dict['f'+str(f)]['obsnum'] = count

        # if lasair doesn't have its images, try isra API
        if len(meta_dict['f1']['withMag']) == 0 and len(meta_dict['f2']['withMag']) == 0:
            print('Lasair does not have %s\' images. Trying isra API. \n'% ztf_id)
            collect_image_from_irsa(ztf_id, disdate, type, size, duration, outdir, magdir)
        else:
            with open(obj_dir + "/image_meta.json", "w") as outfile:
                json.dump(meta_dict, outfile, indent=4)
            
            with open(obj_dir+ '/mag_with_img.json','w') as outfile:
                json.dump(mag_with_img_dict, outfile, indent=4)
            print ('Done: ', obj_name)


        return 0
    



def check_image_shape(filename, ra, dec):
    # add bogus label
    if filename[-2:]!='fz':
        f = fits.open(filename,ignore_missing_end=True)  # open a FITS file
        hdr = f[0].header 
        # if np.std(f[0].data) > 1000:
        #     return 0
        if hdr['NAXIS1'] == 61 or hdr['NAXIS2'] == 61:
            f[0].data = cutout_img(f[0].data, hdr, ra, dec)
            f[0].writeto(filename, overwrite=True)
        if  hdr['NAXIS1'] < 60 or hdr['NAXIS2'] < 60 :
            return 0
    else:
        f = fits.open(filename,ignore_missing_end=True)
        f.verify('fix')
        hdr = f[1].header
        # if np.std(f[1].data) > 1000:
        #     return 0
        if hdr['NAXIS1'] == 61 or hdr['NAXIS2'] == 61:
            f[1].data = cutout_img(f[1].data, hdr, ra, dec)
            f[1].writeto(filename, overwrite=True)
        if  hdr['NAXIS1'] <60 or hdr['NAXIS2'] <60 :
            return 0
    
    return 1






if __name__ == '__main__':
    '''
    User could call this script, or just modify the '__main__' function to download objects they want.
    They need to provide ZTF magnitude json file, or ZTF object table with coordinates.
    '''

    
    size = 1
    duration = 100


