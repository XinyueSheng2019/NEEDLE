
import pandas as pd
import numpy as np
import json
import os
import re

from logging import raiseExceptions
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

import tensorflow as tf
from tensorflow.keras import models
import astropy
from astropy.io import fits
from astropy import visualization
from astropy.nddata import Cutout2D
from astropy.wcs.wcs import WCS

from multiprocessing import Process, Manager, Value
import h5py

### check bad images ###
bad_image_list = []

# Regular expressions for different types of images
OBJ_RE = re.compile('ZTF')
SCI_RE = re.compile('sci')
DIFF_RE = re.compile('diff')
REF_RE = re.compile('ref')

def check_shape(img):
    """
    Check if the image has the shape (60, 60) and does not contain only NaN values.
    """
    return img.shape == (60, 60) and not np.all(np.isnan(img))

def check_quality(model, img, threshold = 0.5):
    """
    Check if the image is good quality based on the model's prediction.
    """
    img = img.reshape(1,60,60,1)
    result = model.predict(img)[0][1]
    return True if result >= threshold else False

def cutout_img(data, header, ra, dec, size=60):
    """
    Cutout the image to the proposed size.
    """
    pixels = WCS(header=header).all_world2pix(ra, dec, 1)
    pixels = [int(x) for x in pixels]
    cutout = Cutout2D(data, position=pixels, size=size)
    return cutout.data

def get_shaped_image_simple(img, size=60, tolerance=2):
    """
    Roughly cut and complement the image to the desired size.
    """
    for i in range(2):
        if img.shape[i] > size:
            img = img[:size, :] if i == 0 else img[:, :size]

    for i in range(2):
        if size - img.shape[i] <= tolerance and size - img.shape[i] > 0:
            _, median, _ = astropy.stats.sigma_clipped_stats(img, mask=None, mask_value=0.0, sigma=3.0)
            while img.shape[i] < size:
                filler = np.repeat(median, size).reshape(1, size) if i == 0 else np.repeat(median, size).reshape(size, 1)
                img = np.append(img, filler, axis=i)
    return img


def get_shaped_image(filename, ra, dec):
    """
    Cutout image with 60x60 size and handle FITS file with different extensions.
    """
    try:
        with fits.open(filename, ignore_missing_end=True) as f:
            if filename.endswith('fz'):
                f.verify('fix')
                hdr = f[1].header
                data = f[1].data
            else:
                hdr = f[0].header
                data = f[0].data

            if hdr['NAXIS1'] >= 60 and hdr['NAXIS2'] >= 60:
                data = cutout_img(data, hdr, ra, dec)
            return data
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
        return None

def save_to_h5py(dataset, metaset, labels, idx_set, filepath):
    """
    Save the dataset, metaset, labels, and index set to an HDF5 file.
    """
    print('Image shape: ', dataset.shape)
    print('Meta shape: ', metaset.shape)
    print('Label shape: ', labels.shape)

    try:
        with h5py.File(filepath, "w") as f:
            f.create_dataset("imageset", dataset.shape, dtype='f', data=dataset)
            f.create_dataset("label", labels.shape, dtype='i', data=labels)
            f.create_dataset("metaset", metaset.shape, dtype='f', data=metaset)
            f.create_dataset("idx_set", idx_set.shape, dtype='f', data=idx_set)
    except Exception as e:
        print(f"Error saving to HDF5 file {filepath}: {e}")


def add_obj_meta(obj, obj_path, filefracday, add_host = False, recent_values = False, add_ext = False, band = 'r'):
    """
    Add object metadata from a CSV file for a specific object and filefracday.

    Parameters:
    - obj: str, object identifier
    - obj_path: str, path to the object's metadata files
    - filefracday: str, identifier for the file fraction day
    - add_host: bool, whether to add host galaxy information
    - recent_values: bool, whether to include recent values in the metadata

    Returns:
    - list, metadata for the object or None if not found
    """
    if add_ext:
        meta = pd.read_csv(f"{obj_path}/{obj}/obj_meta4ML_ext.csv")
    else:
        meta = pd.read_csv(f"{obj_path}/{obj}/obj_meta4ML.csv")

    # print('test, ', meta)
    d_row = meta.loc[meta.filefracday == int(filefracday)].fillna(0)
 
    if d_row.empty:
        print(f"{obj} {filefracday} NOT FOUND.\n")
        return None
    if band == 'mix':
        new_row = [
            d_row['candi_mag'].values[0],
            d_row['disc_mag'].values[0],
            d_row['delta_mag_discovery'].values[0],
            d_row['delta_t_discovery_band'].values[0],
            d_row['delta_t_discovery'].values[0]
        ]
    else: # if r band, ignore the delta_t_discovery_band as it's not accurate. only calcuate it for the disc_ratio.
        new_row = [
            d_row['candi_mag'].values[0],
            d_row['disc_mag'].values[0],
            d_row['delta_mag_discovery'].values[0],
            d_row['delta_t_discovery'].values[0]
        ]

    if recent_values:
        new_row += [
            d_row['delta_mag_recent'].values[0],
            d_row['delta_t_recent'].values[0]
        ]

    ratio_recent, ratio_disc = get_ratio(
        d_row['delta_mag_recent'].values[0],
        d_row['delta_t_recent'].values[0],
        d_row['delta_mag_discovery'].values[0],
        d_row['delta_t_discovery_band'].values[0]
    )

    new_row += [ratio_recent, ratio_disc]

    if add_host:
        new_row.append(d_row['delta_host_mag'].values[0])

    return new_row

def get_ratio(delta_mag_recent, delta_t_recent, delta_mag_disc, delta_t_disc):
    """
    Calculate the ratios of delta magnitude to delta time for recent and discovery values.

    Parameters:
    - delta_mag_recent: float, recent delta magnitude
    - delta_t_recent: float, recent delta time
    - delta_mag_disc: float, discovery delta magnitude
    - delta_t_disc: float, discovery delta time

    Returns:
    - tuple of floats, (ratio_recent, ratio_disc)
    """
    if isinstance(delta_mag_disc, np.ndarray):
        return (
            np.divide(delta_mag_recent, delta_t_recent, out=np.zeros_like(delta_mag_recent), where=delta_t_recent != 0),
            np.divide(delta_mag_disc, delta_t_disc, out=np.zeros_like(delta_mag_disc), where=delta_t_disc != 0)
        )
    else:
        if delta_t_disc == 0.0 or delta_t_recent == 0.0:
            return 0.0, 0.0
        return delta_mag_recent / delta_t_recent, delta_mag_disc / delta_t_disc




def add_host_meta(obj, host_path, only_complete=True):
    """
    Add host galaxy metadata from a CSV file.

    Parameters:
    - obj: str, object identifier
    - host_path: str, path to the host metadata files
    - only_complete: bool, whether to include only complete metadata

    Returns:
    - list, host metadata for the object or None if not found
    """
    def add_mag(line, band):
        return line.get(f'{band}Ap') or line.get(f'{band}PSF') or None

    if os.path.exists(f"{host_path}/{obj}.csv"):
        meta = pd.read_csv(f"{host_path}/{obj}.csv")
        line = meta.iloc[0]
        h_row = [add_mag(line, band) for band in ['g', 'r', 'i', 'z', 'y', 'g-r_', 'r-i_']]
        return h_row if all(h_row) or not only_complete else None

    return None if only_complete else [None] * 7

def add_sherlock_info(mag_records_path, ztf_id, properties, only_complete=True):
    """
    Add Sherlock information from a JSON file.

    Parameters:
    - mag_records_path: str, path to the magnitude records files
    - ztf_id: str, ZTF object identifier
    - properties: list, properties to extract from the Sherlock information
    - only_complete: bool, whether to include only complete information

    Returns:
    - list, Sherlock metadata for the object or None if not found
    """
    try:
        with open(f"{mag_records_path}/{ztf_id}.json") as f:
            obj_mags = json.load(f)

        sherlock_info = obj_mags.get('sherlock', {})
        sherlock_meta = [sherlock_info.get(p, 0 if not only_complete else None) for p in properties]

        return sherlock_meta if all(sherlock_meta) or not only_complete else None

    except (FileNotFoundError, KeyError):
        return None

def zscale(img, log_img=False):
    """
    Apply Z-scale normalization to an image.

    Parameters:
    - img: ndarray, input image
    - log_img: bool, whether to apply logarithmic scaling

    Returns:
    - ndarray, normalized image
    """
    vmin = visualization.ZScaleInterval().get_limits(img)[0]
    _, median, _ = astropy.stats.sigma_clipped_stats(img, mask=None, sigma=3.0, cenfunc='median')
    img = np.nan_to_num(img, nan=median)
    return np.log(img) if log_img else img

def image_normal(img):
    """
    Normalize the image data to the range [0, 1].

    Parameters:
    - img: ndarray, input image

    Returns:
    - ndarray, normalized image
    """
    return (img - np.min(img)) / (np.max(img) - np.min(img))

def img_reshape(img):
    """
    Reshape the image to add a single channel.

    Parameters:
    - img: ndarray, input image

    Returns:
    - ndarray, reshaped image
    """
    return img.reshape(img.shape[0], img.shape[1], 1)



def get_obs_image(obj_path, filefracday, no_diff, band, BClassifier):
    """
    Retrieve and process images from a specific observation.

    Parameters:
    - obj_path: str, path to the object's data
    - filefracday: str, identifier for the file fraction day
    - no_diff: bool, whether to include difference images
    - band: str, filter band
    - BClassifier: model, quality classifier

    Returns:
    - ndarray, combined data array or None
    """
    sci_re = re.compile('sci')
    diff_re = re.compile('diff')
    ref_re = re.compile('ref')

    def check_refs(ref_imgs):
        '''
        if one obj has multiple ref_imgs, and top one is bad, the other is good, we could still use it.
        '''
        final_img = ref_imgs[0]
        if len(ref_imgs) > 1:
            for fi in ref_imgs:
                data = fits.getdata(f"{obj_path}/{band}/{fi}")
                if not np.isnan(np.ravel(data)).all():
                    final_img = fi
                    break
        return final_img
          

    obs_listdir = os.listdir(f"{obj_path}/{band}/{filefracday}")
    sci_img = list(filter(sci_re.match, obs_listdir))[0]
    diff_img = list(filter(diff_re.match, obs_listdir))[0]
    band_path = f"{obj_path}/{band}"
    band_listdir = os.listdir(band_path)
    ref_imgs = list(filter(ref_re.match, band_listdir))
    ref_img = check_refs(ref_imgs)

    sci_data = get_shaped_image_simple(fits.getdata(f"{obj_path}/{band}/{filefracday}/{sci_img}"))
    ref_data = get_shaped_image_simple(fits.getdata(f"{obj_path}/{band}/{ref_img}"))
    if no_diff:
        if check_shape(sci_data) and check_shape(ref_data):
            sci_data = img_reshape(image_normal(zscale(sci_data)))
            ref_data = img_reshape(image_normal(zscale(ref_data)))
            if check_quality(BClassifier, sci_data) and check_quality(BClassifier, ref_data):
                return np.concatenate((sci_data, ref_data), axis=-1)
    else:
        diff_data = get_shaped_image_simple(fits.getdata(f"{obj_path}/{band}/{filefracday}/{diff_img}"))
        if check_shape(sci_data) and check_shape(diff_data) and check_shape(ref_data):
            sci_data = img_reshape(image_normal(zscale(sci_data)))
            ref_data = img_reshape(image_normal(zscale(ref_data)))
            diff_data = img_reshape(image_normal(zscale(diff_data)))
            if check_quality(BClassifier, sci_data) and check_quality(BClassifier, ref_data) and check_quality(BClassifier, diff_data):
                comb_data = np.concatenate((sci_data, ref_data, diff_data), axis=-1)
                return comb_data
    return None


def get_single_transient_peak(ztf_id, image_path, host_path, band='r', no_diff=True, BClassifier=None):
    """
    Retrieve the peak observation data for a single transient.

    Parameters:
    - ztf_id: str, ZTF object identifier
    - image_path: str, path to the image data
    - host_path: str, path to the host data
    - band: str, filter band ('r' or 'g')
    - no_diff: bool, whether to include difference images
    - BClassifier: model, quality classifier

    Returns:
    - tuple: combined image data and metadata arrays
    """
    if band not in ['r', 'g']:
        raise ValueError('Invalid band! Choose "r" or "g".')

    band_num = '2' if band == 'r' else '1'

    obj_path = os.path.join(image_path, ztf_id)

    with open(os.path.join(obj_path, 'mag_with_img.json'), 'r') as j_file:
        mag_wg = json.load(j_file)

    with open(os.path.join(obj_path, 'image_meta.json'), 'r') as mj_file:
        meta = json.load(mj_file)

    candids = mag_wg["candidates_with_image"]['f' + band_num]

    if not candids or meta['f' + band_num]["obj_with_no_ref"]:
        return None

    mags = np.array([[m['magpsf'], m["filefracday"]] for m in candids])
    idx = np.argmin(mags[:, 0])
    filefracday = mags[idx][1]

    if filefracday in meta['f' + band_num]["obs_with_no_diff"]:
        return None

    comb_data = get_obs_image(obj_path, filefracday, no_diff, band_num, BClassifier)

    if comb_data is None:
        return None

    obj_meta = add_obj_meta(ztf_id, image_path, filefracday, recent_values=False)
    host_meta = add_host_meta(ztf_id, host_path)
    sherlock_meta = add_sherlock_info(image_path, ztf_id, properties=['separationArcsec'])

    if obj_meta is None or host_meta is None or sherlock_meta is None:
        return None

    meta_data = obj_meta + host_meta + sherlock_meta

    return np.array(comb_data), np.array(meta_data)

def multi_worker_task(obj, f, image_path, host_path, mag_path, BClassifier, label_dict, mp_meta_set, mp_image_set, mp_label_set, mp_hash_table, mp_idx_set, mp_idx):
    """
    Process images and metadata for an object in a multiprocessing environment.

    Parameters:
    - obj: str, object identifier
    - f: str, filter band identifier
    - image_path: str, path to the image data
    - host_path: str, path to the host data
    - mag_path: str, path to the magnitude data
    - BClassifier: model, quality classifier
    - label_dict: dict, label dictionary
    - mp_meta_set: list, multiprocessing shared list for metadata
    - mp_image_set: list, multiprocessing shared list for images
    - mp_label_set: list, multiprocessing shared list for labels
    - mp_hash_table: dict, multiprocessing shared dictionary for hash table
    - mp_idx_set: list, multiprocessing shared list for indices
    - mp_idx: multiprocessing.Value, shared index value

    Returns:
    - None
    """
    obj_path = os.path.join(image_path, obj)

    with open(os.path.join(obj_path, 'mag_with_img.json'), 'r') as j_file:
        mag_wg = json.load(j_file)

    with open(os.path.join(obj_path, 'image_meta.json'), 'r') as mj_file:
        meta = json.load(mj_file)

    candids = mag_wg["candidates_with_image"]['f' + f]

    if not candids or meta['f' + f]["obj_with_no_ref"]:
        return

    for ffd in (m["filefracday"] for m in candids if m["filefracday"] not in meta['f' + f]["obs_with_no_diff"]):
        comb_data = get_obs_image(obj_path, ffd, True, f, BClassifier)
        if comb_data is None:
            continue

        obj_meta = add_obj_meta(obj, image_path, ffd)
        host_meta = add_host_meta(obj, host_path, True)
        sher_meta = add_sherlock_info(mag_path, obj, ['separationArcsec'], True)

        if obj_meta is not None and host_meta is not None and sher_meta is not None:
            meta_data = obj_meta + host_meta + sher_meta
            mp_meta_set.append(meta_data)
            mp_image_set.append(comb_data)
            mp_label_set.append(label_dict[meta['label']])
            mp_hash_table[mp_idx.value] = {'ztf_id': obj, 'ffd': ffd, 'type': meta['label'], 'label': label_dict[meta['label']]}
            mp_idx_set.append(mp_idx.value)
            mp_idx.value += 1

def serial_worker_task(obj, f, image_path, host_path, mag_path, BClassifier, label_dict):
    """
    Process images and metadata for an object in a serial manner.

    Parameters:
    - obj: str, object identifier
    - f: str, filter band identifier
    - image_path: str, path to the image data
    - host_path: str, path to the host data
    - mag_path: str, path to the magnitude data
    - BClassifier: model, quality classifier
    - label_dict: dict, label dictionary

    Returns:
    - tuple: lists of images, metadata, labels, and hash table
    """
    obj_path = os.path.join(image_path, obj)

    with open(os.path.join(obj_path, 'mag_with_img.json'), 'r') as j_file:
        mag_wg = json.load(j_file)

    with open(os.path.join(obj_path, 'image_meta.json'), 'r') as mj_file:
        meta = json.load(mj_file)

    candids = mag_wg["candidates_with_image"]['f' + f]

    obj_images = []
    obj_metas = []
    obj_labels = []
    obj_hashs = []

    if not candids or meta['f' + f]["obj_with_no_ref"]:
        return obj_images, obj_metas, obj_labels, obj_hashs

    for ffd in (m["filefracday"] for m in candids if m["filefracday"] not in meta['f' + f]["obs_with_no_diff"]):
        try:
            comb_data = get_obs_image(obj_path, ffd, True, f, BClassifier)
            if comb_data is None:
                continue

            obj_meta = add_obj_meta(obj, image_path, ffd, add_host=True)
            host_meta = add_host_meta(obj, host_path, True)
            sher_meta = add_sherlock_info(mag_path, obj, ['separationArcsec'], True)

            if obj_meta is not None and host_meta is not None and sher_meta is not None:
                meta_data = obj_meta + host_meta + sher_meta
                obj_images.append(comb_data)
                obj_metas.append(meta_data)
                obj_labels.append(label_dict[meta['label']])
                obj_hashs.append({'ztf_id': obj, 'ffd': ffd, 'type': meta['label'], 'label': label_dict[meta['label']]})
        except Exception as e:
            print(f"Error processing {obj} on {ffd}: {e}")
            continue

    return obj_images, obj_metas, obj_labels, obj_hashs




def single_band_all_db(image_path, host_path, mag_path, output_path, label_dict, band = 'r', no_diff = True, only_complete = True,  BClassifier = None, parallel = False):
    '''
    get all observations for each transient, and treat each of them as a sample.
    '''

    file_names = os.listdir(image_path) 
    file_names = list(filter(OBJ_RE.match, file_names))

    # sherlock_table = pd.read_csv(mag_path)
    
    image_set = []
    meta_set = []
    label_set = []
    idx_set = []
    hash_table = {}

    if band == 'r':
        f = '2'
    elif band == 'g':
        f = '1'
    else:
        raise Exception('WARNING: unvalid band!')

    if parallel is True:
        manager = Manager()
        mp_image_set = manager.list()
        mp_meta_set = manager.list()
        mp_label_set = manager.list()
        mp_idx_set = manager.list()
        mp_hash_table = manager.dict()
        mp_idx = Value('i', 0)

        n = 0
        p = []
        for obj in file_names:
            p.append(Process(target=multi_worker_task, args = (obj, f, image_path, host_path, mag_path, BClassifier, label_dict, mp_meta_set, mp_image_set, mp_label_set, mp_hash_table, mp_idx_set, mp_idx)))
            p[n].start()
            p[n].join()
            n += 1
        
        image_set += mp_image_set
        meta_set += mp_meta_set
        label_set += mp_label_set
        idx_set += mp_idx_set
        hash_table = {**hash_table, **mp_hash_table}
    else:
        image_set = []
        meta_set = []
        label_set = []
        idx_set = []
        hash_table = {}

        n = 0

        for obj in file_names:
            obj_images, obj_metas, obj_labels, obj_hashs = serial_worker_task(obj, f, image_path, host_path, mag_path, BClassifier, label_dict)
            if len(obj_images)>=1:
                image_set += obj_images
                meta_set += obj_metas
                label_set += obj_labels
                for x, y in zip(np.arange(n, n + len(obj_images), dtype=int), obj_hashs):
                    hash_table[str(x)] = y
                    idx_set.append(x)
                n += len(obj_images)


    
    # dataset, metaset, labels, idx_set, filepath
    save_to_h5py(np.array(image_set), np.array(meta_set), np.array(label_set), np.array(idx_set), output_path + 'data.hdf5')
    with open(output_path + "hash_table.json", "w") as outfile:
        json.dump(hash_table, outfile, indent = 4)


def remove_non_detection(mag_with_img = dict):
    for i in ['f1', 'f2', 'f3']:
        clean_candid = []
        for c in mag_with_img['candidates_with_image'][i]:
            if 'candid' in c.keys():
                clean_candid.append(c)
        mag_with_img['candidates_with_image'][i] = clean_candid

    return mag_with_img

def single_band_peak_db(image_path, host_path, mag_path, output_path, label_dict, band='r', no_diff=True, only_complete=True, add_host=False, BClassifier=None):
    '''
    This function processes the observations at the peak day for science, reference, and optionally difference images.
    It collects metadata and labels for each object, and stores the results in an HDF5 file with the keywords:
    image_set, meta_set, labels.

    Parameters:
    - image_path: str, path to the image data
    - host_path: str, path to the host data
    - mag_path: str, path to the magnitude data
    - output_path: str, path to save the output data
    - label_dict: dict, dictionary mapping labels
    - band: str, filter band ('r' or 'g')
    - no_diff: bool, whether to include difference images
    - only_complete: bool, whether to include only complete data
    - add_host: bool, whether to add host metadata
    - BClassifier: model, quality classifier

    Returns:
    - None
    '''
  
    file_names = [f for f in os.listdir(image_path) if OBJ_RE.match(f)]
    
    image_set = []
    meta_set = []
    label_set = []
    idx_set = []
    hash_table = {}

    mp_idx = 0
    
    band_dict = {'r': '2', 'g': '1'}
    if band not in band_dict:
        raise ValueError('Invalid band! Choose "r" or "g".')
    f = band_dict[band]

    for obj in file_names:
        obj_path = os.path.join(image_path, obj)
        
        with open(os.path.join(obj_path, 'mag_with_img.json'), 'r') as j:
            mag_wg = json.load(j)
        with open(os.path.join(obj_path, 'image_meta.json'), 'r') as mj:
            meta = json.load(mj)
        
        mag_wg = remove_non_detection(mag_with_img = mag_wg)
        candids = mag_wg["candidates_with_image"][f'f{f}']

        if not candids or meta[f'f{f}']["obj_with_no_ref"]:
            # Skip if there are no candidates or the object doesn't have a reference image
            continue

        mags = np.array([[m['magpsf'], m["filefracday"]] for m in candids])
        idx = np.argmin(mags[:, 0])
        filefracday = mags[idx][1]
        
        if filefracday not in meta[f'f{f}']["obs_with_no_diff"]:
            comb_data = get_obs_image(obj_path, filefracday, no_diff, f, BClassifier)
            if comb_data is not None:
                obj_meta = add_obj_meta(obj, image_path, filefracday, add_host, recent_values=False, add_ext=True, band=band)
                if not add_host:
                    meta_set.append(obj_meta)
                    image_set.append(comb_data)
                    label_set.append(label_dict[meta['label']])
                    hash_table[mp_idx] = {'ztf_id': obj, 'type': meta['label'], 'label': label_dict[meta['label']]}
                    idx_set.append(mp_idx)
                    mp_idx += 1 
                else:
                    host_meta = add_host_meta(obj, host_path, only_complete)
                    sher_meta = add_sherlock_info(mag_path, obj, ['separationArcsec'], only_complete)
                    if host_meta and sher_meta:   
                        meta_data = obj_meta + host_meta + sher_meta
                        meta_set.append(meta_data)
                        image_set.append(comb_data)
                        label_set.append(label_dict[meta['label']])
                        hash_table[mp_idx] = {'ztf_id': obj, 'type': meta['label'], 'label': label_dict[meta['label']]}
                        idx_set.append(mp_idx)
                        mp_idx += 1 

    save_to_h5py(np.array(image_set), np.array(meta_set), np.array(label_set), np.array(idx_set), os.path.join(output_path, 'data.hdf5'))
    with open(os.path.join(output_path, "hash_table.json"), "w") as outfile:
        json.dump(hash_table, outfile, indent=4)




def both_band_peak_db(image_path, host_path, mag_path, output_path, label_dict, no_diff=True, add_host=False, only_complete=True, BClassifier=None):
    '''
    This function processes observations at the peak day for both the 'r' and 'g' bands, collects metadata, and stores the results in an HDF5 file.
    Parameters:
    - image_path: str, path to the image data
    - host_path: str, path to the host data
    - mag_path: str, path to the magnitude data
    - output_path: str, path to save the output data
    - label_dict: dict, dictionary mapping labels
    - no_diff: bool, whether to include difference images
    - add_host: bool, whether to add host metadata
    - only_complete: bool, whether to include only complete data
    - BClassifier: model, quality classifier
    Returns:
    - None
    '''
    file_names = [f for f in os.listdir(image_path) if OBJ_RE.match(f)]
    
    image_set = []
    meta_set = []
    label_set = []
    idx_set = []
    hash_table = {}
    mp_idx = 0
    
    def stack_image(stack_list):
        return np.concatenate(stack_list, axis=-1)

    for obj in file_names:
        obj_path = os.path.join(image_path, obj)
        
        with open(os.path.join(obj_path, 'mag_with_img.json'), 'r') as j:
            mag_wg = json.load(j)
        with open(os.path.join(obj_path, 'image_meta.json'), 'r') as mj:
            meta = json.load(mj)

        print(obj)

        stack_list = []
        stack_meta = []
        flag = True  # only store data when both bands are available

        for f in ['1', '2']:
            candids = mag_wg["candidates_with_image"][f'f{f}']

            if len(candids) >= 1 and not meta[f'f{f}']["obj_with_no_ref"]:
                mags = np.array([[m['magpsf'], m["filefracday"]] for m in candids])
                idx = np.argmin(mags[:, 0])
                filefracday = mags[idx][1]

                if filefracday not in meta[f'f{f}']["obs_with_no_diff"]:
                    comb_data = get_obs_image(obj_path, filefracday, no_diff, f, BClassifier)
                    if comb_data is not None:
                        obj_meta = add_obj_meta(obj, image_path, filefracday, add_host, recent_values=False)
                        if obj_meta is not None:
                            stack_list.append(comb_data)
                            stack_meta += obj_meta
                        else:
                            flag = False
                    else:
                        flag = False      
                else:
                    flag = False
            else:
                flag = False

        if add_host:
            host_meta = add_host_meta(obj, host_path, only_complete)
            sher_meta = add_sherlock_info(mag_path, obj, ['separationArcsec'], only_complete)
            if host_meta and sher_meta:
                stack_meta += host_meta
                stack_meta += sher_meta
            else:
                flag = False

        if flag:
            comb_data = stack_image(stack_list)
            image_set.append(comb_data)
            meta_set.append(stack_meta)
            label_set.append(label_dict[meta['label']])
            hash_table[mp_idx] = {'ztf_id': obj, 'type': meta['label'], 'label': label_dict[meta['label']]}
            idx_set.append(mp_idx)
            mp_idx += 1
    
    save_to_h5py(np.array(image_set), np.array(meta_set), np.array(label_set), np.array(idx_set), os.path.join(output_path, 'data.hdf5'))
    with open(os.path.join(output_path, "hash_table.json"), "w") as outfile:
        json.dump(hash_table, outfile, indent=4)




def mixed_band_peak_db(image_path, host_path, mag_path, output_path, label_dict, no_diff = True, add_host = False, only_complete = True, BClassifier = None):
    '''
    This function choose the latest peak detection in g or r band, and take the science and reference images as image inputs.
    The metadata includes two parts:
    1. peak band magnitude and others before.
    2. peak another band magnitude (jd must earlier than the above one) and others
    3. colour difference: g-r, and Tg - Tr, (if g and r detection dates are too separated, dt = Tg - Tr is telling the DNN that the delta t is also big.)
    4. host information: r_host - r_peak, g_host - g_peak, others the same 
    5. sherlock arc-separation
    metadata preprocessing:
    two methods together.

    '''

    def find_peak_mag(candids):
        if len(candids) >= 1:
            mags = np.array([[m["magpsf"], m["filefracday"], m["mjd"]] for m in candids])
            idx = np.argmin(mags[:,0])
            peak_mjd = mags[idx][2]
            peak_mag = mags[idx][0]
            filefracday = mags[idx][1]
            return float(peak_mjd), float(peak_mag), filefracday
        else:
            return None, None, None

    obj_re = re.compile('ZTF')
  
    file_names = os.listdir(image_path) 
    file_names = list(filter(obj_re.match, file_names))
    
    image_set = []
    meta_set = []
    label_set = []
    idx_set = []
    hash_table = {}

    mp_idx = 0
    

    for obj in file_names:
        obj_path = os.path.join(image_path, obj)
        j = obj_path + '/mag_with_img.json'
        j = open(j, 'r')
        mag_wg = json.loads(j.read())

        mj = obj_path + '/image_meta.json'
        mj = open(mj, 'r')
        meta = json.loads(mj.read())
        
        candids_g = mag_wg["candidates_with_image"]['f1']
        candids_g = [x for x in candids_g if 'candid' in x.keys()]
        candids_r = mag_wg["candidates_with_image"]['f2']
        candids_r = [x for x in candids_r if 'candid' in x.keys()]

        print(obj)
        flag = False
        flag_g = False
        flag_r = False
        if len(candids_r) >= 1 and len(candids_g) >= 1:
            # if there are no detections in both band, skip
            if meta['f1']["obj_with_no_ref"] is False or meta['f2']["obj_with_no_ref"] is False:
                # if both bands have candids, but don't have reference images, skip. However, if g band has ref, but r band doesn't have ref but have candidates, r band detection can still be useful.
                
                peak_mjd_r, peak_mag_r, filefracday_r = find_peak_mag(candids_r)
                peak_mjd_g, peak_mag_g, filefracday_g = find_peak_mag(candids_g)

                if peak_mag_g is not None and peak_mag_r is not None:
                    peak_mag_g_minus_r = peak_mag_g - peak_mag_r
                    peak_t_g_minus_r = peak_mjd_g - peak_mjd_r
                    # find the images from g or r band, depends on which is available
                    try:
                        comb_data = get_obs_image(obj_path, filefracday_r, no_diff, '2', BClassifier)
                    except:
                        comb_data = None

                    if comb_data is None:
                        try: 
                            comb_data = get_obs_image(obj_path, filefracday_g, no_diff, '1', BClassifier)
                        except:
                            comb_data = None
                    if comb_data is not None:
                        obj_meta_r = add_obj_meta(obj, image_path, filefracday_r, add_host, recent_values=False, add_ext = True)
                        obj_meta_g = add_obj_meta(obj, image_path, filefracday_g, add_host, recent_values=False, add_ext = True)
                        flag = True

                elif peak_mag_g is not None and peak_mag_r is None:
                    flag_g = True
  
                elif peak_mag_r is not None and peak_mag_g is None:
                    flag_r = True
                    
        elif len(candids_r) >= 1 and len(candids_g) == 0:
            flag_r = True
        elif len(candids_g) >= 1 and len(candids_r) == 0:
            flag_g = True

        if flag_g:
            peak_mag_g_minus_r, peak_t_g_minus_r = 0., 0.
            try:
                comb_data = get_obs_image(obj_path, filefracday_g, no_diff, '1', BClassifier)
            except:
                comb_data = None
            if comb_data is not None:
                obj_meta_g = add_obj_meta(obj, image_path, filefracday_g, add_host, recent_values=False, add_ext = True)
                obj_meta_r = [0.]*len(obj_meta_g)
                flag = True
        if flag_r:
            peak_mag_g_minus_r, peak_t_g_minus_r = 0., 0.
            try:
                comb_data = get_obs_image(obj_path, filefracday_r, no_diff, '2', BClassifier)
            except:
                comb_data = None
            if comb_data is not None:
                    obj_meta_r = add_obj_meta(obj, image_path, filefracday_r, add_host, recent_values=False, add_ext = True)
                    obj_meta_g = [0.]*len(obj_meta_r)
                    flag = True
        if flag:
            if add_host is False:
                meta_data = obj_meta_r + obj_meta_g + [peak_mag_g_minus_r, peak_t_g_minus_r]
                meta_set.append(meta_data)
                image_set.append(comb_data)
                label_set.append(label_dict[meta['label']])
                hash_table[mp_idx] = {'ztf_id': obj, 'type': meta['label'], 'label':label_dict[meta['label']]}
                idx_set.append(mp_idx)
                mp_idx += 1 
            else:
                host_meta = add_host_meta(obj, host_path, only_complete)
                sher_meta = add_sherlock_info(mag_path, obj, ['separationArcsec'], only_complete)
                if host_meta is not None and sher_meta is not None:   
                    meta_data = obj_meta_r + obj_meta_g + [peak_mag_g_minus_r, peak_t_g_minus_r] + host_meta + sher_meta
                    meta_set.append(meta_data)
                    image_set.append(comb_data)
                    label_set.append(label_dict[meta['label']])
                    hash_table[mp_idx] = {'ztf_id': obj, 'type': meta['label'], 'label':label_dict[meta['label']]}
                    idx_set.append(mp_idx)
                    mp_idx += 1 
                else:
                    continue
    
    data_file = os.path.join(output_path, 'data.hdf5')
    save_to_h5py(np.array(image_set), np.array(meta_set), np.array(label_set), np.array(idx_set), data_file)
    hash_file = os.path.join(output_path, "hash_table.json")
    with open(hash_file, "w") as outfile:
        json.dump(hash_table, outfile, indent = 4)




