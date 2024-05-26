'''
sherlock/tde host: 0.0014
guess host: 0.0083
'''
import numpy as np
import requests
from astropy import coordinates as coords
from astropy import units as u
import pandas as pd
import os
import json

def PS1catalog_host(_id, _ra, _dec, radius=0.00139, save_path=None):
    # Check if coordinates are valid
    if _ra is None or _dec is None:
        return 0

    queryurl = f"https://catalogs.mast.stsci.edu/api/v0.1/panstarrs/dr2/stack.json?ra={_ra}&dec={_dec}&radius={radius}"
    columns = "[raStack,decStack,gPSFMag,gPSFMagErr,rPSFMag,rPSFMagErr,iPSFMag,iPSFMagErr,zPSFMag,zPSFMagErr,yPSFMag,yPSFMagErr," \
              "gApMag,gApMagErr,rApMag,rApMagErr,iApMag,iApMagErr,zApMag,zApMagErr,yApMag,yApMagErr,yKronMag]"
    queryurl += f"&columns={columns}"

    query = requests.get(queryurl)
    results = query.json()

    # Check if any results are returned
    if 'data' not in results or len(results['data']) < 1:
        print('Field not good results')
        return 1

    data = np.array(results['data'])

    # Remove invalid coordinates and duplicates
    data = data[(data[:, 0] > -999) & (data[:, 1] > -999)]
    if len(data) < 1:
        return 1

    data[data == -999] = np.nan

    # Find matching coordinates within a certain radius
    catalog = coords.SkyCoord(ra=data[:, 0] * u.degree, dec=data[:, 1] * u.degree)
    matches = []
    for i, source in enumerate(coords.SkyCoord(ra=data[:, 0] * u.degree, dec=data[:, 1] * u.degree)):
        d2d = source.separation(catalog)
        catalogmsk = d2d < 2.5 * u.arcsec
        indexmatch = np.where(catalogmsk)[0]
        if len(indexmatch) > 0:
            matches.extend(data[indexmatch])

    if len(matches) >= 1:
        # Add magnitude difference columns
        wdata = pd.DataFrame(matches, columns=['ra', 'dec', 'gPSF', 'gPSFerr', 'rPSF', 'rPSFerr', 'iPSF', 'iPSFerr',
                                               'zPSF', 'zPSFerr', 'yPSF', 'yPSFerr', 'gAp', 'gAperr', 'rAp',
                                               'rAperr', 'iAp', 'iAperr', 'zAp', 'zAperr', 'yAp', 'yAperr'])
        wdata['g-r_PSF'] = wdata['gPSF'] - wdata['rPSF']
        wdata['r-i_PSF'] = wdata['rPSF'] - wdata['iPSF']
        wdata['g-r_PSFerr'] = np.sqrt(wdata['gPSFerr'] ** 2 + wdata['rPSFerr'] ** 2)
        wdata['r-i_PSFerr'] = np.sqrt(wdata['rPSFerr'] ** 2 + wdata['iPSFerr'] ** 2)
        wdata['g-r_Ap'] = wdata['gAp'] - wdata['rAp']
        wdata['r-i_Ap'] = wdata['rAp'] - wdata['iAp']
        wdata['g-r_Aperr'] = np.sqrt(wdata['gAperr'] ** 2 + wdata['rAperr'] ** 2)
        wdata['r-i_Aperr'] = np.sqrt(wdata['rAperr'] ** 2 + wdata['iAperr'] ** 2)

        if save_path is not None:
            wdata.to_csv(os.path.join(save_path, f'{_id}.csv'))
    else:
        print('Field not good results')

    return 1

def get_host_from_magfile(ZTF_path, mag_path):
    obj_mag_path = os.path.join(mag_path, ZTF_path)
    with open(obj_mag_path) as f:
        obj_mag = json.load(f)
    if obj_mag:
        return obj_mag['sherlock']['raDeg'], obj_mag['sherlock']['decDeg']
    else:
        return None, None



