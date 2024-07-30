
import extinction
from extinctions import reddening
import numpy as np 
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

def ext(ra,dec):
    red = (reddening.Reddening(ra, dec)).query_local_map(dustmap='sfd')*0.86
    AV = 3.1*float(str(red)[1:-1])
    
    wave = np.array([4829.50, 6463.75, 4900.12, 6241.27, 7563.76, 8690.10, 9644.63]) 

    AC_keys = ['ZTF_g', 'ZTF_r', 'PS_g', 'PS_r', 'PS_i', 'PS_z', 'PS_y']
    AC = extinction.fitzpatrick99(wave, AV, 3.1)
    
    ext_val = {}
    for A, B in zip(AC_keys, AC):
        ext_val[A] = B
    
    return ext_val

