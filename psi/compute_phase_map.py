
import numpy as np
from unwrap_2d import *
from select_largest_obj import *

def compute_phase_map(I_1, I_2, I_3, cntr_thr, emask = 1):
    frame = (I_1 + I_2 + I_3) / 3
    h, w = frame.shape
    
    ### Computing intensity modulation, and snr for mask generation
    I_mod = np.sqrt(3*(I_1 - I_3)**2 + (2*I_2-I_1-I_3)**2) / 3
    gamma = I_mod / (frame + 1)
    
    thr_mean = np.median(gamma[h//6:5*h//6, w//6:5*w//6])
    thr_std = gamma[h//6:5*h//6, w//6:5*w//6].std()
    thr = thr_mean - cntr_thr * thr_std
    imask = np.zeros((h,w))
    imask[gamma>= thr] = 1
    imask = select_largest_obj(imask * emask, lab_val=1, fill_holes=True,
                               smooth_boundary=True, kernel_size=5)
    mask_idxs = np.where(imask==1)
    ###------------------------
    
    ### Computing phase map
    phi = np.arctan2(np.sqrt(3) * (I_1 - I_3) , (2 * I_2 - I_1 - I_3))
    phi_unwrapped = unwrap_2d(phi, fig_flag = False)
    ###-----------------
    
    ### mask phase map
    mask = np.ones_like(phi, dtype=np.bool)
    mask[mask_idxs] = False
    phi_unwrapped = np.ma.array(phi_unwrapped, mask=mask)
    ###-------------
    return phi_unwrapped