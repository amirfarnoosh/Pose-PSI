
"""
This function is used to detect pattern ID in color-coded 
phase shift interferometry
"""

import numpy as np

def esync_func(frame_rgb, color_diff_thr, pixel_thr, intensity_thr):

    '''Detects pattern ID given an RGB color-coded input image
        (colo-coded synchronization).
    Args:
        frame_rgb (3D uint8 array): a numpy array of input RGB image
        color_diff_thr (uint8): a threshold for detecting colors in 
            color-coded patterns.
        pixel_thr (int): a threshold for number of pixels
            with a specific color in color-coded patterns.
        intensity_thr (uint8): intensity threshold for detecting patterns
            from no-pattern frames in color-coded patterns.
    Returns: (if "walk" is True)
        p_num (int): pattern ID
        mask (2D array): a binary mask for excluding color stripes
    '''
    #analyse frame
    h, w, _ = frame_rgb.shape
    frame_rgb = frame_rgb.astype('float')
    frame_rgb = frame_rgb * np.concatenate(
                                            (1.1 * np.ones((h,w,1)),
                                             np.ones((h,w,1)),
                                             np.ones((h,w,1))
                                             ),
                                            axis = 2)
    
    diff = frame_rgb - np.tile(frame_rgb.max(axis = 2).reshape(h,w,1),(1,1,3))
    idx_diff = np.where(np.absolute(diff)>color_diff_thr)
    idx = idx_diff[2]
    idx_hist = [len(np.where(idx==0)[0]),len(np.where(idx==1)[0]),len(np.where(idx==2)[0])]
    idxx = np.argmin(idx_hist)
    mm = np.sort(idx_hist)
    if mm[1]-mm[0]>pixel_thr and frame_rgb.mean()>intensity_thr:        
        p_num = idxx
        mask = np.zeros((h,w))
        mask[idx_diff[0],idx_diff[1]] = 1
        mask[idx_diff[0][idx == idxx],idx_diff[1][idx == idxx]] = 0
    elif frame_rgb.mean() <= 50:
        p_num = 3
        mask = []
    else:
        p_num = 4
        mask = []
    
    return p_num, mask
    
    