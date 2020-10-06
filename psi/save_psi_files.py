#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 02:24:09 2019

@author: amirreza
"""
import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage import exposure

def save_psi_files(dump_path,
                   cnt_save,
                   phi_unwrapped,
                   pts_cloud,
                   mask_idxs,
                   frame_rgb,
                   I_1,
                   I_2,
                   I_3,
                   corr_coef_save,
                   times_save
                   ):
    I_phi = np.uint8(exposure.rescale_intensity(phi_unwrapped, out_range=(0, 255)))
    I_phi = np.ma.array(I_phi, mask = phi_unwrapped.mask)
    I_phi_masked = I_phi * 1
    I_phi_masked[mask_idxs] = 0
    
    np.save(dump_path + 'pts_cloud_%d.npy' %cnt_save, pts_cloud)
    np.save(dump_path + 'mask_idxs_%d.npy' %cnt_save, mask_idxs)
    np.save(dump_path + 'times.npy', times_save)
    cv2.imwrite(dump_path + 'frame_rgb_%d.png' %cnt_save, frame_rgb)
    cv2.imwrite(dump_path + 'frame_0_%d.png' %cnt_save, np.uint8(I_1))
    cv2.imwrite(dump_path + 'frame_1_%d.png' %cnt_save, np.uint8(I_2))
    cv2.imwrite(dump_path + 'frame_2_%d.png' %cnt_save, np.uint8(I_3))

    plt.imsave(dump_path + 'Phi_%d.png' %cnt_save, I_phi, cmap = 'rainbow')
    plt.imsave(dump_path + 'Phi_masked_%d.png' %cnt_save, I_phi_masked, cmap = 'rainbow')
    
    np.save(dump_path + 'corr_coef.npy', corr_coef_save)
    
    return I_phi_masked