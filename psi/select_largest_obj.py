
"""
Automatically searching for and selecting the region of interest for 
phase shift interferometry in order to remove spurious/unwanted regions. 
"""
import cv2 
import numpy as np

def select_largest_obj(img_bin, lab_val=1, fill_holes=True,
                       smooth_boundary=True, kernel_size=5):
        '''Select the largest object from a binary image and optionally
            fill holes inside it and smooth its boundary.
        Args:
            img_bin (2D array): 2D numpy array of binary image.
                (e.g. a thresholded intensity modulation image.)
            lab_val ([int]): integer value used for the label of the largest 
                    object. Default is 1.
            fill_holes ([boolean]): whether fill the holes inside the largest 
                    object or not. Default is True.
            smooth_boundary ([boolean]): whether smooth the boundary of the 
                    largest object using morphological opening or not. Default 
                    is True. It is also useful for noise removal 
            kernel_size ([int]): the size of the kernel used for morphological 
                    operation. Default is 5.
        Returns:
            a binary image as a mask for the largest object.
        '''
        img_bin = np.uint8(img_bin)
        if smooth_boundary:
            kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
            img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, 
                                            kernel_)
        
        n_labels, img_labeled, lab_stats, _ = \
            cv2.connectedComponentsWithStats(img_bin, connectivity=8, 
                                             ltype=cv2.CV_32S)
        largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
        largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
        largest_mask[img_labeled == largest_obj_lab] = lab_val
        
        
        if fill_holes:
            
            ## to avoid touching-boundaries to block flood
            largest_mask_ = np.zeros((img_bin.shape[0] + 10,
                                      img_bin.shape[1] + 10),
                                     dtype=np.uint8)
            largest_mask_[5:-5,5:-5] = largest_mask
            largest_mask = largest_mask_
            ##
            
            bkg_seed = (0,0)
            img_floodfill = largest_mask.copy()
            h_, w_ = largest_mask.shape
            mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
            cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, 
                          newVal=lab_val)
            
            holes_mask = lab_val - img_floodfill # mask of the holes.
            largest_mask = largest_mask + holes_mask
            largest_mask = largest_mask[5:-5,5:-5]
            
        return largest_mask