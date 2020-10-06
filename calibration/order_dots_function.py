""" Functions to arrange projected dots in order
    from left to right and top to down for finding 
    correspondence between dots coordinates in projector
    and camera image coordinates """

import numpy as np
from numpy.linalg import norm
import cv2


def func(p0, p1, p, n): 
    
    """ Select and then arrange dots along a specific direction
        specified by its end points.
    Args:
        p0 (1D array): An ndarray of size [2,] to specify 
            starting point of the direction.
        p1 (1D array): An ndarray of size [2,] to specify
            end point of the direction.
        p (2D array): An ndarray of size [N,2] containing
            all dots coordinates in image. 
        n ([int]): An integer to specify number of expected
            dots along the direction.
    Returns:
        An ndarray of size [n,2] containing dots in order along the direction
    """
    
    d = [norm(np.cross(p1-p0, p0-p[i]))/norm(p1-p0) for i in range(len(p))]
    d = np.asarray(d)
    idx = np.argsort(d)[0:n]
    
    dist = [norm(p0-p[idx[i]]) for i in range(len(idx))]
    dist = np.asarray(dist)
    
    idx_dist = np.argsort(dist)
    p_ordered = p[idx[idx_dist]]
    
    return p_ordered


def order_dots(pts, grid, img, refPt = None):
    
    """ Arrange and then number grid of dots in a binary image
        in order from left to right and top to down.
    Args:
        pts (2D array): An ndarray of size [N,2] containig
            all dots coordinates in image.
        grid (1D tuple): A tuple containig two integer values 
            to specify grid height, and width (in number of dots).
        img (2D array): A binary image of (0, 255) containing dots. 
        refPt (2D tuple): A 2D tuple of size [4,2] containig coordinates of
            corners of grid (roughly) clockwise starting from top-left corner.
            Default is None, and the function proceeds automatically,
            but if provided will use them for arrangement.
    Returns:
        pts_ord (2D array): An ndarray of size [N,2] 
            containing arranged dots coordinates in order
        img (2D array): input image ovelayed with numbered dots
            to help visually check result
    """
    
    if refPt == None:
        h, w = img.shape
        p00 = [0, 0]
        p10 = [w, 0]
        p20 = [w, h]
        p30 = [0, h]
    else:
        p00 = [refPt[0][0],refPt[0][1]]
        p10 = [refPt[1][0],refPt[1][1]]
        p20 = [refPt[2][0],refPt[2][1]]
        p30 = [refPt[3][0],refPt[3][1]]
        

    p0 = pts[np.argmin(norm(pts - p00, axis = 1))]
    p1 = pts[np.argmin(norm(pts - p10, axis = 1))]
    p2 = pts[np.argmin(norm(pts - p20, axis = 1))]
    p3 = pts[np.argmin(norm(pts - p30, axis = 1))]
        
    # p0, p1, p2, p3 #clockwise from top-left
    
    pts_l = func(p0, p3, pts, grid[0])
    pts_r = func(p1, p2, pts, grid[0])

    pts_ord = np.array([]).reshape(0,2)
    for i in range(grid[0]):
        
        p_ord = func(pts_l[i], pts_r[i], pts, grid[1])
        pts_ord = np.concatenate((pts_ord, p_ord), axis = 0)

    pts_ord_int = np.round(pts_ord).astype('int')    
    
    # draw points and orders
    
    img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for i in range(len(pts_ord_int)):
    
        # Write some Text
        font = cv2.FONT_HERSHEY_SIMPLEX
        bottomLeftCornerOfText = (pts_ord_int[i,0], pts_ord_int[i,1])
        fontScale = 0.7
        fontColor = (255,0,0)
        lineType = 1
        cv2.putText(img,
                    '%d' %i,
                    bottomLeftCornerOfText, 
                    font, 
                    fontScale,
                    fontColor,
                    lineType)
    
        cv2.circle(img,
                   bottomLeftCornerOfText,
                   radius = 2,
                   color = (0, 0, 255),
                   thickness = 1, 
                   lineType = 2,
                   shift = 0)
    
    
    return pts_ord, img
