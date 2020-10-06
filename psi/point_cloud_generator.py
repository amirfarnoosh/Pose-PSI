
"""
including functions to handle transformation matrix generation
and point cloud generation.
"""

import numpy as np
from numpy.linalg import inv
import cv2

def cam_proj_calib_mat_generator(calib_path, img_num = 0, cam_num = 1):
    
    '''load camera(s) and projector calibration files and generates
        their correspponding transformation matrices 
    Args:
        calib_path (str): top-level path to calibration files.
            (a folder including "projector_view", "camera_view_0" ,..)
        img_num ([int]): image number to consider for reference coordinate.
            Default is image number 0.
        cam_num (int): number of cameras used for point cloud generation.
    Returns:
        rt_cam (tuple): a tuple including camera(s) transformation matrix
        rt_proj (2D array): a 2D array of size [4,4] 
            as projector transformation matrix
    '''
    # load calibration parameters of camera and projector
    rt_cam = []
    for i in range(cam_num):
        load_path_cam = calib_path + 'camera_%d_view/' %i
        
        mtx = np.load(load_path_cam + 'cam_mtx.npy')
        rvecs = np.load(load_path_cam + 'cam_rvecs.npy')
        tvecs = np.load(load_path_cam + 'cam_tvecs.npy')
        
        # reference coordinate is img_num.
        rvec= rvecs[img_num]
        tvec = tvecs[img_num]
        
        # rotation and translation matrix
        rt_cam.append(np.matmul(mtx, 
                                np.concatenate((cv2.Rodrigues(rvec)[0], tvec), axis = 1)))
        
    load_path_proj = calib_path + 'projector_view/'
    mtx_proj = np.load(load_path_proj + 'cam_mtx.npy')
    rvecs_proj = np.load(load_path_proj + 'cam_rvecs.npy')
    tvecs_proj = np.load(load_path_proj + 'cam_tvecs.npy')
    rvec_proj= rvecs_proj[img_num]
    tvec_proj = tvecs_proj[img_num]
    
    rt_proj = np.matmul(mtx_proj, 
                        np.concatenate((cv2.Rodrigues(rvec_proj)[0], tvec_proj), axis = 1))
    
    return rt_cam, rt_proj 

def point_cloud_generator(phi_unwrapped, w_proj, p, rt_cam, rt_proj, s=(5,5)):
    '''generating point cloud from a given unwrapped phase map.
    Args:
        phi_unwrapped (2D array): unwrapped phase map.
        w_proj ([int]): width of projector image.
            Default is image number 0.
        cam_num (int): number of cameras used for point cloud generation.
        p (int): 
    Returns:
        rt_cam (tuple): a tuple including camera(s) transformation matrix
        rt_proj (2D array): a 2D array of size [4,4] 
            as projector transformation matrix
    '''
        
    h, w = phi_unwrapped.shape
    u_ps = np.round(p * phi_unwrapped / (2 * np.pi)).astype('int')
    u_ps = (w_proj - 1) - u_ps
    
    # Mask generation-- you can remove mask_u
    mask_s = np.zeros((h, w), dtype=np.bool)
    mask_s[::s[0], ::s[1]] = True
    mask_u = np.ones((h, w), dtype=np.bool)
    mask_u[u_ps < 0] = False
    mask_u[u_ps > w_proj - 1] = False
    mask_idxs = np.where(mask_s * mask_u * np.invert(phi_unwrapped.mask) == 1)
    
    ### u_ps = np.clip(u_ps, 0, w_proj - 1) --use this if mask_u is removed
    
    pts_cloud = [np.matmul(inv(np.asarray([mask_idxs[1][i] * rt_cam[2, :-1] - rt_cam[0, :-1],
                                           mask_idxs[0][i] * rt_cam[2, :-1] - rt_cam[1, :-1],
                                           u_ps[mask_idxs[0][i], mask_idxs[1][i]] 
                                           * rt_proj[2, :-1] - rt_proj[0, :-1]])),
                           np.asarray([rt_cam[0, -1] - mask_idxs[1][i] * rt_cam[2, -1],
                                       rt_cam[1, -1] - mask_idxs[0][i] * rt_cam[2, -1],
                                       rt_proj[0, -1] - u_ps[mask_idxs[0][i], mask_idxs[1][i]]
                                       * rt_proj[2, -1]])).reshape(-1)
                                                            for i in range(len(mask_idxs[0]))]
    
    return np.asarray(pts_cloud), mask_idxs
