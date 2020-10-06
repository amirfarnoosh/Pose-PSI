
"""
performs automatic phase shift interferometry (PSI) given 
a video of projected patterns. This script supports gray-scale patterns
with hardware or color-coded synchronization as well as rainbow patterns.
It supports both one, and two cameras.
"""

import numpy as np
import cv2
from point_cloud_generator import *
from esync_func import *
from compute_phase_map import *
from phase_scaling import *
from save_psi_files import *
import os


def PSI_automatic_func(video_name,
                       rt_cam,
                       rt_proj,
                       cntr_flag = True,
                       p = 25,
                       s = (5,5),
                       thr_cntr = 0.5,
                       cntr_thr = 1.2,
                       fr_cam=30,
                       w_proj = 640,
                       cam_num = 1,
                       apsi_path = "./apsi_files/",
                       
                       walk = False,
                       vid_cap = None,
                       times_save = None,
                       corr_coef_save = None,
                       
                       lag=3,
                       delay_proj = 0.2,
                       tol_bias = 5,
                       color_diff_thr=60,
                       pixel_thr=1000,
                       intensity_thr =80,
                       
                       use_cm = False,
                       cc_path = "./miscellaneous/color_calibration_files/",
                       
                       cnt_save = 0,
                       cnt_frame = 0,
                       max_save = 500,
                       save_flag = True,
                       time_seq = None,
                       I_seq = None,
                       rgb_seq = None,
                       pattern_id = None):
    
    '''Apply automatic phase shift interferometry (PSI) given 
        a video of projected patterns.
    Args:
        video_name (string): name of video file (with extension) to load. 
            a folder with this name will be created to dump PSI results.
        rt_cam (tuple): a tuple including camera(s) transformation matrix.
        rt_proj (2D array): an array of size [4,4] including projector
            transformation matrix.
        cntr_flag (boolean): set True if projecting centerline pattern,
            otherwise False.
        p (int): integer specifying phase pitch of patterns.
        s (tuple): integer tuple of length 2, specifying downsampling rate
            for point cloud computation
        thr_cntr (float): float number in the range of [0,1] for
            thresholding centerline image. The closer to 1, the less
            number of centerline points are detected.
        cntr_thr (float): float number for thresholding intensity
            modulation image to mask out unwanted regions in phase map. 
            Typical values are between [0,3]. The larger the value,
            the more regions will be filterd out in phase map.
        fr_cam (float): frame rate of camera.
        w_proj (int): width of projector image.
        cam_num (int): number of cameras being used.
        apsi_path (string): path to load video from, or
            top-level path to dump point cloud results.
        walk (boolean): whether to return results each time
            a point cloud is generated.
        vid_cap (opencv video cap): if provided will use it to continue
            processing the video. It can be used with "walk" argument
            to walk through the video by recursively feeding function
            with its own ouputs.
        times_save (2D array): 2D array of size [M, 2] including/for saving
            time stamps of point cloud generation. It can be used with
            "walk" argument. Default is None. (optional)
        corr_coef_save (2D array): 2D array of size [cam_num, M]
            including/for saving goodness of point cloud estimation. 
            It can be used with "walk" argument. 
            Default is None. (optional)
        lag (int): integer specifying lag (in number of frames) after
            detection of a pattern to beat tearing. 
            Used only if gray-scale color-coded patterns are projected.
        delay_proj (float): period of pattern projection.
            Used only if gray-scale color-coded patterns are projected.
        tol_bias (int): tolerance bias (in number of frames) 
            for detecting a sequence of patterns. 
            Used only if gray-scale color-coded paaterns are projected.
        color_diff_thr (uint8): a threshold for detecting colors in 
            color-coded patterns.
        pixel_thr (int): a threshold for number of pixels
            with a specific color in color-coded patterns.
        intensity_thr (uint8): intensity threshold for detecting patterns
            from no-pattern frames in color-coded patterns.
        cnt_save (int): save new files starting at this number. 
            It can be set with "walk" argument.
        cnt_frame (int): an integer pointing to the start time of video.
        max_save (int): maximum number of result files to save.
        save_flag (boolean): whether to save results or not.
    Returns: (if "walk" is True)
        vid_cap,
        cnt_save,
        cnt_frame,
        times_save,
        corr_coef_save: all have the same definition as input.
        I_1, I_2, I_3, I_c: all pattern images
        I_phi_masked: masked phase map image
    '''
    
    ### Initialization of parameters
    
    if vid_cap is None:
        cap = [None]*cam_num
        for i in range(cam_num):
            cap[i] = cv2.VideoCapture(apsi_path + video_name[:-4] + "_%d.mp4" %i)
    else:
        cap = vid_cap
        cam_num = len(cap)
    
    # dump path
    dump_path_top = apsi_path + video_name.split(".")[0] + "/"
    if not os.path.exists(dump_path_top):
            os.mkdir(dump_path_top)
    dump_path = [None]*cam_num 
    for i in range(cam_num):
        dump_path[i] = apsi_path + video_name.split(".")[0] + "/" + "camera_%d/" %i 
        if not os.path.exists(dump_path[i]):
            os.mkdir(dump_path[i]) 

    if times_save is None:
        times_save = np.zeros((max_save, 2))
    else:
        max_save = len(times_save)
    
    masks = [None] * 3
    if corr_coef_save is None:
        corr_coef_save = np.zeros((cam_num, max_save))
    
    if cntr_flag == True:
        n_pattern = 4
    else:
        n_pattern = 3
    
    if walk == False:
        tgt = 0 # target pattern ID 
        t = [None]*n_pattern
        I = [[None for i in range(n_pattern)] for j in range(cam_num)]
        I_surf = [[None for i in range(n_pattern)] for j in range(cam_num)]
    else:
        tgt = pattern_id # target pattern ID 
        t = time_seq
        I = I_seq
        I_surf = rgb_seq
    
    eflag= False # when a sequence is found it turns to True
    ref_time = 0 # reference time (first frame) 
    
    frame = [None]*cam_num
    frame_rgb = [None]*cam_num
    I_1 = [None]*cam_num
    I_2 = [None]*cam_num
    I_3 = [None]*cam_num
    I_c = [None]*cam_num
    corr_coef = [None]*cam_num
    I_phi_masked = [None]*cam_num
    ### -----------------------------
    
    ### initialize color-coded sync 

    cnt = 0
    cnt_pattern = [0 for _ in range(n_pattern)]
    tol = int((fr_cam * delay_proj - lag - 1) * (n_pattern - 1) + tol_bias)
    ###----------------


    ### processing starts here
    while(cap[0].isOpened()):
        
        for i in range(cam_num):
            ret, frame[i] = cap[i].read()
            if frame[i] is None and walk == False:
                return
            elif frame[i] is None:
                return [None]*14
            
            frame_rgb[i] = frame[i] * 1
            frame[i] = cv2.cvtColor(frame[i], cv2.COLOR_BGR2GRAY)
        
        cnt_frame += 1
        
        f_time = cnt_frame/fr_cam - ref_time 
        
        ## search for patterns

        p_num, mask = esync_func(frame_rgb[0],
                                     color_diff_thr,
                                     pixel_thr,
                                     intensity_thr)
        if p_num == tgt:
            if tgt > 0:
                cnt_pattern[tgt-1] = 0
            cnt_pattern[tgt] += 1
            if cnt_pattern[tgt] == lag:
                eflag == True
                cnt_pattern[tgt] = 0
                t[tgt] = f_time
                if tgt == 0:
                    cnt = 0
                masks[tgt] = mask
                for i in range(cam_num):
                    I[i][tgt] = frame[i]
                    I_surf[i][tgt] = frame_rgb[i]
                if tgt == n_pattern-1:
                    tgt = 0
                else:
                    tgt += 1
        else:
            cnt_pattern = [0 for _ in range(n_pattern)]
            cnt += 1
        
        if cnt > tol: # delay tolerance
            I = [[None for i in range(n_pattern)] for j in range(cam_num)]
            I_surf = [[None for i in range(n_pattern)] for j in range(cam_num)]
            cnt = 0
            tgt = 0

        if (I[0][-1] is not None) and eflag:
            for i in range(cam_num):
                I_1[i] = I[i][0].astype('float')
                I_2[i] = I[i][1].astype('float')
                I_3[i] = I[i][2].astype('float')
                if cntr_flag == True:
                    I_c[i] = I[i][3].astype('float')
                frame_rgb[i] = 1/3*(I_surf[i][0].astype('float')+
                                    I_surf[i][1].astype('float')+
                                    I_surf[i][2].astype('float'))
                frame_rgb[i] = np.uint8(frame_rgb[i])
            cnt = 0
        else:
            eflag = False
         
        ###---if patterns are detected----
        
        if eflag:
            
            start_time = min(t)
            end_time = max(t)
            for i in range(cam_num):
                
                # mask index generation
                if cam_num == 1:
                    emask = 1 - masks[0] * masks[1] * masks[2]
                else:
                    emask = 1
                phi_unwrapped = compute_phase_map(I_1[i], I_2[i], I_3[i], cntr_thr, emask)
            
                ### point cloud generation 
                phi_unwrapped, corr_coef[i] = phase_scaling(phi_unwrapped, p, w_proj, I_c[i], thr_cntr)
        
                pts_cloud, mask_idxs = point_cloud_generator(phi_unwrapped, w_proj, p, rt_cam[i], rt_proj, s)
                
                if save_flag == True: 
                    
                    print('captured at %.2f-%.2f with corr %.4s' %(start_time, end_time, corr_coef[i]))
                    times_save[cnt_save] = [start_time, end_time]
                    corr_coef_save[i,cnt_save] = corr_coef[i]

                    I_phi_masked[i] = save_psi_files(dump_path[i],
                                                     cnt_save,
                                                     phi_unwrapped,
                                                     pts_cloud,
                                                     mask_idxs,
                                                     frame_rgb[i],
                                                     I_1[i],
                                                     I_2[i],
                                                     I_3[i],
                                                     corr_coef_save,
                                                     times_save)
                    
            cnt_save += 1
            if cnt_save == max_save:
                cnt_save = 0
            eflag = False
            if walk == True:
                return cap,\
                        cnt_save,\
                        cnt_frame,\
                        times_save,\
                        corr_coef_save,\
                        list(np.uint8(I_1)), list(np.uint8(I_2)), list(np.uint8(I_3)), list(np.uint8(I_c)),\
                        I_phi_masked,t, I, I_surf, tgt
                    
