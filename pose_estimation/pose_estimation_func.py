
from relative_pose_cv2 import *
from relative_pose_pycpd import *
import numpy as np
from numpy.linalg import inv
import math
import cv2
import sys
sys.path.append(sys.path[0][:-15] + 'psi')
from plot_surface_plotly import *
import os

def isclose(x, y, rtol=1.e-5, atol=1.e-8):
    return abs(x-y) <= atol + rtol * abs(y)

def euler_angles_from_rotation_matrix(R):
    '''
    From a paper by Gregory G. Slabaugh (undated),
    "Computing Euler angles from a rotation matrix
    '''
    phi = 0.0
    if isclose(R[2,0],-1.0):
        theta = math.pi/2.0
        psi = math.atan2(R[0,1],R[0,2])
    elif isclose(R[2,0],1.0):
        theta = -math.pi/2.0
        psi = math.atan2(-R[0,1],-R[0,2])
    else:
        theta = -math.asin(R[2,0])
        cos_theta = math.cos(theta)
        psi = math.atan2(R[2,1]/cos_theta, R[2,2]/cos_theta)
        phi = math.atan2(R[1,0]/cos_theta, R[0,0]/cos_theta)
    return psi, theta, phi


def pose_estimation_func(src_id, dst_id, pts_path, fig_flag=True, bias = 0, load_pose=False, cam_num = 1):
    
    dump_path = pts_path + "estimated_poses/"
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)
    
    load_path = [None]*cam_num
    for i in range(cam_num):
        load_path[i] = pts_path + "camera_%d/" %i
        
    try:
        times_save = np.load(load_path[0] + 'times.npy')
    except:
        times_save = np.tile(np.arange(0, max(src_id, dst_id),1).reshape(-1,1), (1,2))
    
    corr_coef_save = [None]*cam_num
    for i in range(cam_num):
        corr_coef_save[i] = np.load(load_path[i] + 'corr_coef.npy', allow_pickle=True).reshape(1,-1)
    corr_coef_save = np.concatenate(corr_coef_save)
    
    srcPC = [None]*cam_num
    srcMask = [None]*cam_num
    dstPC = [None]*cam_num
    dstMask = [None]*cam_num
    src_rgb = [None]*cam_num
    dst_rgb = [None]*cam_num
    for src_idx, dst_idx in [(src_id,dst_id)]:
        for i in range(cam_num):
            srcPC[i] = np.load(load_path[i] + 'pts_cloud_%d.npy' %src_idx)
            srcMask[i] = np.load(load_path[i] + 'mask_idxs_%d.npy' %src_idx)
            dstPC[i] = np.load(load_path[i] + 'pts_cloud_%d.npy' %dst_idx) + bias # (bias could be added to increase robustness and escape local minimums)
            dstMask[i] = np.load(load_path[i] + 'mask_idxs_%d.npy' %dst_idx)
            src_rgb[i] = cv2.imread(load_path[i] + 'frame_rgb_%d.png' %src_idx)
            dst_rgb[i] = cv2.imread(load_path[i] + 'frame_rgb_%d.png' %dst_idx)
            dst_rgb[i] = dst_rgb[i] * 0
            dst_rgb[i][:,:,0] = 255
        
        if load_pose == False:
            
            #scale, tform = relative_pose_pycpd(np.concatenate(srcPC), np.concatenate(dstPC))
            retval, residual, pose = relative_pose_cv2(np.concatenate(srcPC),
                                                       np.concatenate(dstPC))
            
            pose_est = pose * 1
            pose_est[:-1,-1] = pose_est[:-1,-1] - bias # trasforming pose back to its real values
            np.save(dump_path + "pose_%d_to_%d.npy" %(src_idx, dst_idx), pose_est)
            
        else:
            retval = np.load(dump_path + 'retval.npy')[dst_id]
            residual = np.load(dump_path+'residual.npy', residual)[dst_id]
            pose_est = np.load(dump_path + "pose_%d_to_%d.npy" %(src_idx, dst_idx))        
            pose = pose_est * 1
            pose[:-1,-1] = pose[:-1,-1] + bias # trasforming pose back to its real values
        
        psi, theta, phi = euler_angles_from_rotation_matrix(pose_est[:-1,:-1])
        print('\n psi = %.2f, theta = %.2f, phi = %.2f' %(psi * 180/np.pi, theta * 180/np.pi, phi * 180/np.pi))
        print('tx = %.2f, ty = %.2f, tz = %.2f' %(pose_est[0,-1], pose_est[1,-1], pose_est[2,-1]))
        print('corr_coefs = %.2f, %.2f' %(corr_coef_save[:,src_idx].mean(),corr_coef_save[:,dst_idx].mean()))
        
        if fig_flag == True:
            
            if not os.path.exists(dump_path + "3d_files/"):
                os.mkdir(dump_path + "3d_files/")
            if not os.path.exists(dump_path + "overlay_files/"):
                os.mkdir(dump_path + "overlay_files/")
            if not os.path.exists(dump_path + "registration_files/"):
                os.mkdir(dump_path + "registration_files/")
                
            src_file_name = dump_path + "3d_files/" + 'obj_%d___%.2f_%.2f.html' %(src_idx, times_save[src_idx,0], times_save[src_idx,1])
            dst_file_name = dump_path + "3d_files/" + 'obj_%d___%.2f_%.2f.html' %(dst_idx, times_save[dst_idx,0], times_save[dst_idx,1])
            src_dst_file_name = dump_path + "overlay_files/" + 'obj_%d_%d.html' %(src_idx, dst_idx)
            dst_to_src_file_name = dump_path + "registration_files/" + 'obj_%d_to_%d.html' %(dst_idx, src_idx)
            
            dst_to_src_PC = [None]*cam_num
            data_src = [None]*cam_num
            data_dst = [None]*cam_num
            data_dst_to_src = [None]*cam_num
            for i in range(cam_num):
                dst_to_src_PC[i] = np.matmul(np.concatenate((dstPC[i],
                                                             np.ones((dstPC[i].shape[0], 1))), axis = 1), inv(pose).T)[:,:-1]
            
            
                data_src[i], layout = plot_surface_plotly(srcPC[i],
                                                          src_rgb[i],
                                                          srcMask[i],
                                                          times_save[src_idx],
                                                          axis_flag = True,
                                                          show = False)
                data_dst[i], _ = plot_surface_plotly(dstPC[i],
                                                     dst_rgb[i],
                                                     dstMask[i],
                                                     times_save[dst_idx],
                                                     axis_flag = True,
                                                     show = False)
        
                data_dst_to_src[i], _ = plot_surface_plotly(dst_to_src_PC[i],
                                                            dst_rgb[i],
                                                            dstMask[i],
                                                            times_save[dst_idx],
                                                            axis_flag = True,
                                                            show = False)
            
            fig1_1 = go.Figure(data = data_src, layout=layout)
            plot(fig1_1, filename=src_file_name, auto_open=False)
            
            fig2_1 = go.Figure(data = data_dst, layout=layout)
            plot(fig2_1, filename=dst_file_name, auto_open=False)
            
            fig2_2 = go.Figure(data = data_src + data_dst, layout=layout)
            plot(fig2_2, filename=src_dst_file_name, auto_open=False)
            
            fig1 = go.Figure(data = data_src + data_dst_to_src, layout=layout)
            plot(fig1, filename=dst_to_src_file_name, auto_open=False)
            
        
        return [psi,
                theta,
                phi,
                pose_est[0,-1],
                pose_est[1,-1],
                pose_est[2,-1]], times_save[dst_idx].mean(), corr_coef_save[:,dst_idx], retval, residual