
"""
This script will do phase shift interferometry (PSI) given a video and/or 
do pose estimation thereafter if corresponding flags are set.
"""

from pose_estimation_func import *
import sys
sys.path.append(sys.path[0][:-15] + 'psi')
from point_cloud_generator import cam_proj_calib_mat_generator
import argparse
from glob import glob
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--psi", action="store_true", help="run automatic psi")
parser.add_argument("--pose", action="store_true", help="run automatic pose estimation")
parser.add_argument("--cam_num", type=int, default= 1, choices = [1,2],
                    help="number of cameras")

parser.add_argument("--load_path", type=str, default= "./apsi_files/",
                    help="path to load automatic psi files")
parser.add_argument("--vid_name", type=str, default= "video_0.mp4",
                    help="name of video file to load")
parser.add_argument("--orig_id", type=int, default= 0,
                    help="ID of origin point cloud to which poses are estimated")
parser.add_argument("--bias", type=float, default= 0,
                    help="bias given to point clouds")
parser.add_argument("--calib_path", type=str, default= "./chessboard_calibration_images/",
                    help="top-level path to calibration files")
parser.add_argument("--cam_rate", type=float, default= 30,
                    help="frame rate of camera")
parser.add_argument("--sync", action="store_true",
                    help="whether camera/projector are synced")
parser.add_argument("--rgb", action="store_true",
                    help="whether we are dong rgb psi")
parser.add_argument("--proj_width", type=int, default= 640,
                    help="width of projector image")
parser.add_argument("--lag", type=int, default= 3,
                    help="frame lag for stabilizing pattern")
parser.add_argument("--delay", type=float, default= 0.2,
                    help="projector delay for each pattern in seconds")
parser.add_argument("--tol_bias", type=int, default= 5,
                    help="bias of tolerence for pattern detection")
parser.add_argument("--pitch", type=int, default= 40,
                    help="fringe phase pitch")
parser.add_argument("--s", type=int, default= 5,
                    help="down sampling factor for point cloud generation")
parser.add_argument("--color_thr", type=int, default= 60,
                    help="color difference threshold for color-encoded synchronization")
parser.add_argument("--pixel_thr", type=int, default= 1000,
                    help="pixel number threshold for color-encoded synchronization")
parser.add_argument("--int_thr", type=int, default= 80,
                    help="intensity threshold for color-encoded synchronization")
parser.add_argument("--thr_cntr", type=float, default= 0.7,
                    help="centerline pattern threshold ratio")
parser.add_argument("--mask_thr", type=float, default= 1.2,
                    help="mask threshold ratio to remove spurious regions")
parser.add_argument("--max_save", type=int, default= 100,
                    help="maximum number of models to save")
parser.add_argument("--cc", action="store_true", help="use color correction")
parser.add_argument("--cc_path", type=str, default= "./color_calibration_files/",
                    help="path to color calibration files")

args = parser.parse_args()
import sys
import pdb
pdb.set_trace()

if args.sync:    
    from PSI_automatic_sync import *
elif not args.rgb:
    from PSI_automatic_esync import *
if args.rgb:
    from PSI_automatic_rgb import *

if args.psi:
    
    rt_cam, rt_proj = cam_proj_calib_mat_generator(args.calib_path,
                                                   img_num = 0,
                                                   cam_num = args.cam_num)
    
    PSI_automatic_func(args.vid_name,
                       rt_cam,
                       rt_proj,
                       lag=args.lag,
                       fr_cam=args.cam_rate,
                       delay_proj = args.delay,
                       tol_bias = args.tol_bias,
                       p = args.pitch,
                       s = (args.s,args.s),
                       thr_cntr = args.thr_cntr,
                       color_diff_thr=args.color_thr,
                       pixel_thr=args.pixel_thr,
                       intensity_thr =args.int_thr,
                       cntr_thr = args.mask_thr,
                       apsi_path = args.load_path,
                       w_proj = args.proj_width,
                       max_save = args.max_save,
                       use_cm = args.cc,
                       cc_path = args.cc_path,
                       cntr_flag = True,
                       cam_num = args.cam_num
                       )
if args.pose:
    
    pts_path = args.load_path + args.vid_name.split('.')[0] + "/"
    
    N = len(glob(pts_path + "camera_0/" + "pts_cloud*.npy"))
    src_id = args.orig_id
    
    """
    psi = 1st column
    theta = 2nd column
    phi = 3rd column
    tx = 4th column
    ty = 5th column
    tz = 6th column 
    time = 7th column
    corr_coef = 8th column
    """
    poses = np.zeros((N, 8))
    retval = np.zeros(N, dtype = np.bool)
    residual = np.zeros(N)
    
    for dst_id in range(N):
        
        pose,\
        time,\
        corr_coef, retval[dst_id], residual[dst_id] = \
        pose_estimation_func(src_id, dst_id, pts_path, fig_flag=False,
                             bias = args.bias, cam_num = args.cam_num)
        
        psi, theta, phi, tx, ty, tz = pose
        poses[dst_id] = psi, theta, phi, tx, ty, tz, time, min(corr_coef)
    
    dump_path = pts_path + "estimated_poses/"
    np.save(dump_path+'poses.npy', poses)
    np.save(dump_path+'retval.npy', retval)
    np.save(dump_path+'residual.npy', residual)