
"""
Records video from streaming camera and save it to a file.
It supports upto two cameras at the same time. 
Video files are appended with camera ID {0 and/or 1}
 
 --vid_rate: is set to 30 for "usb" cameras,
             and can be ignored for synced cameras e.g. "flir",
             since it only controls representation rate of video

"""

import cv2
import os
import argparse
import time
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--vid_rate", type=float, default=10,
                    help="video frame rate")
parser.add_argument("--num", type=int, default=600,
                    help="number of images to save")
parser.add_argument("--vid_name", type=str, default= "video.mp4",
                    help="name of video file with extension")
parser.add_argument("--dump_path", type=str, default= "./apsi_files/",
                    help="path to dump psi video")
parser.add_argument("--cam_height", type=int, default= 512,
                    help="height of camera image")
parser.add_argument("--cam_width", type=int, default= 640,
                    help="width of camera image")
parser.add_argument("--cam_num", type=int, default= 1, choices = [1,2],
                    help="number of cameras")
parser.add_argument("--cam_type", type=str, default= "flir", choices = ["usb","flir"],
                    help="type of cameras")

args = parser.parse_args()


if args.cam_type == "usb":
    cap = [None]*args.cam_num
    for i in range(args.cam_num):
        cap[i] = cv2.VideoCapture(i)
        cap[i].set(5, 30) # set frame rate to 30 frames per second
        args.vid_rate = 30
elif args.cam_type == "flir":
    import PySpin
    from flir_camera_setup import *

    class cam_params:
        frame_rate_enable = False #(True, False)
        frame_rate_to_set = 30.0
        exposure_auto = 'Continuous' #(Off, Continuous)
        exposure_time_to_set = 6700
        gain_auto = 'Continuous' #(Off, Continuous)
        gain_to_set = 4.0
        white_auto = 'Continuous' #(Off, Continuous)
        gamma_to_set = 1.0
        trigger_mode = 'On' #(Off, On)
        CHOSEN_TRIGGER = 2 # 1 for software, 2 for hardware
        line_num = 3
        trigger_edge = 'FallingEdge' #(RisingEdge, FallingEdge)
        pixel_format = "mono8" #rgb8
    
    for i in range(args.cam_num):
        configure_camera(cam_params, cam_id = i)
    
    system = PySpin.System.GetInstance()
    cam_list = system.GetCameras()
    cam = [None]*args.cam_num
    for i in range(args.cam_num):
        _, cam[i] = list(enumerate(cam_list))[i]
        cam[i].Init()
        cam[i].BeginAcquisition()


save_path = args.dump_path
if not os.path.exists(save_path):
    os.mkdir(save_path)

video = [None]*args.cam_num
for i in range(args.cam_num):
    video[i] = cv2.VideoWriter(save_path + args.vid_name[:-4] + "_%d.mp4" %i,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               args.vid_rate,
                               (args.cam_width,args.cam_height))

times = np.zeros(args.num)
frame = [None]*args.cam_num
for cnt in range(args.num):
    if args.cam_type == "usb":
        for i in range(args.cam_num):
            ret, frame[i] = cap[i].read()
            frame[i] = cv2.resize(frame[i], (args.cam_width, args.cam_height))
    elif args.cam_type == "flir":
        for i in range(args.cam_num):
            if i == 0:
                t1 = time.time()
            image_result = cam[i].GetNextImage()
            if i == 0:
                t2 = time.time()
            width = image_result.GetWidth()
            height = image_result.GetHeight()
            frame[i] = image_result.GetData().reshape(height, width)
            frame[i] = cv2.resize(frame[i], (args.cam_width, args.cam_height))
            frame[i] = cv2.cvtColor(frame[i], cv2.COLOR_GRAY2BGR)
            image_result.Release()
        times[cnt] = t2 - t1
    for i in range(args.cam_num):
        video[i].write(frame[i])
        

cv2.destroyAllWindows()
for i in range(args.cam_num):
    video[i].release()

if args.cam_type == "flir":
    print("received frame rate = %.2f" %(1/times[:-1].mean()))    

