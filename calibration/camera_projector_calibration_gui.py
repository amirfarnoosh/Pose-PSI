#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 10:13:27 2019

@author: amirreza
"""

import numpy as np
import cv2
from glob import glob
import os
from skimage import transform as tf
from skimage.transform import warp
from order_dots_function import *
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--row", type=int, default = 8,
                    help="number of rows in chessboard")
parser.add_argument("--col", type=int, default = 10,
                    help="number of columns in chessboard")
parser.add_argument("-s", "--size", type=float, default= 7,
                    help="size of chessboard squares in millimeter")
parser.add_argument("--cam_height", type=int, default= 360,
                    help="height of camera image")
parser.add_argument("--cam_width", type=int, default= 640,
                    help="width of camera image")
parser.add_argument("--cam_num", type=int, default= 1, choices = [1,2],
                    help="number of cameras")
parser.add_argument("--cam_type", type=str, default= "flir", choices = ["usb","flir"],
                    help="type of cameras")
parser.add_argument("--load", type=bool, default= False,
                    help="load existing chessboard images")
parser.add_argument("--path", type=str, default= "./chessboard_calibration_images/",
                    help="path to dump chessboard images")
parser.add_argument("--path_dots", type=str, default= "./dots_black_images/dots.png",
                    help="path to dotted grid image file")

args = parser.parse_args()

if not os.path.exists(args.path):
    os.mkdir(args.path)

# dump path----------
dump_path = args.path
if not os.path.exists(dump_path):
    os.mkdir(dump_path)
#-----------------

# save_path---------------
save_path_cam = [None]*args.cam_num
for i in range(args.cam_num):
    save_path_cam[i] = dump_path + 'camera_%d_view/' %i
    if not os.path.exists(save_path_cam[i]):
        os.mkdir(save_path_cam[i])

save_path_proj = dump_path + 'projector_view/'
if not os.path.exists(save_path_proj):
    os.mkdir(save_path_proj)
#-------------------------------    


### Setup SimpleBlobDetector parameters---------
params = cv2.SimpleBlobDetector_Params()

# Change thresholds
params.minThreshold = 10
params.maxThreshold = 200
# Filter by Area.
params.filterByArea = True
# Filter by Circularity 
params.minCircularity = 0.1
# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87
# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01
#--------------------------------


# load dotted grid image------------
img_dots = cv2.imread(args.path_dots, 0)
dots_idxs = np.where(img_dots == 255)
grid_height = len(np.where(dots_idxs[1] == dots_idxs[1][0])[0])
grid_width = len(np.where(dots_idxs[0] == dots_idxs[0][0])[0])
grid = (grid_height,grid_width) #dots grid size
#-----------------------

# index preparation for dots in projector-----
pts_dots = np.flip(np.array(dots_idxs).T, axis = 1)
#--------------------------

# setup chessboard detector grid sizes-----------
HWs = np.mgrid[3:args.col,3:args.row].T.reshape(-1,2)
HWs = HWs[np.flip(np.argsort(np.prod(HWs, axis = 1)), axis = 0)]

# termination criteria for subpixel operations
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
calib_flags = cv2.CALIB_ZERO_TANGENT_DIST + cv2.CALIB_FIX_K1 + cv2.CALIB_FIX_K2 + cv2.CALIB_FIX_K3

# Arrays to store object points and image points from all the images.`
objpoints = [] # 3d point in real world space
imgpoints_cam = [[]]*args.cam_num # 2d points in image plane of camera.
imgpoints_proj = [] # 2d points in image plane of projector.


if sys.version_info[0] == 3:
    # for Python3
    from tkinter import *   ## notice lowercase 't' in tkinter here
else:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter
    
from PIL import Image
from PIL import ImageTk 

def cv_to_tk(frame_input):
    '''convert opencv image to tkinter image.
    Args:
        frame_input (2D array): 2D numpy array of image.
    Returns:
        tkinter version of the input image.
    '''
    image = Image.fromarray(frame_input)
    image = ImageTk.PhotoImage(image)
    return image

if args.load == False:
    if args.cam_type == "usb":
        cap = [None]*args.cam_num
        for i in range(args.cam_num):
            cap[i] = cv2.VideoCapture(i)
    elif args.cam_type == "flir":
        import PySpin
        sys.path.append(sys.path[0][:-11] + 'capture')
        from flir_camera_setup import *

        class cam_params:
            frame_rate_enable = True #(True, False)
            frame_rate_to_set = 30.0
            exposure_auto = 'Continuous' #(Off, Continuous)
            exposure_time_to_set = 6700
            gain_auto = 'Continuous' #(Off, Continuous)
            gain_to_set = 4.0
            white_auto = 'Continuous' #(Off, Continuous)
            gamma_to_set = 1.0
            trigger_mode = 'Off' #(Off, On)
            CHOSEN_TRIGGER = 2 # 1 for software, 2 for hardware
            line_num = 3
            trigger_edge = 'RisingEdge' #(RisingEdge, FallingEdge)
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
        
master = Tk()
screen_height = master.winfo_screenheight()
master.title("Camera-Projector Calibration")

### constrainting display dimensions in gui
disp_h = screen_height//5
disp_w = int(disp_h/args.cam_height * args.cam_width) 
###


img_z = np.zeros((disp_h, disp_w))
frame = [None]*args.cam_num
frame_show = [None]*args.cam_num
pts_img = [None]*args.cam_num
x0,y0,x1,y1,x2,y2,x3,y3 = [None]*8
frame_without_dots = [img_z]*args.cam_num
frame_with_dots = [img_z]*args.cam_num
pts = [None]*args.cam_num
img_diff = [None]*args.cam_num
frame_warped = [img_z]*args.cam_num
cnt = 0
refPt = [[]]*args.cam_num
img_mouse = [None]*args.cam_num
im_with_keypoints = [img_z]*args.cam_num
tform = [None]*args.cam_num
corners2 = [None]*args.cam_num
objp, corners2_proj = [None]*2

def getfirst(eventorigin_0):
    global refPt, img_mouse
    refPt[cp] = []
    x0 = eventorigin_0.x
    y0 = eventorigin_0.y
    refPt[cp].append((x0*2, y0*2))
    img_mouse[cp] = im_with_keypoints[cp] * 1
    cv2.circle(img_mouse[cp],
               center = (x0, y0),
               radius = 3,
               color = (0, 255, 0),
               thickness = 1,
               lineType = 2,
               shift = 0)
    image = cv_to_tk(img_mouse[cp])
    canvas_3.itemconfig(image_on_canvas_3, image = image)
    canvas_3.image = image
    canvas_3.bind("<Button 1>",getsecond)

def getsecond(eventorigin_1):
    global refPt
    x1 = eventorigin_1.x
    y1 = eventorigin_1.y
    refPt[cp].append((x1*2, y1*2))
    cv2.circle(img_mouse[cp],
               center = (x1, y1),
               radius = 3,
               color = (0, 255, 0),
               thickness = 1,
               lineType = 2,
               shift = 0)
    image = cv_to_tk(img_mouse[cp])
    canvas_3.itemconfig(image_on_canvas_3, image = image)
    canvas_3.image = image
    canvas_3.bind("<Button 1>",getthird)

def getthird(eventorigin_2):
    global refPt
    x2 = eventorigin_2.x
    y2 = eventorigin_2.y
    refPt[cp].append((x2*2, y2*2))
    cv2.circle(img_mouse[cp],
               center = (x2, y2),
               radius = 3,
               color = (0, 255, 0),
               thickness = 1,
               lineType = 2,
               shift = 0)
    image = cv_to_tk(img_mouse[cp])
    canvas_3.itemconfig(image_on_canvas_3, image = image)
    canvas_3.image = image
    canvas_3.bind("<Button 1>",getfourth)
    
def getfourth(eventorigin_3):
    global refPt, pts_img
    x3 = eventorigin_3.x
    y3 = eventorigin_3.y
    refPt[cp].append((x3*2, y3*2))
    cv2.circle(img_mouse[cp],
               center = (x3, y3),
               radius = 3,
               color = (0, 255, 0),
               thickness = 1,
               lineType = 2,
               shift = 0)
    image = cv_to_tk(img_mouse[cp])
    canvas_3.itemconfig(image_on_canvas_3, image = image)
    canvas_3.image = image
    pts_img[cp], img = order_dots(pts[cp], grid, img_diff[cp], refPt[cp])
    img = cv2.resize(img, (disp_w, disp_h))
    image = cv_to_tk(img)
    canvas_3.itemconfig(image_on_canvas_3, image = image)
    canvas_3.image = image
    canvas_3.bind("<Button 1>",noget)
    
def noget(eventorigin_4):
    pass


cp = 0
def callback(button_id):
    global frame_show, frame_without_dots, frame_with_dots, pts, img_diff,\
            frame_warped, pts_img, cnt, im_with_keypoints,\
            tform, objp, corners2, corners2_proj, cp
    if button_id == 0:
        if args.load == False:
            frame_without_dots = frame * 1
        else:
             for i in range(args.cam_num):   
                frame_without_dots[i] = cv2.imread(dump_path + 'chkb_%d_%.3d.png' %(i,cnt), 0)
                frame_show[i] = cv2.resize(frame_without_dots[i], (disp_w, disp_h))
        
        image = cv_to_tk(frame_show[cp])
        canvas_1.itemconfig(image_on_canvas_1, image = image)
        canvas_1.image = image
        img_z = np.zeros((disp_h, disp_w))
        frame_warped = [img_z]*args.cam_num
        im_with_keypoints = [img_z]*args.cam_num
        image = cv_to_tk(img_z)
        canvas_3.itemconfig(image_on_canvas_3, image = image)
        canvas_3.image = image
    if button_id == 1:
        if args.load == False:
            frame_with_dots = frame * 1
        else:
            for i in range(args.cam_num):
                frame_with_dots[i] = cv2.imread(dump_path + 'chkb_%d_dots_%.3d.png' %(i,cnt), 0)
                frame_show[i] = cv2.resize(frame_with_dots[i], (disp_w, disp_h))
        image = cv_to_tk(frame_show[cp])
        canvas_2.itemconfig(image_on_canvas_2, image = image)
        canvas_2.image = image
        img_z = np.zeros((disp_h, disp_w))
        frame_warped = [img_z]*args.cam_num
        im_with_keypoints = [img_z]*args.cam_num
        image = cv_to_tk(img_z)
        canvas_3.itemconfig(image_on_canvas_3, image = image)
        canvas_3.image = image
    if button_id == 2:
        thr = float(v_0.get())
        params.minArea = float(v_1.get())
        if v_2.get() == "True":
            params.filterByCircularity = True
        elif v_2.get() == "False":
            params.filterByCircularity = False
            
        img_diff[cp] = np.absolute(frame_without_dots[cp].astype('float') - frame_with_dots[cp].astype('float'))
        img_diff[cp][img_diff[cp]>thr] = 255
        img_diff[cp][img_diff[cp]<=thr] = 0
        img_diff[cp] = np.uint8(255 - img_diff[cp])
    
        # Detect blobs.
        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img_diff[cp])
        pts[cp] = np.asarray([keypoints[idx].pt for idx in range(0, len(keypoints))])
                
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints[cp] = cv2.drawKeypoints(img_diff[cp], keypoints, np.array([]), (255,0,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
     
        # Show keypoints
        im_with_keypoints[cp] = cv2.resize(im_with_keypoints[cp], (disp_w, disp_h))
        image = cv_to_tk(im_with_keypoints[cp])
        canvas_3.itemconfig(image_on_canvas_3, image = image)
        canvas_3.image = image
    
    if button_id == 3:
        pts_img[cp], img = order_dots(pts[cp], grid, img_diff[cp]) 
        img = cv2.resize(img, (disp_w, disp_h))
        image = cv_to_tk(img)
        canvas_3.itemconfig(image_on_canvas_3, image = image)
        canvas_3.image = image
    
    if button_id == 4:
        # Determine the origin by clicking
        image = cv_to_tk(im_with_keypoints[cp])
        canvas_3.itemconfig(image_on_canvas_3, image = image)
        canvas_3.image = image        
        canvas_3.bind("<Button 1>",getfirst)
    
    if button_id == 5:
        # Warp to projector view
           # to see inside an object: 
           # https://stackoverflow.com/questions/1006169/how-do-i-look-inside-a-python-object
           # dir(object), or vars(object)
        tform[cp] = tf.estimate_transform('projective', pts_dots, pts_img[cp])
        flag_warp = np.allclose(tform[cp].inverse(tform[cp](pts_dots)), pts_dots)

        if flag_warp:            
            # warp image using the estimated transformation
            frame_warped[cp] = warp(frame_without_dots[cp], inverse_map=tform[cp])
            frame_warped[cp] = np.uint8(frame_warped[cp] * 255)
            img_warped = cv2.resize(frame_warped[cp], (disp_w, disp_h))
            image = cv_to_tk(img_warped)
            canvas_3.itemconfig(image_on_canvas_3, image = image)
            canvas_3.image = image
        else:
            print('Warping Not Possible!')
    if button_id == 6:
        for i in range(args.cam_num):
            cv2.imwrite(dump_path + 'chkb_%d_dots_%.3d.png' %(i,cnt), frame_with_dots[i])
            cv2.imwrite(dump_path + 'chkb_%d_%.3d.png' %(i,cnt), frame_without_dots[i])
            cv2.imwrite(save_path_cam[i] + 'chkb_%d_%.3d.png' %(i,cnt), frame_without_dots[i])
            cv2.imwrite(save_path_proj + 'chkb_%d_%.3d.png' %(i,cnt), frame_warped[i])
            np.save(save_path_cam[i] + '/cam_%d_to_proj_%.3d.npy' %(i,cnt), tform[i]._inv_matrix)
        print('#### checkerboard images %.3d saved ####' %cnt)
        cnt+= 1
        
        image = cv_to_tk(np.zeros((disp_h, disp_w)))
        img_z = np.zeros((disp_h, disp_w))
        frame_without_dots = [img_z]*args.cam_num
        frame_with_dots = [img_z]*args.cam_num
        frame_warped = [img_z]*args.cam_num
        im_with_keypoints = [img_z]*args.cam_num
        canvas_1.itemconfig(image_on_canvas_1, image = image)
        canvas_1.image = image
        canvas_2.itemconfig(image_on_canvas_2, image = image)
        canvas_2.image = image
        canvas_3.itemconfig(image_on_canvas_3, image = image)
        canvas_3.image = image
        canvas_4.itemconfig(image_on_canvas_4, image = image)
        canvas_4.image = image
        canvas_5.itemconfig(image_on_canvas_5, image = image)
        canvas_5.image = image
        labelText_5.set("")
        v_0.set("40")
        v_1.set("3")
        v_2.set("False")
        v_3.set("")
        v_4.set("")
        
    if button_id == 7:
        # Our operations on the frame come here
        """
        **CALIB_CB_ADAPTIVE_THRESH** Use adaptive thresholding to convert the image
        to black and white
        **CALIB_CB_NORMALIZE_IMAGE** Normalize the image gamma with equalizeHist 
        before thresholding.
        **CALIB_CB_FILTER_QUADS** Use additional criteria 
        (like contour area, perimeter, square-like shape) to filter out false quads
        extracted at the contour retrieval stage.
         """
        # Find the chess board corners
        if v_3.get() == "" or v_4.get() == "":
            HW = HWs
        else:
            HW = [(int(v_3.get()), int(v_4.get()))]
        sq = args.size
        for i in range(len(HW)):
            hw = (HW[i][0],HW[i][1])
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            objp = np.zeros((np.prod(hw),3), np.float32)
            objp[:,:2] = np.mgrid[0:hw[0],0:hw[1]].T.reshape(-1,2) * sq
            
            ret, corners = cv2.findChessboardCorners(frame_without_dots[cp], hw,
                                                     flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
                        
            # If found, add object points, image points (after refining them)
            if ret == True:
        
                corners2[cp] = cv2.cornerSubPix(frame_without_dots[cp],corners,(5,5),(-1,-1),criteria)
                corners2_proj = np.matmul(tform[cp]._inv_matrix, 
                                          np.concatenate((corners2[cp].reshape(-1,2), np.ones((corners2[cp].shape[0],1))), axis = 1).T)
                corners2_proj = (corners2_proj/corners2_proj[2]).T[:,0:2]
                corners2_proj = np.expand_dims(corners2_proj, axis = 1)
                corners2_proj = np.array(corners2_proj, dtype = 'float32' )
                
                if args.cam_num == 2:

                    corners2[1-cp] = np.matmul(tform[1-cp].params, 
                                              np.concatenate((corners2_proj.reshape(-1,2), np.ones((corners2_proj.shape[0],1))), axis = 1).T)
                    corners2[1-cp] = (corners2[1-cp]/corners2[1-cp][2]).T[:,0:2]
                    corners2[1-cp] = np.expand_dims(corners2[1-cp], axis = 1)
                    corners2[1-cp] = np.array(corners2[1-cp], dtype = 'float32' )
                
            
                # Draw and display the corners
                gray = cv2.cvtColor(frame_without_dots[cp], cv2.COLOR_GRAY2BGR)
                gray = cv2.drawChessboardCorners(gray, hw, corners2[cp], ret)
                
                gray_proj = cv2.cvtColor(frame_warped[cp], cv2.COLOR_GRAY2BGR)
        
                for j in range(len(corners2_proj)):
                    # draw circles around regions of interest
                    cv2.circle(gray_proj,
                               center = (int(corners2_proj[j,0,0]),int(corners2_proj[j,0,1])),
                               radius = 3,
                               color = (255, 0, 0),
                               thickness = 1,
                               lineType = 2,
                               shift = 0)
                gray = cv2.resize(gray, (disp_w, disp_h))
                image = cv_to_tk(gray)
                canvas_4.itemconfig(image_on_canvas_4, image = image)
                canvas_4.image = image
                gray_proj = cv2.resize(gray_proj, (disp_w, disp_h))
                image = cv_to_tk(gray_proj)
                canvas_5.itemconfig(image_on_canvas_5, image = image)
                canvas_5.image = image
                v_3.set("%d" %hw[0])
                v_4.set("%d" %hw[1])
                labelText_5.set("")
                break
            elif i == len(HW) - 1:
                labelText_5.set("Chessboard not found!")
                image = cv2.resize(frame_without_dots[cp], (disp_w, disp_h))
                image = cv_to_tk(image)
                canvas_4.itemconfig(image_on_canvas_4, image = image)
                canvas_4.image = image
                image = cv2.resize(frame_warped[cp], (disp_w, disp_h))
                image = cv_to_tk(image)
                canvas_5.itemconfig(image_on_canvas_5, image = image)
                canvas_5.image = image
                

    if button_id == 8:
        objpoints.append(objp)
        for i in range(args.cam_num):
            imgpoints_cam[i].append(corners2[i])
        imgpoints_proj.append(corners2_proj)
        labelText_5.set("Points Added! (%d point(s) so far)" %len(objpoints))

    if button_id == 9:
        
        for imgpoints, save_path, name in [(imgpoints_cam, save_path_cam, "Camera"), ([imgpoints_proj], [save_path_proj], "Projector")]:
    
            # camera calibration
            for ii in range(len(imgpoints)):
                #pdb.set_trace()
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints[ii][ii::len(imgpoints)], frame_without_dots[ii].shape[::-1],None,None,flags=calib_flags)
                
                np.save(save_path[ii] + 'cam_mtx.npy', mtx)
                np.save(save_path[ii] + 'cam_rvecs', rvecs)
                np.save(save_path[ii] + 'cam_tvecs', tvecs)
                np.save(save_path[ii] + 'cam_dist', dist)
                
                """Re-projection Error"""
                tot_error = 0
                for i in range(len(objpoints)):
                    imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
                    error = cv2.norm(imgpoints[ii][ii::len(imgpoints)][i],imgpoints2, cv2.NORM_L2)/len(imgpoints2)
                    tot_error += error
                if name == "Camera" and ii == 0:
                    labelText_6.set(name + "0 mean error: %.2f" %(tot_error/len(objpoints)))
                elif name == "Camera" and ii == 1:
                    labelText_6.set(labelText_6.get() + "\n" + name + "1 mean error: %.2f" %(tot_error/len(objpoints)))
                else:
                    labelText_6.set(labelText_6.get() + "\n" + name + " mean error: %.2f" %(tot_error/len(objpoints)))
                
        labelText_7.set("Calibration Saved!")
    
    if button_id == 10:
        cp = 1 - cp
        v_5.set("Switch to Camera %d" %(1-cp))
        image = cv2.resize(frame_without_dots[cp], (disp_w, disp_h))
        image = cv_to_tk(image)
        canvas_1.itemconfig(image_on_canvas_1, image = image)
        canvas_1.image = image
        
        image = cv2.resize(frame_with_dots[cp], (disp_w, disp_h))
        image = cv_to_tk(image)
        canvas_2.itemconfig(image_on_canvas_2, image = image)
        canvas_2.image = image
        
        if frame_warped[cp].any():
            image = cv2.resize(frame_warped[cp], (disp_w, disp_h))
            image = cv_to_tk(image)
            canvas_3.itemconfig(image_on_canvas_3, image = image)
            canvas_3.image = image
        else:
            image = cv_to_tk(im_with_keypoints[cp])
            canvas_3.itemconfig(image_on_canvas_3, image = image)
            canvas_3.image = image
                
btn = Button(master, text="Capture frame without dots", command = lambda i=0: callback(i))
btn.grid(row=4, column=2, columnspan=2, pady=5)

btn = Button(master, text="Capture frame with dots", command = lambda i=1: callback(i))
btn.grid(row=4, column=4, columnspan=2, pady=5)

if args.cam_num == 2:
    v_5 = StringVar()
    v_5.set("Switch to Camera %d" %(1-cp))
    btn = Button(master, textvariable=v_5, command = lambda i=10: callback(i))
    btn.grid(row=4, column=0, columnspan=2, pady=5)

v_0 = StringVar()
v_0.set("40")
e = Entry(master, textvariable=v_0)
e.grid(row=5, column=1, pady=5, sticky=E)

labelText_0 = StringVar()
labelText_0.set("Threshold")
label = Label(master, textvariable=labelText_0)
label.grid(row=5, column=0, pady=5, sticky=E)

v_1 = StringVar()
v_1.set("3")
e = Entry(master, textvariable=v_1)
e.grid(row=6, column=1, pady=5, sticky=E)

labelText_1 = StringVar()
labelText_1.set("min Area")
label = Label(master, textvariable=labelText_1)
label.grid(row=6, column=0, pady=5, sticky=E)

v_2 = StringVar()
v_2.set("False")
e = Entry(master, textvariable=v_2)
e.grid(row=7, column=1, pady=5, sticky=E)

labelText_2 = StringVar()
labelText_2.set("filter By Circularity")
label = Label(master, textvariable=labelText_2)
label.grid(row=7, column=0, pady=5, sticky=E)

labelText_dummy = StringVar()
labelText_dummy.set("")
label = Label(master, textvariable=labelText_dummy)
label.grid(row=9, column=0)

labelText_dummy = StringVar()
labelText_dummy.set("")
label = Label(master, textvariable=labelText_dummy)
label.grid(row=9, column=0)
label = Label(master, textvariable=labelText_dummy)
label.grid(row=14, column=0)

v_3 = StringVar()
v_3.set("")
e = Entry(master, textvariable=v_3)
e.grid(row=10, column=1, pady=5, sticky=E)

labelText_3 = StringVar()
labelText_3.set("col")
label = Label(master, textvariable=labelText_3)
label.grid(row=10, column=0, pady=5, sticky=E)

v_4 = StringVar()
v_4.set("")
e = Entry(master, textvariable=v_4)
e.grid(row=11, column=1, pady=5, sticky=E)

labelText_4 = StringVar()
labelText_4.set("row")
label = Label(master, textvariable=labelText_4)
label.grid(row=11, column=0, pady=5, sticky=E)


labelText_5 = StringVar()
labelText_5.set("")
label = Label(master, textvariable=labelText_5)
label.grid(row=13, column=3, columnspan = 2, pady=5)


labelText_6 = StringVar()
labelText_6.set("\n")
label = Label(master, textvariable=labelText_6)
label.grid(row=16, column=2, rowspan=1+args.cam_num, columnspan = 2, pady=5)

labelText_7 = StringVar()
labelText_7.set("")
label = Label(master, textvariable=labelText_7)
label.grid(row=17, column=4, columnspan = 2, pady=5)

btn = Button(master, text="Process images", command = lambda i=2: callback(i))
btn.grid(row=8, column=1, pady=5, sticky=E)

btn = Button(master, text="Order projected dots", command = lambda i=3: callback(i))
btn.grid(row=8, column=2, pady=5)

btn = Button(master, text="Manually order dots", command = lambda i=4: callback(i))
btn.grid(row=8, column=3, pady=5)

btn = Button(master, text="Warp image", command = lambda i=5: callback(i))
btn.grid(row=8, column=4, pady=5)

if args.load == False:
    txt = "Save Images"
else:
    txt = "Save / Next"
btn = Button(master, text=txt, font='Helvetica 12 bold',
             command = lambda i=6: callback(i))
btn.grid(row=15, column=5, pady=5, padx=5, sticky=E)

btn = Button(master, text="Find chessboard", command = lambda i=7: callback(i))
btn.grid(row=12, column=1, pady=5, sticky=E)

btn = Button(master, text="Add points", font='Helvetica 12 bold',
             command = lambda i=8: callback(i))
btn.grid(row=15, column=4, pady=5, sticky=E)

btn = Button(master, text="Calibrate System",font='Helvetica 16 bold',
             command = lambda i=9: callback(i))
btn.grid(row=16, column=0, rowspan=2, columnspan=2, pady=5, padx=5)

canvas_0 = Canvas(master, width = disp_w, height = disp_h)  
canvas_0.grid(row=0, column=0, columnspan=2, rowspan=3, padx=5, pady=5)
zframe = np.zeros((disp_h, disp_w))
image = cv_to_tk(zframe)
image_on_canvas_0 = canvas_0.create_image(0, 0, anchor=NW, image=image)  
canvas_0.image = image

canvas_1 = Canvas(master, width = disp_w, height = disp_h)  
canvas_1.grid(row=0, column=2, columnspan=2, rowspan=3, padx=5, pady = 5)  
zframe = np.zeros((disp_h, disp_w))
image = cv_to_tk(zframe)
image_on_canvas_1 = canvas_1.create_image(0, 0, anchor=NW, image=image)  
canvas_1.image = image

canvas_2 = Canvas(master, width = disp_w, height = disp_h)   
canvas_2.grid(row=0, column=4, columnspan=2, rowspan=3, padx=5, pady = 5)
zframe = np.zeros((disp_h, disp_w))
image = cv_to_tk(zframe)
image_on_canvas_2 = canvas_2.create_image(0, 0, anchor=NW, image=image)  
canvas_2.image = image

canvas_3 = Canvas(master, width = disp_w, height = disp_h)  
canvas_3.grid(row=5, column=2, columnspan=2, rowspan=3, padx=5, pady = 5)
zframe = np.zeros((disp_h, disp_w))
image = cv_to_tk(zframe)
image_on_canvas_3 = canvas_3.create_image(0, 0, anchor=NW, image=image)  
canvas_3.image = image


canvas_4 = Canvas(master, width = disp_w, height = disp_h)  
canvas_4.grid(row=10, column=2, columnspan=2, rowspan=3, padx=5, pady = 5)
zframe = np.zeros((disp_h, disp_w))
image = cv_to_tk(zframe)
image_on_canvas_4 = canvas_4.create_image(0, 0, anchor=NW, image=image)  
canvas_4.image = image


canvas_5 = Canvas(master, width = disp_w, height = disp_h)  
canvas_5.grid(row=10, column=4, columnspan=2, rowspan=3, padx=5, pady = 5)
zframe = np.zeros((disp_h, disp_w))
image = cv_to_tk(zframe)
image_on_canvas_5 = canvas_5.create_image(0, 0, anchor=NW, image=image)  
canvas_5.image = image



def task():
    global frame, frame_show
    if args.load == False:
        if args.cam_type == "usb":
            for i in range(args.cam_num):
                ret, frame[i] = cap[i].read()
                frame[i] = cv2.resize(frame[i], (args.cam_width, args.cam_height))
                frame[i] = cv2.cvtColor(frame[i], cv2.COLOR_BGR2GRAY)
                frame_show[i] = cv2.resize(frame[i], (disp_w, disp_h))
        elif args.cam_type == "flir":
            for i in range(args.cam_num):
                image_result = cam[i].GetNextImage()
                width = image_result.GetWidth()
                height = image_result.GetHeight()
                frame[i] = image_result.GetData().reshape(height, width)
                frame[i] = cv2.resize(frame[i], (args.cam_width, args.cam_height))
                frame_show[i] = cv2.resize(frame[i], (disp_w, disp_h))
                image_result.Release()
        image = cv_to_tk(frame_show[cp])
        canvas_0.itemconfig(image_on_canvas_0, image = image)
        canvas_0.image = image
    master.after(1, task)

master.after(1, task)
master.mainloop()