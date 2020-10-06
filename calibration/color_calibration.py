#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 17:30:41 2019

@author: amirreza
"""

import numpy as np
import cv2
import argparse
import os


""" Press "c" on frame window to capture image """

parser = argparse.ArgumentParser()
parser.add_argument("--cam_height", type=int, default= 360,
                    help="height of camera image")
parser.add_argument("--cam_width", type=int, default= 640,
                    help="width of camera image")
parser.add_argument("--path", type=str, default= "./color_calibration_files/",
                    help="path to save color calibration matrices")

args = parser.parse_args()


import sys
if sys.version_info[0] == 3:
    # for Python3
    from tkinter import *   ## notice lowercase 't' in tkinter here
else:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter
    
from PIL import Image
from PIL import ImageTk 

def cv_to_tk(frame_input):
    image = Image.fromarray(frame_input)
    image = ImageTk.PhotoImage(image)
    return image

cap = cv2.VideoCapture(1)
master = Tk()
master.title("Color Calibration")

save_path = args.path

if not os.path.exists(save_path):
    os.mkdir(save_path)

color_list = ['Blue', 'Green', 'Red']
# Capture frame-by-frame
cnt = 0
c_cnt = 0 #color counter
frames = []
frames_show = []
coupling = np.zeros((3,3))
K = np.zeros((3,3))
frame, frame_show = None, None

def callback():
    global frames, frames_show, cnt, c_cnt, K, coupling
    frames.append(frame)
    cnt += 1
    image = cv_to_tk(frame_show)
    canvas_1.itemconfig(image_on_canvas_1, image = image)
    canvas_1.image = image
    
    if cnt == 3:
        I_av = (frames[0].astype('float') + frames[1].astype('float') + frames[2].astype('float'))/3
        I_av = I_av[args.cam_height//4:3 * args.cam_height//4,
                    args.cam_width//4:3 * args.cam_width//4].sum(axis = 2).mean()
        K[c_cnt, c_cnt] = I_av
        for i in range(3):        
            I_1 = frames[0][:,:,i].astype('float')
            I_2 = frames[1][:,:,i].astype('float')
            I_3 = frames[2][:,:,i].astype('float')
            
            I_mod = np.sqrt(3*(I_1 - I_3)**2 + (2*I_2-I_1-I_3)**2) / 3
            I_mod = I_mod[args.cam_height//4:3 * args.cam_height//4,
                          args.cam_width//4:3 * args.cam_width//4].mean()
            coupling[c_cnt,i] = I_mod
        
        coupling[c_cnt] = coupling[c_cnt] / coupling[c_cnt, c_cnt]     
        
        frames_show.append(cv2.resize(frames[1], (args.cam_width//2, args.cam_height//2)))
        
        frames = []
        cnt = 0
        c_cnt += 1
        
        if c_cnt == 3:
            K = K / K.max()
            
            print('Coupling matrix obtained: (B G R):')
            print(coupling.T)
            np.save(save_path + 'coupling_mat.npy', coupling.T)
            np.save(save_path + 'K_mat.npy', K)
            c_cnt = 0
            btn_text.set("Capture %s Phase #%d" %(color_list[c_cnt], cnt))
            show_results()
            
    btn_text.set("Capture %s Phase #%d" %(color_list[c_cnt], cnt))
            
def show_results():
    global frames_show
    
    slave = Toplevel(master)
    slave.title("Color Calibration Results")    
    
    for i in range(3):
        
        labels = Label(slave, text=color_list[i])
        labels.grid(row=2*i+1, column=0, padx=5, sticky=E)
        labels = Label(slave, text=color_list[i])
        labels.grid(row=0, column=i+1, pady=5)
        for j in range(3):
            labels = Label(slave, text="%.2f" %coupling[i,j])
            labels.grid(row=i*2+2, column=j+1, pady=5)
            rcanvas = Canvas(slave, width = args.cam_width//2, height = args.cam_height//2)  
            rcanvas.grid(row=i*2+1, column=j+1, padx=5, pady=5)
            image = cv_to_tk(frames_show[i][:,:,j])
            rcanvas.create_image(0, 0, anchor=NW, image=image)  
            rcanvas.image = image
    
    frames_show = []
    slave.mainloop()
    
             
canvas_0 = Canvas(master, width = args.cam_width//2, height = args.cam_height//2)  
canvas_0.grid(row=0, column=0, padx=5, pady=5)
frame = np.zeros((args.cam_height//2, args.cam_width//2))
image = cv_to_tk(frame)
image_on_canvas_0 = canvas_0.create_image(0, 0, anchor=NW, image=image)  
canvas_0.image = image

canvas_1 = Canvas(master, width = args.cam_width//2, height = args.cam_height//2)  
canvas_1.grid(row=0, column=1, padx=5, pady=5)  
frame = np.zeros((args.cam_height//2, args.cam_width//2))
image = cv_to_tk(frame)
image_on_canvas_1 = canvas_1.create_image(0, 0, anchor=NW, image=image)  
canvas_1.image = image

btn_text = StringVar()
btn_text.set("Capture %s Phase #%d" %(color_list[c_cnt], cnt))
btn = Button(master, textvariable=btn_text, command = callback)
btn.grid(row=1, column=1, pady=5)

def task():
    global frame, frame_show
    ret, frame = cap.read()
    frame = cv2.resize(frame, (args.cam_width, args.cam_height))
    frame_show = cv2.resize(frame, (args.cam_width//2, args.cam_height//2))
    image = cv_to_tk(frame_show)
    canvas_0.itemconfig(image_on_canvas_0, image = image)
    canvas_0.image = image
    master.after(1, task)

master.after(1, task)
master.mainloop()


    
        
"""

[[1.         0.10258808 0.07679351]
 [0.4879571  1.         0.2996711 ]
 [0.12603854 0.49929177 1.        ]]

K = array([[0.50173509, 0.        , 0.        ],
       [0.        , 1.        , 0.        ],
       [0.        , 0.        , 0.74000631]])

"""
