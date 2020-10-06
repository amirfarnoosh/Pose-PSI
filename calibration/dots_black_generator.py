#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 14:47:42 2019

@author: amirreza
"""

import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--proj_height", type=int, default= 360,
                    help="height of projector image")
parser.add_argument("--proj_width", type=int, default= 640,
                    help="width of projector image")
parser.add_argument("--grid_height", type=int, default= 6,
                    help="height of dotted grid in number of dots")
parser.add_argument("--grid_width", type=int, default= 7,
                    help="width of dotted grid in number of dots")
parser.add_argument("--grid_height_offset", type=int, default= 75,
                    help="height offset of dotted grid in pixels")
parser.add_argument("--grid_width_offset", type=int, default= 150,
                    help="width offset of dotted grid in pixels")
parser.add_argument("--path", type=str, default= "./dots_black_images/",
                    help="path to save dotted grid and black images")

args = parser.parse_args()

save_path = args.path 
if not os.path.exists(save_path):
    os.mkdir(save_path)

h, w = args.proj_height, args.proj_width
off_h, off_w = args.grid_height_offset, args.grid_width_offset
grid_size = (args.grid_height, args.grid_width)

h_delta = int(np.round((h - 2.0 * off_h)/(grid_size[0] + 1)))
w_delta = int(np.round((w - 2.0 * off_w)/(grid_size[1] + 1)))

idx = np.mgrid[off_h+h_delta:off_h+h_delta+grid_size[0]*h_delta:h_delta,
               off_w+w_delta:off_w+w_delta+grid_size[1]*w_delta:w_delta].T.reshape(-1,2)

img = np.zeros((h, w))
cv2.imwrite(save_path + 'black.png', img)
img[idx[:,0], idx[:,1]] = 255
img = np.uint8(img)
cv2.imwrite(save_path + 'dots.png', img)