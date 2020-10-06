#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 11:52:07 2019

@author: amirreza
"""
import numpy as np
import cv2
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--row", type=int, default = 8,
                    help="number of rows in chessboard")
parser.add_argument("--col", type=int, default = 10,
                    help="number of columns in chessboard")
parser.add_argument("-c", "--color", type=int, default= 120,
                    help="gray intensity of chessboard squares")
parser.add_argument("-s", "--size", type=float, default= 7,
                    help="size of chessboard squares in millimeter")
parser.add_argument("--dpi", type=int, default= 72,
                    help="printer dots per inch (dpi)")
parser.add_argument("--path", type=str, default= "./calibration_chessboard/",
                    help="path to save chessboard")

args = parser.parse_args()

clr = args.color
ncol = args.col
nrow = args.row
DPI = args.dpi
mm = args.size
if not os.path.exists(args.path):
    os.mkdir(args.path)
    
path = args.path + "calibration_chessboard_%dx%d_%dmm_%ddpi.png" %(nrow, ncol, mm, DPI)

scale_row = int(mm * nrow * DPI / 25.4)
scale_col = int(mm * ncol * DPI / 25.4)

img  = np.ones((nrow,ncol))*255
img[1::2,1::2] = clr
img[0::2,0::2] = clr

img = np.uint8(img)
img = cv2.resize(img, (scale_col, scale_row), interpolation = 0)

cv2.imwrite(path, img)