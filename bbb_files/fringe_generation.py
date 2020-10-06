
"""
Generating fringe patterns for PSI
"""
import numpy as np
import cv2
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--type", type=str, default = "gray",
                    choices = ["gray", "egray", "color", "color_calib", "centerline"], 
                    help="type of fringe to generate")
parser.add_argument("-p", "--pitch", type=int, default=40,
                    help="pitch of fringe in pixels")
parser.add_argument("--vmin", type=int, default= 0,
                    help="minimum intensity value")
parser.add_argument("--vmax", type=int, default= 255,
                    help="maximum intensity value")
parser.add_argument("--proj_height", type=int, default= 360,
                    help="height of projector image")
parser.add_argument("--proj_width", type=int, default= 640,
                    help="width of projector image")
parser.add_argument("--path", type=str, default= "./fringe_images/",
                    help="path to dump fringes")
parser.add_argument("--sw", type=int, default= 20,
                    help="width of horizontal stripes for color-encoded gray fringes")
parser.add_argument("--ext", type=str, default="png", choices = ["png", "bmp"],
                    help="extension of fringe files")

args = parser.parse_args()

# save Path
save_path = args.path
if not os.path.exists(save_path):
    os.mkdir(save_path)

h, w = args.proj_height, args.proj_width  #height, and width of projector image

# fringe generation
u_p = np.tile(np.arange(0, w), h).reshape(h, w)

vmin = args.vmin
vmax = args.vmax
a = (vmax - vmin)//2 # amplitude
b = vmin # bias 
p = args.pitch # pitch in pixels
s_k = (2 * np.pi / 3, 0, -2 * np.pi / 3)

I_B = a * (1 + np.cos(2 * np.pi / p * u_p + s_k[0])) + b
I_G = a * (1 + np.cos(2 * np.pi / p * u_p + s_k[1])) + b
I_R = a * (1 + np.cos(2 * np.pi / p * u_p + s_k[2])) + b

I_B = np.expand_dims(np.uint8(np.round(I_B)), 2)
I_G = np.expand_dims(np.uint8(np.round(I_G)), 2)
I_R = np.expand_dims(np.uint8(np.round(I_R)), 2)

if args.type == "color": 
    I = np.concatenate((I_B, I_G, I_R), axis = 2)
    cv2.imwrite(save_path + 'fringe_pattern_a%d_b%d_p%d.' %(a,b,p) + args.ext, I)

if args.type == "color_calib":
    I_z = np.uint8(np.zeros((h,w,1)))
    
    I_R_0 = np.concatenate((I_z, I_z, I_B), axis = 2)
    I_R_1 = np.concatenate((I_z, I_z, I_G), axis = 2)
    I_R_2 = np.concatenate((I_z, I_z, I_R), axis = 2)
    cv2.imwrite(save_path + 'Red_pattern_a%d_b%d_p%d_0.' %(a,b,p) + args.ext, I_R_0)
    cv2.imwrite(save_path + 'Red_pattern_a%d_b%d_p%d_1.' %(a,b,p) + args.ext, I_R_1)
    cv2.imwrite(save_path + 'Red_pattern_a%d_b%d_p%d_2.' %(a,b,p) + args.ext, I_R_2)
    
    I_G_0 = np.concatenate((I_z, I_B, I_z), axis = 2)
    I_G_1 = np.concatenate((I_z, I_G, I_z), axis = 2)
    I_G_2 = np.concatenate((I_z, I_R, I_z), axis = 2)
    cv2.imwrite(save_path + 'Green_pattern_a%d_b%d_p%d_0.' %(a,b,p) + args.ext, I_G_0)
    cv2.imwrite(save_path + 'Green_pattern_a%d_b%d_p%d_1.' %(a,b,p) + args.ext, I_G_1)
    cv2.imwrite(save_path + 'Green_pattern_a%d_b%d_p%d_2.' %(a,b,p) + args.ext, I_G_2)
    
    I_B_0 = np.concatenate((I_B, I_z, I_z), axis = 2)
    I_B_1 = np.concatenate((I_G, I_z, I_z), axis = 2)
    I_B_2 = np.concatenate((I_R, I_z, I_z), axis = 2)
    cv2.imwrite(save_path + 'Blue_pattern_a%d_b%d_p%d_0.' %(a,b,p) + args.ext, I_B_0)
    cv2.imwrite(save_path + 'Blue_pattern_a%d_b%d_p%d_1.' %(a,b,p) + args.ext, I_B_1)
    cv2.imwrite(save_path + 'Blue_pattern_a%d_b%d_p%d_2.' %(a,b,p) + args.ext, I_B_2)

# centerlines patterns
if args.type == "centerline":
    I_c = np.zeros((h, w, 1))
    I_c[:, w//2] = 255
    I_c[h//2, :] = 255
    I_c = np.uint8(I_c) 
    
    I_vc = np.zeros((h, w, 1))
    I_vc[:, w//2] = 255
    I_vc = np.uint8(I_vc) 
    
    cv2.imwrite(save_path + 'centerline_pattern.' + args.ext, np.tile(I_c, (1,1,3)))
    cv2.imwrite(save_path + 'vcenterline_pattern.' + args.ext, np.tile(I_vc, (1,1,3)))

# gray scale patterns
if args.type == "gray":
    cv2.imwrite(save_path + 'Gray_pattern_a%d_b%d_p%d_0.' %(a,b,p) + args.ext, np.tile(I_B, (1,1,3)))
    cv2.imwrite(save_path + 'Gray_pattern_a%d_b%d_p%d_1.' %(a,b,p) + args.ext, np.tile(I_G, (1,1,3)))
    cv2.imwrite(save_path + 'Gray_pattern_a%d_b%d_p%d_2.' %(a,b,p) + args.ext, np.tile(I_R, (1,1,3)))

if args.type == "egray":
    lw = args.sw
    
    I_B = np.tile(I_B, (1,1,3))
    I_B[0:lw,:,0] = 255
    I_B[0:lw,:,1:] = 0
    I_B[-lw:,:,0] = 255
    I_B[-lw:,:,1:] = 0
    
    I_G = np.tile(I_G, (1,1,3))
    I_G[0:lw,:,1] = 255
    I_G[0:lw,:,[0,2]] = 0
    I_G[-lw:,:,1] = 255
    I_G[-lw:,:,[0,2]] = 0
    
    
    I_R = np.tile(I_R, (1,1,3))
    I_R[0:lw,:,2] = 255
    I_R[0:lw,:,:-1] = 0
    I_R[-lw:,:,2] = 255
    I_R[-lw:,:,:-1] = 0

    cv2.imwrite(save_path + 'eGray_pattern_a%d_b%d_p%d_0.' %(a,b,p) + args.ext, I_B)
    cv2.imwrite(save_path + 'eGray_pattern_a%d_b%d_p%d_1.' %(a,b,p) + args.ext, I_G)
    cv2.imwrite(save_path + 'eGray_pattern_a%d_b%d_p%d_2.' %(a,b,p) + args.ext, I_R)
