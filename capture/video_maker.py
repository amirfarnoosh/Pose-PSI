
"""
Write video file from collected images and save it.
 
 --vid_rate: can be ignored as it only controls representation rate of video
"""

import cv2
import os
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--vid_rate", type=float, default=0,
                    help="video frame rate")
parser.add_argument("--load_path", type=str, default= "./psi_images/",
                    help="path to load psi images from")
parser.add_argument("--vid_name", type=str, default= "video.mp4",
                    help="name of video file with extension")
parser.add_argument("--dump_path", type=str, default= "./apsi_files/",
                    help="path to dump psi video")
parser.add_argument("--ext", type=str, default= "png", 
                    choices = ["png", "Bmp"],
                    help="extension of image files")
parser.add_argument("--cam_height", type=int, default= 512,
                    help="height of camera image")
parser.add_argument("--cam_width", type=int, default= 640,
                    help="width of camera image")

args = parser.parse_args()

image_folder = args.load_path
save_path = args.dump_path

if not os.path.exists(save_path):
    os.mkdir(save_path)


from PIL import Image

def bmp_to_png(file_in):
    """ changes format of Bmp image to png and save it.
    Args:
        file_in (string): A string containing the name of
            bitmap file (without extension).
    Returns:
        None: saves png format in the same path as original file
    """
    img = Image.open(file_in + "." + args.ext)
    file_out = file_in + ".png"
    img.save(file_out)
    

video_name = save_path + args.vid_name

files_list = glob(image_folder + "*." + args.ext)
files_list.sort()

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), args.vid_rate, (args.cam_width,args.cam_height))

for file in files_list:
    
    if args.ext != "png":
        bmp_to_png(file[:-4])
    
    frame = cv2.imread(file[:-4] + ".png")
    frame = cv2.resize(frame, (args.cam_width, args.cam_height))
    video.write(frame)

cv2.destroyAllWindows()
video.release()