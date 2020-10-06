#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 12:12:12 2019

@author: amirreza
"""

import os
import subprocess
import Adafruit_BBIO.GPIO as GPIO
import time
from time import sleep
import argparse
import pygame
import numpy as np

# run "export DISPLAY=:0" in command line
# set exposure to 6700us, and trigger to rising edge
# or exposure Auto and falling edge and freq 12
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--freq", type=float, default= 15,
                    help="frequency of pin setup")
parser.add_argument("-i", "--intensity", type=int, default= 100,
                    help="projection intensity")
parser.add_argument("--cam_num", type=int, default= 1, choices = [1,2],
                    help="number of cameras")
parser.add_argument("--num", type=int, default= 6000,
                    help="number of time points to save")
parser.add_argument("--load_path", type=str, default= "./fringe_images/",
                    help="path to load fringe images from")
parser.add_argument("--files", type=str, 
                    default= "Gray_pattern_a127_b0_p100_0.png \
                    Gray_pattern_a127_b0_p100_1.png \
                    Gray_pattern_a127_b0_p100_2.png \
                    vcenterline_pattern.png",
                    help="image file names to load")
args = parser.parse_args()

### setup display
amp = args.intensity
os.chdir("/opt/scripts/device/bone/capes/DLPDLCR2000")
subprocess.call("i2cset -y 2 0x1b 0x0b 0x00 0x00 0x00 0x00 i".split())  
subprocess.call("i2cset -y 2 0x1b 0x0c 0x00 0x00 0x00 0x1b i".split())  
subprocess.call("export DISPLAY=:0", shell=True, executable='/bin/bash')
#line = "export DISPLAY=:0"
#new_env = dict(os.environ)
#new_env['DISPLAY'] = '0'
#subprocess.Popen(line, env=new_env, shell=True)
subprocess.call(("python LEDSet.py %d %d %d" %(amp, amp, amp)).split())
###


file_names = args.files.split()
file_paths = ""
for file_name in file_names:
    file_paths += args.load_path + file_name + " "
file_paths = file_paths.split()

### set trigger
delay = 1/args.freq
pre_delay = delay/100
post_delay = delay - pre_delay
GPIO.setup("P9_23", GPIO.OUT)
if args.cam_num == 2:
    GPIO.setup("P9_25", GPIO.OUT)
###


pygame.init()
pygame.mouse.set_visible(False)
screen = DISPLAYSURF = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
black = (0,0,0)

images = [pygame.image.load(file_paths[i]) for i in range(len(file_paths))]

times_func1 = []
def func1():
    strt = time.time()
    while True:
        for i in range(len(images)):
            
            #strt = time.time()
            GPIO.output("P9_23", GPIO.LOW)
            if args.cam_num == 2:
                GPIO.output("P9_25", GPIO.LOW)
            sleep(pre_delay)
            #screen.fill(black) # black
            screen.blit(images[i],(0, 0))
            pygame.display.flip()
            GPIO.output("P9_23", GPIO.HIGH)
            if args.cam_num == 2:
                GPIO.output("P9_25", GPIO.HIGH)
            times_func1.append(time.time() - strt)
            sleep(post_delay)
            #stp = time.time()
            #print((stp-strt))
            if len(times_func1) == args.num:
                np.save("times_proj", times_func1) 
                break
                
            

GPIO.setup("P8_14", GPIO.IN)
#GPIO.add_event_detect("P8_14", GPIO.RISING)

times_func2 = []     
def func2():
    strt = time.time()
    while True:
        GPIO.wait_for_edge("P8_14", GPIO.RISING)
        times_func2.append(time.time() - strt)
        
        if len(times_func2) == args.num:
            np.save("times_mri", times_func2) 
            break
             


from multiprocessing import Process
def runInParallel(*fns):
  proc = []
  for fn in fns:
    p = Process(target=fn)
    p.start()
    proc.append(p)
  for p in proc:
    p.join()

runInParallel(func1, func2)