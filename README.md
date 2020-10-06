# Pose Estimation with Phase Shift Interferometry (PSI)

## Contents
* [calibration](#calibration)   
* [bbb_files](#bbb_files)
* [capture](#capture)
* [psi](#psi)
* [pose_estimation](#pose_estimation)


To check what (additional) arguments each python file takes, use `python file_name.py -h`.

## calibration

This folder includes files needed for camera(s)/projection calibration.

* To generate a gray-white chessboard with 8 rows and 10 columns and square size of 7 millimeters:

`python ./calibration/chessboard_generator.py --row 8 --col 10 --size 7 --path "./calibration_chessboard/"`

* To generate a grid of dots with specified height and width (in dots) , and offsets (in pixels):

`python ./calibration/dots_black_generator.py --grid_height 6 --grid_width 7 --grid_height_offset 75 --grid_width_offset 150 --path "./dots_black_images/"

* The following script gets information of chessboard and path to dots image, and will run a GUI to calibrate two FLIR cameras with projector:

`python ./calibration/camera_projector_calibration_gui.py --row 8 --col 10 --size 7 --path_dots "./dots_black_images/dots.png" --cam_height 512 --cam_width 640 --cam_num 2 --cam_type "flir" --path "./chessboard_calibration_images/"`

* To run a GUI for color calibration for rainbow patterns:

`python ./calibration/color_calibration.py


## bbb_files

This folder includes files for BeagleBone Black (BBB) that handle pattern generation and presentation.

* The following scripts generates gray scale fringe images with phase pitch of 100 pixels:

`python ./bbb_files/fringe_generation.py --type "gray" --pitch 100 --path "./fringe_images/"`

`Python ./bbb_files/fringe_generation.py --type "centerline" --path "./fringe_images/"

* Transfer generated fringe folder and `sync_presentation.py` to BBB and run:

`python ./sync_presentation.py --freq 10 --intensity 100 --load_path "./fringe_images/" --files "Gray_pattern_a127_b0_p100_0.png Gray_pattern_a127_b0_p100_1.png Gray_pattern_a127_b0_p100_2.png vcenterline_pattern.png"

to show patterns with frequency of 10 Hz and intensity of 100.

## capture 

The following script will record video from two FLIR cameras for 600 frames. The name of saved video files are appended with camera number:

`python ./capture/video_recorder.py --cam_height 512 --cam_width 640 --cam_num 2 --cam_type "flir" --dump_path "./apsi_files/" --vid_name test.mp4 --num 600 

On the other hand `video_recorder.py` saves video file from images included in a specified path.

## psi

This folder includes files needed for phase shift interferometry and point cloud generation/visualization. 

* The following script gets camera information, path to calibration files, pattern presentation rate (here 10), and whether camera/projector are synced, and runs a GUI for phase shift interferometry, and dumps files into the specified folders:

`python ./psi/PSI_gui.py --cam_height 512 --cam_width 640 --calib_path "./chessboard_calibration_images/" --apsi_path "./apsi_files/" --psi_path "./psi_files/" --vid_rate 10 --sync --cam_num 2 --cam_type "flir"

* The following script will load videos `test_0.mp4` and `test_1.mp4` (from two cameras) from the specified folder (`./apsi_files/` here) and runs GUI for phase shift interferometry:

`python ./psi/PSI_gui.py --cam_height 512 --cam_width 640 --calib_path "./chessboard_calibration_images/" --apsi_path "./apsi_files/" --psi_path "./psi_files/" --sync --cam_num 2 --load --vid_rate 2 --vid_name "test.mp4"


## pose_estimation 

This folder include files for pose estimation.

* The following script will apply phase shift interferometry to “test.mp4” video file and generates point clouds:

`python ./pose_estimation/PSI_pose_estimation_auto.py --psi --cam_num 2 --load_path "./apsi_files/" --vid_name "test.mp4" --calib_path "./chessboard_calibration_images/" --cam_rate 10 --sync --pitch 100 --s 5 --thr_cntr 0.5 --mask_thr 2 --max_save 500

* This script will load point clouds related to the video file and process them to extract poses with respect to a reference specified by `orig_id`.

`python ./pose_estimation/PSI_pose_estimation_auto.py --pose --load_path "./apsi_files/" --vid_name "test.mp4" --orig_id 0 --cam_num 2

* To plot pose results:

`python ./pose_estimation/pose_plot.py --load_path "./apsi_files" --vid_name "test.mp4" 

* To generate a video overlaid with pose results:

`python ./pose_estimation/video_with_pose.py --load_path "./apsi_files" --vid_name "test.mp4" --src_idx 0 --cam_num 2 
