
import numpy as np
import cv2
import argparse
from glob import glob

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, default= "./apsi_files/",
                    help="path to load automatic psi files")
parser.add_argument("--vid_name", type=str, default= "michal.mp4",
                    help="name of video file to load")
parser.add_argument("--src_idx", type=int, default= 0,
                    help="ID of origin point cloud to which poses are estimated")
parser.add_argument("--cam_num", type=int, default= 1, choices = [1,2],
                    help="number of cameras")
parser.add_argument("--thr", type=float, default= 0.9,
                    help="corr_coef threshold to remove spurious poses")
parser.add_argument("--res_max", type=float, default= 0.05,
                    help="residual threshold to remove spurious poses")
parser.add_argument("--cam_height", type=int, default= 512,
                    help="height of camera image")
parser.add_argument("--cam_width", type=int, default= 640,
                    help="width of camera image")
args = parser.parse_args()

pts_path = args.load_path + args.vid_name.split('.')[0] + "/"
dump_path = pts_path + "estimated_poses/"

cam_path = [None]*args.cam_num
for i in range(args.cam_num):
    cam_path[i] = pts_path + "camera_%d/" %i

times_save = np.load(cam_path[0] + 'times.npy')
f_init = int((times_save[1,0] - times_save[0,0])/times_save[0,0] - 1)
vid_rate = int(1/(times_save[1,0] - times_save[0,0])/4)

corr_coef = [None]*args.cam_num
for i in range(args.cam_num):
    corr_coef[i] = np.load(cam_path[i] + 'corr_coef.npy', allow_pickle=True).reshape(1,-1)
corr_coef = np.concatenate(corr_coef)
corr_coef = np.min(corr_coef, axis = 0)
retval = np.load(dump_path+'retval.npy')
residual = np.load(dump_path+'residual.npy')

c_x = args.cam_width / 2
c_y = args.cam_height / 2
f_x = c_x / np.tan(60/2 * np.pi / 180)
f_y = f_x
p_start = (int(c_x), int(c_y))
p_stop = [(int(c_x), int(c_y)) for _ in range(3)]

camera_matrix = np.float32([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])

axis = np.float32([[0.1, 0.0, 0.0], 
                   [0.0, 0.1, 0.0], 
                   [0.0, 0.0, 10]])
ax_colors = [(255,0,0), (0,255,0), (0,0,255)]

tvec = np.array([0.0, 0.0, 1.0], np.float) # translation vector



video = cv2.VideoWriter(dump_path + args.vid_name,
                               cv2.VideoWriter_fourcc(*'mp4v'),
                               vid_rate,
                               (args.cam_width,args.cam_height))

cap = cv2.VideoCapture(args.load_path + args.vid_name[:-4] + "_0.mp4")

for i in range(f_init):
    ret, frame = cap.read()
    #video.write(frame)

N = len(glob(dump_path + "pose_*.npy"))
    
for dst_idx in range(N):
    if corr_coef[dst_idx]>=args.thr and retval[dst_idx] == 1 and residual[dst_idx]<args.res_max:
        pose_est = np.load(dump_path + "pose_%d_to_%d.npy" %(args.src_idx, dst_idx))
        pose_est = pose_est[:-1,:-1]
    
        rvec, _ = cv2.Rodrigues(pose_est)
        imgpts, _ = cv2.projectPoints(axis, rvec, tvec, camera_matrix, distCoeffs=None)
        for i in range(3):
            p_stop[i] = (int(imgpts[i][0][0]), int(imgpts[i][0][1]))
    for j in range(4):
        ret, frame = cap.read()
        if j == 0:
            for i in range(3):
                frame = cv2.line(frame, p_start, p_stop[i], ax_colors[i], 3)
            video.write(frame)

cv2.destroyAllWindows()
video.release()
