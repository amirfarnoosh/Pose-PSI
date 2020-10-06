
import argparse
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--load_path", type=str, default= "./apsi_files/",
                    help="path to load automatic psi files")
parser.add_argument("--vid_name", type=str, default= "face_mnq1.mp4",
                    help="name of video file to load")
parser.add_argument("--thr", type=float, default= 0.9,
                    help="corr_coef threshold to remove spurious poses")
parser.add_argument("--res_max", type=float, default= 0.05,
                    help="residual threshold to remove spurious poses")

args = parser.parse_args()

pts_path = args.load_path + args.vid_name.split('.')[0] + "/"
dump_path = pts_path + "estimated_poses/"

poses = np.load(dump_path+'poses.npy').T
retval = np.load(dump_path+'retval.npy')
residual = np.load(dump_path+'residual.npy')
corr_coef = poses[-1]
psi, theta, phi, tx, ty, tz, time, corr_coef = poses[:, (corr_coef>=args.thr
                                                         and retval == 1
                                                         and residual<args.res_max)]

plt.figure()

psi = psi*180/np.pi
plt.subplot(2,3,1)
plt.bar(time, psi)
plt.title("psi")

theta = theta*180/np.pi
plt.subplot(2,3,2)
plt.bar(time, theta)
plt.title("theta")

phi = phi*180/np.pi
plt.subplot(2,3,3)
plt.bar(time, phi)
plt.title("phi")

plt.subplot(2,3,4)
plt.bar(time, tx)
plt.title("tx")

plt.subplot(2,3,5)
plt.bar(time, ty)
plt.title("ty")

plt.subplot(2,3,6)
plt.bar(time, tz)
plt.title("tz")

plt.show()
