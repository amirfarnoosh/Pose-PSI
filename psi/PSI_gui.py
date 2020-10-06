
import numpy as np
import cv2
import os
import argparse
from unwrap_2d import *
from select_largest_obj import *
import sys
from esync_func import *

parser = argparse.ArgumentParser()
parser.add_argument("--cam_height", type=int, default= 360,
                    help="height of camera image")
parser.add_argument("--cam_width", type=int, default= 640,
                    help="width of camera image")
parser.add_argument("--proj_height", type=int, default= 360,
                    help="height of projector image")
parser.add_argument("--proj_width", type=int, default= 640,
                    help="width of projector image")
parser.add_argument("--apsi_path", type=str, default= "./apsi_files/",
                    help="path to dump automatic psi files")
parser.add_argument("--psi_path", type=str, default= "./psi_files/",
                    help="path to dump psi files")
parser.add_argument("--calib_path", type=str, default= "./chessboard_calibration_images/",
                    help="top-level path to calibration files")
parser.add_argument("--load", action="store_true",
                    help="whether to load from a file or not")
parser.add_argument("--vid_name", type=str, default= "video_0.mp4",
                    help="name of video file to load")
parser.add_argument("--vid_rate", type=float, default= 30,
                    help="frame rate to show video file")
parser.add_argument("--sync", action="store_true",
                    help="whether camera/projector are synced")
parser.add_argument("--rgb", action="store_true",
                    help="whether we are doing rainbow psi")
parser.add_argument("--cam_num", type=int, default= 1, choices = [1,2],
                    help="number of cameras")
parser.add_argument("--cam_type", type=str, default= "flir", choices = ["usb","flir"],
                    help="type of cameras")


args = parser.parse_args()

if not os.path.exists(args.psi_path):
    os.mkdir(args.psi_path)
    
if not os.path.exists(args.apsi_path):
    os.mkdir(args.apsi_path)


from point_cloud_generator import *

if args.sync:    
    from PSI_automatic_sync import *
elif not args.rgb:
    from PSI_automatic_esync import *
if args.rgb:
    from PSI_automatic_rgb import *
    
from plot_surface_plotly import *

if sys.version_info[0] == 3:
    # for Python3
    from tkinter import *   ## notice lowercase 't' in tkinter here
else:
    # for Python2
    from Tkinter import *   ## notice capitalized T in Tkinter
    
from PIL import Image
from PIL import ImageTk 


frame = [None]*args.cam_num
image = [None]*args.cam_num
num_task = 4
flag = [False]*num_task
slave = [None]*num_task

rt_cam, rt_proj = cam_proj_calib_mat_generator(args.calib_path,
                                               img_num = 0,
                                               cam_num = args.cam_num)

def main_task():
    global frame, image, loop_cnt
    loop_cnt += 1
    if loop_cnt == loop_max:
        loop_cnt = 0
        if args.cam_type == "usb" or args.load == True:
            for i in range(args.cam_num):
                ret, frame[i] = cap[i].read()
                frame[i] = cv2.resize(frame[i], (args.cam_width, args.cam_height))
                image[i] = cv2.resize(frame[i], (disp_w, disp_h))
                image[i] = cv_to_tk(image[i])
        elif args.cam_type == "flir":
            for i in range(args.cam_num):
                image_result = cam[i].GetNextImage()
                width = image_result.GetWidth()
                height = image_result.GetHeight()
                if args.rgb:
                    frame[i] = image_result.GetData().reshape(height, width, 3)
                else:
                    frame[i] = image_result.GetData().reshape(height, width)
                frame[i] = cv2.resize(frame[i], (args.cam_width, args.cam_height))
                if not args.rgb:
                    frame[i] = cv2.cvtColor(frame[i], cv2.COLOR_GRAY2BGR)
                image[i] = cv2.resize(frame[i], (disp_w, disp_h))
                image[i] = cv_to_tk(image[i])
                image_result.Release()
                
    master.after(1,lambda i=0: tasks(i))

def cv_to_tk(frame_input):
    image = Image.fromarray(frame_input)
    image = ImageTk.PhotoImage(image)
    return image

def set_flag(j):
    global flag
    slave[j].destroy()
    flag[j] = False

def compute_phase(eventorigin):
    global phi_unwrapped, I_phi_unwrapped, image_phi, gamma, Is_psi
    
    if v_2[2].get() == "":
        choices = [cp]
    else:
        choices = np.arange(args.cam_num)
    for i in choices:
        I_1 = Is_psi[i][0].astype('float')
        I_2 = Is_psi[i][1].astype('float')
        I_3 = Is_psi[i][2].astype('float')
            
        #  mask index generation
        
        I_avg = (I_1 + I_2 + I_3) / 3
        I_mod = np.sqrt(3*(I_1 - I_3)**2 + (2*I_2-I_1-I_3)**2) / 3
        gamma[i] = I_mod / (I_avg + 1)
        mask = np.ones(frame[i].shape[0:2])
        mask_idxs = np.where(mask == 1)
        ###
            
        phi = np.arctan2(np.sqrt(3) * (I_1 - I_3) , (2 * I_2 - I_1 - I_3))
        phi_unwrapped[i] = unwrap_2d(phi, mask_idxs = mask_idxs, fig_flag = False)
        
        if v_2[2].get() == "":
            x0 = int(eventorigin.x * ratio)
            y0 = int(eventorigin.y * ratio)
            refPt = (y0, x0)
            image = cv2.resize(Is_psi[i][3], (disp_w, disp_h))
            cv2.circle(image,
                       center = (x0, y0),
                       radius = 3,
                       color = (0, 255, 0),
                       thickness = 1,
                       lineType = 2,
                       shift = 0)
            image = cv_to_tk(image)
            canvas_2[4].itemconfig(image_on_canvas_2[3], image = image)
            canvas_2[4].image = image
            canvas_2[4].unbind("<Button 1>")
            
            
            phi_0 = 2 * np.pi / float(v_2[0].get()) * args.proj_width/2
            off_set = phi_0 - phi_unwrapped[i][refPt[0], refPt[1]]
            phi_unwrapped[i] = phi_unwrapped[i] + off_set
        
        I_phi_unwrapped[i] = np.uint8(exposure.rescale_intensity(phi_unwrapped[i], out_range=(0, 255)))
        I_phi_unwrapped[i] = cv2.applyColorMap(I_phi_unwrapped[i], cv2.COLORMAP_RAINBOW)
        image_phi[i] = cv2.resize(I_phi_unwrapped[i], (disp_w, disp_h))
    image = cv_to_tk(image_phi[cp])
    canvas_2[5].itemconfig(image_on_canvas_2[5], image = image)
    canvas_2[5].image = image
    
def compute_point_cloud(eventorigin):
    global refPts, image_phi, image_phi_masked
    
    if v_2[3].get() == "":
        x0 = int(eventorigin.x * ratio)
        y0 = int(eventorigin.y * ratio)
        refPts.append((x0, y0))
        cv2.circle(image_phi[cp],
                   center = (x0, y0),
                   radius = 3,
                   color = (0, 255, 0),
                   thickness = 1,
                   lineType = 2,
                   shift = 0)
        image = cv_to_tk(image_phi[cp])
        canvas_2[5].itemconfig(image_on_canvas_2[5], image = image)
        canvas_2[5].image = image
    if len(refPts) == 3 or v_2[3].get() != "":

        if len(refPts) == 3 or v_2[2].get() == "":
            choices = [cp]
        else:
            choices = np.arange(args.cam_num)
        for i in choices:
            step = int(v_2[1].get())
            
            mask = np.ones_like(phi_unwrapped[i].data, dtype=np.bool)
    
            if v_2[3].get() != "":
                h, w = gamma[i].shape
                thr_mean = np.median(gamma[i][h//6:5*h//6, w//6:5*w//6])
                thr_std = gamma[i][h//6:5*h//6, w//6:5*w//6].std()
                thr = thr_mean - float(v_2[3].get()) * thr_std
                imask = np.zeros((h,w))
                imask[gamma[i]>= thr] = 1
                imask = select_largest_obj(imask, lab_val=1, fill_holes=True,
                                       smooth_boundary=True, kernel_size=5)
                mask_idxs = np.where(imask==1)
                mask[mask_idxs] = False
            else:    
                mask[refPts[0][1]:refPts[2][1], refPts[0][0]:refPts[1][0]] = False
                
            phi_unwrapped_masked = np.ma.array(phi_unwrapped[i].data, mask=mask)
            
            if v_2[2].get() != "":
                I_c = Is_psi[i][3].astype('float')
                I_c = np.ma.array(I_c, mask=mask)
                idxs = np.where(I_c > float(v_2[2].get()) * I_c.max())
                offset_c = - phi_unwrapped_masked[idxs] + 2 * np.pi/float(v_2[0].get()) * (args.proj_width // 2)
                offset_c = np.median(offset_c)
                phi_unwrapped_masked = phi_unwrapped_masked + offset_c
                    
                ##### rejection check
                x = phi_unwrapped_masked[np.unique(idxs[0])].T.reshape(args.cam_width, -1)
                y = np.tile(np.arange(args.cam_width-1, -1, -1), (x.shape[1],1)).T
                y = np.ma.array(y, mask=x.mask)
    
                corr_coef = (((x-x.mean(axis=0)) * (y-y.mean(axis=0))).mean(axis=0)
                                /(x.std(axis=0) * y.std(axis=0))).mean()
                print("corr_coef = %.2f" %corr_coef)
            else: ### consider midpoint of the image
                x = phi_unwrapped_masked[disp_h]
                y = np.arange(args.cam_width-1, -1, -1)
                y = np.ma.array(y, mask=x.mask)
    
                corr_coef = (((x-x.mean()) * (y-y.mean())).mean()
                                /(x.std() * y.std()))
                print("corr_coef = %.2f" %corr_coef)
        
        
            pts_cloud, mask_idxs = point_cloud_generator(phi_unwrapped_masked,
                                                         args.proj_width,
                                                         float(v_2[0].get()),
                                                         rt_cam[i], rt_proj,
                                                         s = (step, step))
            
            np.save(args.psi_path + 'pts_cloud_%d.npy' %i, pts_cloud)
            np.save(args.psi_path + 'mask_idxs_%d.npy' %i, mask_idxs)
            frame_surf = 1/3*(Is_psi[i][0].astype('float') + Is_psi[i][1].astype('float') + Is_psi[i][2].astype('float'))
            frame_surf = cv2.cvtColor(np.uint8(frame_surf), cv2.COLOR_GRAY2BGR)
            np.save(args.psi_path + 'texture_%d.npy' %i, frame_surf)
            plot_surface_plotly(pts_cloud, frame_surf, mask_idxs, center_flag = True, psi_path = args.psi_path, cam_id = i)
            canvas_2[5].unbind("<Button 1>")
            refPts = []
            image_phi[i] = cv2.resize(I_phi_unwrapped[i], (disp_w, disp_h))
            I_phi_unwrapped_masked = np.uint8(exposure.rescale_intensity(phi_unwrapped_masked.filled(0), out_range=(0, 255)))
            I_phi_unwrapped_masked = cv2.applyColorMap(I_phi_unwrapped_masked, cv2.COLORMAP_RAINBOW)
            image_phi_masked[i] = cv2.resize(I_phi_unwrapped_masked, (disp_w, disp_h))
        image = cv_to_tk(image_phi[cp])
        canvas_2[5].itemconfig(image_on_canvas_2[5], image = image)
        canvas_2[5].image = image
        
        image = cv_to_tk(image_phi_masked[cp])
        canvas_2[6].itemconfig(image_on_canvas_2[6], image = image)
        canvas_2[6].image = image

def callback_psi(j):
    global Is_psi
    if j in [0,1,2,3]:
        for i in range(args.cam_num):
            if args.rgb and j<3:
                Is_psi[i][0] = frame[i][:,:,0]
                Is_psi[i][1] = frame[i][:,:,1]
                Is_psi[i][2] = frame[i][:,:,2]
            else:
                Is_psi[i][j] = cv2.cvtColor(frame[i], cv2.COLOR_BGR2GRAY)
        if args.rgb and j<3:
            choices = [0,1,2]
        else:
            choices = [j]
        for j in choices:
            gray = cv2.resize(Is_psi[cp][j], (disp_w, disp_h))
            image = cv_to_tk(gray)
            canvas_2[j+1].itemconfig(image_on_canvas_2[j+1], image = image)
            canvas_2[j+1].image = image
    if j == 4:
        if v_2[2].get() != "":
            compute_phase(0)
        else:
            canvas_2[4].bind("<Button 1>",compute_phase)
    if j == 5:
        if v_2[3].get() != "":
            compute_point_cloud(0)
        else:
            canvas_2[5].bind("<Button 1>",compute_point_cloud)
        
def callback_centerline():
    global gray
    for i in range(args.cam_num):
        gray[i] = cv2.cvtColor(frame[i], cv2.COLOR_BGR2GRAY).astype("float")
        max_ratio = float(v_0[0].get())
        idxs = np.where(gray[i] >= max_ratio * gray[i].max())
        gray[i] = cv2.cvtColor(np.uint8(gray[i]), cv2.COLOR_GRAY2BGR)
        for ii in range(len(idxs[0])):
            cv2.circle(gray[i],
                       center = (idxs[1][ii], idxs[0][ii]),
                       radius = 2,
                       color = (255, 0, 0),
                       thickness = 1,
                       lineType = 2,
                       shift = 0)
        gray[i] = cv2.resize(gray[i], (disp_w, disp_h))
    image = cv_to_tk(gray[cp])
    canvas_0[1].itemconfig(image_on_canvas_0[1], image = image)
    canvas_0[1].image = image


def callback_sync(j):
    global start_sync, v_1, color_diff_thr, intensity_thr, pixel_thr,\
    esync_vals, flags_sync
    
    flags_sync = [False]*5
    if j==0:
        if v_1[0].get() == "Start":
            start_sync = True
            v_1[0].set("Stop")
            color_diff_thr = float(v_1[1].get())
            intensity_thr = float(v_1[2].get())
            pixel_thr = int(v_1[3].get())
            T_1[0].delete("1.0", "end")
            T_1[0].insert(END,"\n"*23)
            esync_vals =["."]*24
        else:
            start_sync = False
            v_1[0].set("Start")
    if j==1:
        color_diff_thr = float(v_1[1].get())
        intensity_thr = float(v_1[2].get())
        pixel_thr = int(v_1[3].get())
        T_1[0].delete("1.0", "end")
        T_1[0].insert(END, "\n"*23)

def callback_apsi(j):
    global start_apsi, v_3, video_frames,\
            video_name, vid_cap, cnt_save, cnt_frame, times_save, corr_coef_save,\
            Is_1, Is_2, Is_3, Is_c, I_phi_masked, time_seq, I_seq, rgb_seq, pattern_id
    
    if j==0:
        if v_3[0].get() == "Start Recording":
            start_apsi = True
            v_3[0].set("Stop")
            for i in range(args.cam_num):
                video_frames[i] = []
        else:
            start_apsi = False
            v_3[0].set("Start Recording")
            for i in range(args.cam_num):
                video = cv2.VideoWriter(args.apsi_path + v_3[11].get()[:-4] + "_%d.mp4" %i,
                                        cv2.VideoWriter_fourcc(*'mp4v'),
                                        args.vid_rate,
                                        (args.cam_width,args.cam_height))
                for video_frame in video_frames[i]: 
                    video.write(video_frame)
                cv2.destroyAllWindows()
                video.release()
                video_frames[i] = []
    if j==1:
        "TODO automatic"
        
        PSI_automatic_func(video_name= v_3[11].get(),
                           rt_cam = rt_cam,
                           rt_proj = rt_proj,
                           lag=int(v_3[3].get()),
                           fr_cam=float(v_3[4].get()),
                           delay_proj = float(v_3[5].get()),
                           tol_bias = int(v_3[6].get()),
                           p = int(v_3[2].get()),
                           s = (int(v_3[1].get()),int(v_3[1].get())),
                           thr_cntr = float(v_3[7].get()),
                           color_diff_thr=int(v_3[8].get()),
                           pixel_thr=int(v_3[10].get()),
                           intensity_thr =int(v_3[9].get()),
                           cntr_thr = float(v_3[12].get()),
                           apsi_path = args.apsi_path,
                           w_proj = args.proj_width, cam_num = args.cam_num)

    if j==2:
        "TODO automatic"
        video_name = v_3[11].get()
        for i in range(args.cam_num):
            vid_cap[i] = cv2.VideoCapture(args.apsi_path + video_name[:-4] + "_%d.mp4" %i)
        cnt_save = 0
        cnt_frame = 0
        times_save = np.zeros((500, 2)) #max_save
        corr_coef_save = np.zeros((args.cam_num,500)) #max_save
        if args.rgb:
            n_pattern = 2
        else:
            n_pattern = 4
        time_seq = [None]*n_pattern
        I_seq = [[None for i in range(n_pattern)] for j in range(args.cam_num)]
        rgb_seq = [[None for i in range(n_pattern)] for j in range(args.cam_num)]
        pattern_id = 0
        callback_apsi(3)
    if j==3:
        vid_cap,\
        cnt_save,\
        cnt_frame,\
        times_save,\
        corr_coef_save,\
        Is_1, Is_2, Is_3, Is_c,\
        I_phi_masked, time_seq, I_seq, rgb_seq, pattern_id =\
        PSI_automatic_func(walk = True,
                           vid_cap = vid_cap,
                           video_name= video_name,
                           cnt_save = cnt_save,
                           cnt_frame = cnt_frame,
                           times_save = times_save,
                           corr_coef_save = corr_coef_save,
                           rt_cam = rt_cam,
                           rt_proj = rt_proj,
                           lag=int(v_3[3].get()),
                           fr_cam=float(v_3[4].get()),
                           delay_proj = float(v_3[5].get()),
                           tol_bias = int(v_3[6].get()),
                           p = int(v_3[2].get()),
                           s = (int(v_3[1].get()),int(v_3[1].get())),
                           thr_cntr = float(v_3[7].get()),
                           color_diff_thr=int(v_3[8].get()),
                           pixel_thr=int(v_3[10].get()),
                           intensity_thr =int(v_3[9].get()),
                           cntr_thr = float(v_3[12].get()),
                           apsi_path = args.apsi_path,
                           w_proj = args.proj_width,
                           time_seq = time_seq,
                           I_seq = I_seq,
                           rgb_seq = rgb_seq,
                           pattern_id = pattern_id)
        
        image = cv2.resize(Is_1[cp], (disp_w, disp_h))
        image = cv_to_tk(image)
        canvas_3[1].itemconfig(image_on_canvas_3[1], image = image)
        canvas_3[1].image = image
        
        image = cv2.resize(Is_2[cp], (disp_w, disp_h))
        image = cv_to_tk(image)
        canvas_3[2].itemconfig(image_on_canvas_3[2], image = image)
        canvas_3[2].image = image
        
        image = cv2.resize(Is_3[cp], (disp_w, disp_h))
        image = cv_to_tk(image)
        canvas_3[3].itemconfig(image_on_canvas_3[3], image = image)
        canvas_3[3].image = image
        
        image = cv2.resize(Is_c[cp], (disp_w, disp_h))
        image = cv_to_tk(image)
        canvas_3[4].itemconfig(image_on_canvas_3[4], image = image)
        canvas_3[4].image = image
        
        image = cv2.resize(I_phi_masked[cp], (disp_w, disp_h))
        image = cv_to_tk(image)
        canvas_3[5].itemconfig(image_on_canvas_3[5], image = image)
        canvas_3[5].image = image
        
def tasks(i):
    global flags_sync, video_frames, code_names
    if flag[i] == True:
        if i == 0:
            canvas_0[0].itemconfig(image_on_canvas_0[0], image = image[cp])
            canvas_0[0].image = image[cp]
        
        if i == 1:
            canvas_1[0].itemconfig(image_on_canvas_1[0], image = image[cp])
            canvas_1[0].image = image[cp]
            if start_sync == True:
                p_num, _ = esync_func(frame[cp], color_diff_thr, pixel_thr, intensity_thr)
                if flags_sync[p_num] == False:
                    flags_sync = [False]*5
                    flags_sync[p_num] = True
                    del esync_vals[0]
                    esync_vals.append(code_names[p_num])
                    T_1[0].delete("1.0", "end")
                    T_1[0].insert(END,"\n".join(esync_vals))
        if i == 2:
            canvas_2[0].itemconfig(image_on_canvas_2[0], image = image[cp])
            canvas_2[0].image = image[cp]
        
        if i == 3:
            canvas_3[0].itemconfig(image_on_canvas_3[0], image = image[cp])
            canvas_3[0].image = image[cp]
            if start_apsi == True and loop_cnt == 0:
                for j in range(args.cam_num):
                    video_frames[j].append(frame[j])
            
    if i>num_task-2:
        master.after(1, main_task)
    else:
        master.after(1, lambda j=i+1: tasks(j))
        

canvas_0 = [None]*2
image_on_canvas_0 = [None]*2
v_0 = [None]
canvas_1 = [None]
image_on_canvas_1 = [None]
v_1 = [None]*4
T_1 = [None]
canvas_2 = [None]*7
image_on_canvas_2 = [None]*7
v_2 = [None]*4
canvas_3 = [None]*6
image_on_canvas_3 = [None]*6
v_3 = [None]*13

def NewWindow(i):
    global flag, slave, canvas_0, canvas_1, canvas_2, canvas_3,\
    image_on_canvas_0, image_on_canvas_1, image_on_canvas_2, image_on_canvas_3,\
    v_0, v_1, v_2, v_3, T_1
    
    if i == 0:
        global gray
        I_z = np.zeros((disp_h,disp_w), dtype = np.uint8)
        gray = [I_z]*args.cam_num

        if flag[i] == True: 
            slave[i].destroy()
        flag[i] = True
        slave[i] = Toplevel(master)
        slave[i].protocol("WM_DELETE_WINDOW", lambda j=i: set_flag(j))
        slave[i].title("Centerline Test")
        
        canvas_0[0] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_0[0].grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_0[0] = canvas_0[0].create_image(0, 0, anchor=NW, image=image)  
        canvas_0[0].image = image
        
        canvas_0[1] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_0[1].grid(row=0, column=2, columnspan=2, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_0[1] = canvas_0[1].create_image(0, 0, anchor=NW, image=image)  
        canvas_0[1].image = image
        
        btn = Button(slave[i], text="Test", command = callback_centerline)
        btn.grid(row=1, column=0, columnspan=2, pady=5)
        
        label = Label(slave[i], text="max ratio")
        label.grid(row=1, column=2, pady=5, sticky=E)
        
        v_0[0] = StringVar()
        v_0[0].set("0.8")
        e = Entry(slave[i], textvariable=v_0[0])
        e.grid(row=1, column=3, pady=5, sticky=W)

    if i == 1:
        global start_sync, flags_sync, esync_vals, code_names,\
            color_diff_thr, intensity_thr, pixel_thr
        start_sync = False
        flags_sync = [False]*5
        esync_vals =["."]*24
        code_names = ["Blue", "Green", "Red", "Center", "None"]
        color_diff_thr, intensity_thr, pixel_thr = [None]*3
        
        if flag[i] == True: 
            slave[i].destroy()
        flag[i] = True
        slave[i] = Toplevel(master)
        slave[i].protocol("WM_DELETE_WINDOW", lambda j=i: set_flag(j))
        slave[i].title("Synchronization Test")
        
        canvas_1[0] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_1[0].grid(row=0, column=0, columnspan=2, rowspan=5, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_1[0] = canvas_1[0].create_image(0, 0, anchor=NW, image=image)  
        canvas_1[0].image = image
        
        v_1[0] = StringVar()
        v_1[0].set("Start")
        btn = Button(slave[i], textvariable=v_1[0], command = lambda j=0: callback_sync(j))
        btn.grid(row=5, column=0, columnspan=2, pady=5)
        
        btn = Button(slave[i], text="Update", command = lambda j=1: callback_sync(j))
        btn.grid(row=6, column=0, columnspan=2, pady=5)
        
        label = Label(slave[i], text="color diff thr")
        label.grid(row=7, column=0, pady=5, sticky=E)
        
        v_1[1] = StringVar()
        v_1[1].set("60")
        e = Entry(slave[i], textvariable=v_1[1])
        e.grid(row=7, column=1, pady=5, sticky=W)
        
        label = Label(slave[i], text="intensity thr")
        label.grid(row=8, column=0, pady=5, sticky=E)
        
        v_1[2] = StringVar()
        v_1[2].set("80")
        e = Entry(slave[i], textvariable=v_1[2])
        e.grid(row=8, column=1, pady=5, sticky=W)
        
        label = Label(slave[i], text="pixel thr")
        label.grid(row=9, column=0, pady=5, sticky=E)
        
        v_1[3] = StringVar()
        v_1[3].set("1000")
        e = Entry(slave[i], textvariable=v_1[3])
        e.grid(row=9, column=1, pady=5, sticky=W)

        T_1[0] = Text(slave[i], height=20, width=10)
        T_1[0].grid(row=0, column=2, rowspan=20, padx=5, sticky=W+E+N+S)
        T_1[0].insert(END,"\n"*23)
        
    if i == 2:
        global phi_unwrapped, I_phi_unwrapped, image_phi, gamma, refPts,\
            image_phi_masked, Is_psi
        I_z = np.zeros((disp_h,disp_w), dtype = np.uint8)
        phi_unwrapped = [None]*args.cam_num
        I_phi_unwrapped = [None]*args.cam_num
        image_phi = [I_z]*args.cam_num
        gamma = [None]*args.cam_num
        refPts = []
        image_phi_masked = [I_z]*2
        if args.cam_num == 1:
            Is_psi = [[I_z, I_z, I_z, I_z]]
        elif args.cam_num == 2:
            Is_psi = [[I_z, I_z, I_z, I_z],
                      [I_z, I_z, I_z, I_z]]


        if flag[i] == True: 
            slave[i].destroy()
        flag[i] = True
        slave[i] = Toplevel(master)
        slave[i].protocol("WM_DELETE_WINDOW", lambda j=i: set_flag(j))
        slave[i].title("Interferometry-Walkthrough")
        
        canvas_2[0] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_2[0].grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_2[0] = canvas_2[0].create_image(0, 0, anchor=NW, image=image)  
        canvas_2[0].image = image
        
        canvas_2[1] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_2[1].grid(row=0, column=2, columnspan=2, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_2[1] = canvas_2[1].create_image(0, 0, anchor=NW, image=image)  
        canvas_2[1].image = image
        
        canvas_2[2] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_2[2].grid(row=0, column=4, columnspan=2, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_2[2] = canvas_2[2].create_image(0, 0, anchor=NW, image=image)  
        canvas_2[2].image = image
        
        canvas_2[3] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_2[3].grid(row=0, column=6, columnspan=2, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_2[3] = canvas_2[3].create_image(0, 0, anchor=NW, image=image)  
        canvas_2[3].image = image
        
        canvas_2[4] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_2[4].grid(row=2, column=2, columnspan=2, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_2[4] = canvas_2[4].create_image(0, 0, anchor=NW, image=image)  
        canvas_2[4].image = image
        
        canvas_2[5] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_2[5].grid(row=2, column=4, columnspan=2, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_2[5] = canvas_2[5].create_image(0, 0, anchor=NW, image=image)  
        canvas_2[5].image = image
        
        canvas_2[6] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_2[6].grid(row=2, column=6, columnspan=2, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_2[6] = canvas_2[6].create_image(0, 0, anchor=NW, image=image)  
        canvas_2[6].image = image
        
        btn = Button(slave[i], text="Capture 'first' phase", command = lambda j=0: callback_psi(j))
        btn.grid(row=1, column=2, columnspan=2, pady=5)

        btn = Button(slave[i], text="Capture 'second' phase", command = lambda j=1: callback_psi(j))
        btn.grid(row=1, column=4, columnspan=2, pady=5)
        
        btn = Button(slave[i], text="Capture 'third' phase", command = lambda j=2: callback_psi(j))
        btn.grid(row=1, column=6, columnspan=2, pady=5)
        
        btn = Button(slave[i], text="Capture centerpoint", command = lambda j=3: callback_psi(j))
        btn.grid(row=3, column=2, columnspan=2, pady=5)
        
        btn = Button(slave[i], text="Compute phase map", command = lambda j=4: callback_psi(j))
        btn.grid(row=3, column=4, columnspan=2, pady=5)
        
        btn = Button(slave[i], text="Generate point cloud", command = lambda j=5: callback_psi(j))
        btn.grid(row=3, column=6, columnspan=2, pady=5)
        
        
        label = Label(slave[i], text="phase pitch")
        label.grid(row=1, column=0, pady=5, sticky=E)
        
        v_2[0] = StringVar()
        v_2[0].set("25")
        e = Entry(slave[i], textvariable=v_2[0])
        e.grid(row=1, column=1, pady=5, sticky=W)
        
        label = Label(slave[i], text="down sampling")
        label.grid(row=2, column=0, pady=5, sticky=E)
        
        v_2[1] = StringVar()
        v_2[1].set("5")
        e = Entry(slave[i], textvariable=v_2[1])
        e.grid(row=2, column=1, pady=5, sticky=W)
        
        label = Label(slave[i], text="center thr")
        label.grid(row=3, column=0, pady=5, sticky=E)
        
        v_2[2] = StringVar()
        v_2[2].set("")
        e = Entry(slave[i], textvariable=v_2[2])
        e.grid(row=3, column=1, pady=5, sticky=W)
        
        label = Label(slave[i], text="mask thr")
        label.grid(row=4, column=0, pady=5, sticky=E)
        
        v_2[3] = StringVar()
        v_2[3].set("")
        e = Entry(slave[i], textvariable=v_2[3])
        e.grid(row=4, column=1, pady=5, sticky=W)

    if i == 3:
        global start_apsi, video_frames, video_name, cnt_save, vid_cap,\
            Is_1, Is_2, Is_3, Is_c
        start_apsi = False
        if args.cam_num == 1:
            video_frames = [[]]
        elif args.cam_num == 2:
            video_frames = [[],[]]
        video_name, cnt_save = [None]*2
        vid_cap = [None]*args.cam_num
        I_z = np.zeros((disp_h,disp_w), dtype = np.uint8)
        Is_1 = [I_z]*args.cam_num
        Is_2 = [I_z]*args.cam_num
        Is_3 = [I_z]*args.cam_num
        Is_c = [I_z]*args.cam_num
    
        if flag[i] == True: 
            slave[i].destroy()
        flag[i] = True
        slave[i] = Toplevel(master)
        slave[i].protocol("WM_DELETE_WINDOW", lambda j=i: set_flag(j))
        slave[i].title("Automatic Interferometry")
        
        canvas_3[0] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_3[0].grid(row=0, column=0, columnspan=2, rowspan=1, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_3[0] = canvas_3[0].create_image(0, 0, anchor=NW, image=image)  
        canvas_3[0].image = image
        
        canvas_3[1] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_3[1].grid(row=0, column=2, columnspan=2, rowspan=1, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_3[1] = canvas_3[1].create_image(0, 0, anchor=NW, image=image)  
        canvas_3[1].image = image
        
        canvas_3[2] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_3[2].grid(row=0, column=4, columnspan=2, rowspan=1, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_3[2] = canvas_3[2].create_image(0, 0, anchor=NW, image=image)  
        canvas_3[2].image = image
        
        canvas_3[3] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_3[3].grid(row=0, column=6, columnspan=2, rowspan=1, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_3[3] = canvas_3[3].create_image(0, 0, anchor=NW, image=image)  
        canvas_3[3].image = image

        canvas_3[4] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_3[4].grid(row=1, column=2, columnspan=2, rowspan=2, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_3[4] = canvas_3[4].create_image(0, 0, anchor=NW, image=image)  
        canvas_3[4].image = image
        
        canvas_3[5] = Canvas(slave[i], width = disp_w, height = disp_h)  
        canvas_3[5].grid(row=1, column=4, columnspan=2, rowspan=2, padx=5, pady=5)
        frame = np.zeros((disp_h, disp_w))
        image = cv_to_tk(frame)
        image_on_canvas_3[5] = canvas_3[5].create_image(0, 0, anchor=NW, image=image)  
        canvas_3[5].image = image
        
        
        v_3[0] = StringVar()
        v_3[0].set("Start Recording")
        btn = Button(slave[i], textvariable=v_3[0], command = lambda j=0: callback_apsi(j))
        btn.grid(row=1, column=0, columnspan=1, pady=5)
        
        btn = Button(slave[i], text="Process", command = lambda j=1: callback_apsi(j))
        btn.grid(row=1, column=1, columnspan=1, pady=5)
        
        btn = Button(slave[i], text="Walk through", command = lambda j=2: callback_apsi(j))
        btn.grid(row=2, column=0, columnspan=1, pady=5)
        
        btn = Button(slave[i], text="Go Next", command = lambda j=3: callback_apsi(j))
        btn.grid(row=2, column=1, columnspan=1, pady=5)
        
        label = Label(slave[i], text="down sampling")
        label.grid(row=3, column=0, pady=5, sticky=E)
        
        v_3[1] = StringVar()
        v_3[1].set("5")
        e = Entry(slave[i], textvariable=v_3[1])
        e.grid(row=3, column=1, pady=5, sticky=W)
        
        label = Label(slave[i], text="phase pitch")
        label.grid(row=4, column=0, pady=5, sticky=E)
        
        v_3[2] = StringVar()
        v_3[2].set("25")
        e = Entry(slave[i], textvariable=v_3[2])
        e.grid(row=4, column=1, pady=5, sticky=W)
        
        label = Label(slave[i], text="lag")
        label.grid(row=3, column=2, pady=5, sticky=E)
        
        v_3[3] = StringVar()
        v_3[3].set("3")
        e = Entry(slave[i], textvariable=v_3[3])
        e.grid(row=3, column=3, pady=5, sticky=W)
        
        label = Label(slave[i], text="camera fr")
        label.grid(row=3, column=4, pady=5, sticky=E)
        
        v_3[4] = StringVar()
        v_3[4].set("30")
        e = Entry(slave[i], textvariable=v_3[4])
        e.grid(row=3, column=5, pady=5, sticky=W)
        
        label = Label(slave[i], text="projector delay")
        label.grid(row=4, column=2, pady=5, sticky=E)
        
        v_3[5] = StringVar()
        v_3[5].set("0.2")
        e = Entry(slave[i], textvariable=v_3[5])
        e.grid(row=4, column=3, pady=5, sticky=W)
        
        label = Label(slave[i], text="tol bias")
        label.grid(row=4, column=4, pady=5, sticky=E)
        
        v_3[6] = StringVar()
        v_3[6].set("5")
        e = Entry(slave[i], textvariable=v_3[6])
        e.grid(row=4, column=5, pady=5, sticky=W)
        
        label = Label(slave[i], text="center thr")
        label.grid(row=5, column=0, pady=5, sticky=E)
        
        v_3[7] = StringVar()
        v_3[7].set("0.5")
        e = Entry(slave[i], textvariable=v_3[7])
        e.grid(row=5, column=1, pady=5, sticky=W)
        
        label = Label(slave[i], text="color diff thr")
        label.grid(row=5, column=2, pady=5, sticky=E)
        
        v_3[8] = StringVar()
        v_3[8].set("60")
        e = Entry(slave[i], textvariable=v_3[8])
        e.grid(row=5, column=3, pady=5, sticky=W)
        
        label = Label(slave[i], text="intensity thr")
        label.grid(row=5, column=4, pady=5, sticky=E)
        
        v_3[9] = StringVar()
        v_3[9].set("80")
        e = Entry(slave[i], textvariable=v_3[9])
        e.grid(row=5, column=5, pady=5, sticky=W)
        
        label = Label(slave[i], text="pixel thr")
        label.grid(row=6, column=0, pady=5, sticky=E)
        
        v_3[10] = StringVar()
        v_3[10].set("1000")
        e = Entry(slave[i], textvariable=v_3[10])
        e.grid(row=6, column=1, pady=5, sticky=W)
        
        label = Label(slave[i], text="video name")
        label.grid(row=6, column=2, pady=5, sticky=E)
        
        v_3[11] = StringVar()
        v_3[11].set("video_0.mp4")
        e = Entry(slave[i], textvariable=v_3[11])
        e.grid(row=6, column=3, pady=5, sticky=W)
        
        label = Label(slave[i], text="mask thr ratio")
        label.grid(row=6, column=4, pady=5, sticky=E)
        
        v_3[12] = StringVar()
        v_3[12].set("1.2")
        e = Entry(slave[i], textvariable=v_3[12])
        e.grid(row=6, column=5, pady=5, sticky=W)

def switch_camera():
    global cp
    cp = 1 - cp
    v_b.set("Switch to Camera %d" %(1-cp))
    for i in range(num_task):
        if flag[i] == True:
            if i == 0:
                image = cv_to_tk(gray[cp])
                canvas_0[1].itemconfig(image_on_canvas_0[1], image = image)
                canvas_0[1].image = image
            if i == 2:
                for j in [0,1,2,3]:
                    image = cv2.resize(Is_psi[cp][j], (disp_w, disp_h))
                    image = cv_to_tk(image)
                    canvas_2[j+1].itemconfig(image_on_canvas_2[j+1], image = image)
                    canvas_2[j+1].image = image
                
                image = cv_to_tk(image_phi[cp])
                canvas_2[5].itemconfig(image_on_canvas_2[5], image = image)
                canvas_2[5].image = image
        
                image = cv_to_tk(image_phi_masked[cp])
                canvas_2[6].itemconfig(image_on_canvas_2[6], image = image)
                canvas_2[6].image = image
            if i == 3:
                image = cv2.resize(Is_1[cp], (disp_w, disp_h))
                image = cv_to_tk(image)
                canvas_3[1].itemconfig(image_on_canvas_3[1], image = image)
                canvas_3[1].image = image
                
                image = cv2.resize(Is_2[cp], (disp_w, disp_h))
                image = cv_to_tk(image)
                canvas_3[2].itemconfig(image_on_canvas_3[2], image = image)
                canvas_3[2].image = image
                
                image = cv2.resize(Is_3[cp], (disp_w, disp_h))
                image = cv_to_tk(image)
                canvas_3[3].itemconfig(image_on_canvas_3[3], image = image)
                canvas_3[3].image = image
                
                image = cv2.resize(Is_c[cp], (disp_w, disp_h))
                image = cv_to_tk(image)
                canvas_3[4].itemconfig(image_on_canvas_3[4], image = image)
                canvas_3[4].image = image
                
                image = cv2.resize(I_phi_masked[cp], (disp_w, disp_h))
                image = cv_to_tk(image)
                canvas_3[5].itemconfig(image_on_canvas_3[5], image = image)
                canvas_3[5].image = image
                    

if args.load == True:
    cap = [None]*args.cam_num
    for i in range(args.cam_num):
        cap[i] = cv2.VideoCapture(args.apsi_path + args.vid_name[:-4] + "_%d" %i + args.vid_name[-4:])
    loop_max = int(np.round(1 / args.vid_rate * 1e3 / num_task))
else:    
    if args.cam_type == "usb":
        loop_max = 1
        cap = [None]*args.cam_num
        for i in range(args.cam_num):
            cap[i] = cv2.VideoCapture(i)
            cap[i].set(5, 30) # set frame rate to 30 frames per second
    elif args.cam_type == "flir":
        loop_max = max(1,int(np.round(1 / args.vid_rate * 1e3 / num_task / 30)))
        import PySpin
        sys.path.append(sys.path[0][:-3] + 'capture')
        from flir_camera_setup import *

        class cam_params:
            frame_rate_enable = False #(True, False)
            frame_rate_to_set = 30.0
            exposure_auto = 'Continuous' #(Off, Continuous)
            exposure_time_to_set = 6700
            gain_auto = 'Continuous' #(Off, Continuous)
            gain_to_set = 4.0
            white_auto = 'Continuous' #(Off, Continuous)
            gamma_to_set = 1.0
            trigger_mode = 'On' #(Off, On)
            CHOSEN_TRIGGER = 2 # 1 for software, 2 for hardware
            line_num = 3
            trigger_edge = 'FallingEdge' #(RisingEdge, FallingEdge)
            if args.rgb:
                pixel_format = "bgr8" #bgr8
            else:
                pixel_format = "mono8" #bgr8
        
        for i in range(args.cam_num):
            configure_camera(cam_params, cam_id = i)
        
        system = PySpin.System.GetInstance()
        cam_list = system.GetCameras()
        cam = [None]*args.cam_num
        for i in range(args.cam_num):
            _, cam[i] = list(enumerate(cam_list))[i]
            cam[i].Init()
            cam[i].BeginAcquisition()
    
loop_cnt = 0
cp = 0

master = Tk()
master.title("Phase Shift Interferometry")

### constrainting display dimensions in gui
screen_height = master.winfo_screenheight()
disp_h = screen_height//5
disp_w = int(disp_h/args.cam_height * args.cam_width)
ratio = args.cam_height/disp_h
###

if args.cam_num == 2:
    v_b = StringVar()
    v_b.set("Switch to Camera %d" %(1-cp))
    btn = Button(master, textvariable=v_b,
                 height=5, width=20,
                 command = switch_camera)
    btn.grid(row=0, column=0, pady=5)

btn = Button(master, text="Centerline Test",
             height=5, width=20,
             command = lambda i=0: NewWindow(i))
btn.grid(row=1, column=0, pady=5)

btn = Button(master, text="Synchronization Test",
             height=5, width=20,
             command = lambda i=1: NewWindow(i))
btn.grid(row=2, column=0, pady=5)

btn = Button(master, text="Interferometry-Walkthrough",
             height=5, width=20,
             command = lambda i=2: NewWindow(i))
btn.grid(row=3, column=0, pady=5)

btn = Button(master, text="Automatic Interferometry",
             height=5, width=20,
             command = lambda i=3: NewWindow(i))
btn.grid(row=4, column=0, pady=5)

master.after(1, lambda i=0: tasks(i))
master.mainloop()