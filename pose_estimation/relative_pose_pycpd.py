
"""
Pose extraction by point clouds registation using 
coherent point drift (CPD) algorithm:
    https://github.com/siavashk/pycpd
"""

from functools import partial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pycpd.rigid_registration
from pycpd import rigid_registration

def visualize(iteration, error, X, Y, ax):
    
    '''Visualizing point clouds registration progress.
    Args:
        iteration ([int]): iteration number.
        error ([float]): registration error.
        X (2D array): 2D numpy array of size [N,3] containing
            3D coordinates of registered source at i^th iteration.
        Y (2D array): 2D numpy array of size [N,3] containing
            3D coordinates of destination.
    Returns:
        None: plots progress.
    '''
    plt.cla()
    ax.scatter(X[:,0],  X[:,1], X[:,2], color='red', label='Target')
    ax.scatter(Y[:,0],  Y[:,1], Y[:,2], color='blue', label='Source')
    ax.text2D(0.87, 0.92,
              'Iteration: {:d}\nError: {:06.4f}'.format(iteration, error),
              horizontalalignment='center', verticalalignment='center',
              transform=ax.transAxes, fontsize='x-large')
    ax.legend(loc='upper left', fontsize='x-large')
    plt.draw()
    plt.pause(0.001)

def relative_pose_pycpd(srcPC, dstPC, fig_flag = False):
    
    '''Estimating relative pose.
    Args:
        srcPC (2D array): 2D numpy array of size [N,3] 
            containing 3D coordinates of the source point cloud points.
        dstPC (2D array): 2D numpy array of size [N,3] 
            containing 3D coordinates of the destination point cloud points.
        fig_flag (boolean): whether to visualize progress or not.
            Default is False.
    Returns:
        rotation matrix, and translation vector.
    '''
    
    reg = rigid_registration(**{ 'X': srcPC, 'Y':dstPC })
    
    if fig_flag == True:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        callback = partial(visualize, ax=ax)
        params = reg.register(callback)
        plt.show()
    else:
        params = reg.register()
    
    return params[1][0], params[1][1]