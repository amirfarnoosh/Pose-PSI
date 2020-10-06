
"""
3D plotting of a point cloud using plotly
"""

import plotly.graph_objs as go
from plotly.offline import plot

import numpy as np
from scipy.spatial import Delaunay

def plot_surface_plotly(pts_cloud,
                        frame,
                        mask_idxs,
                        times = None,
                        axis_flag = False,
                        center_flag = False,
                        psi_path = "./psi_files/",
                        cam_id = 0,
                        show = True):
    """ visualize point cloud with rgb overlay.
    Args:
        pts_cloud (2D array): A 2D numpy array of size [N,3] including 3D points.
        frame (3D array): RGB image of the scene.
        mask_idxs (2D tuple): A tuple including the corresponding indices of
            3D points in RGB frame.
        times (tuple): a tuple including start and end times of acquisition.
            If provided it will be included in plot title.
        axis_flag (boolean): whether to keep axis or not.
        center_flag (boolean): whether to bring point cloud to center (0,0,0).
        psi_path (string): path to dump plot.
        cam_id (int): camera ID. This will be appended to file name
        show (boolean): whether to save and show the plot or
            return plot-data and layout for further processes. 
    Returns: (if show is False)
        plot-data and layout for further manipulation 
    """
    
    # functions #
    def map_rgb(R, G, B):

        return 'rgb('+'{:d}'.format(int(R))+','+'{:d}'.format(int(G))+','+'{:d}'.format(int(B))+')'


    def tri_indices(simplices):
        #simplices is a numpy array defining the simplices of the triangularization
        #returns the lists of indices i, j, k
    
        return ([triplet[c] for triplet in simplices] for c in range(3))
    
    def plotly_trisurf(x, y, z, simplices, frame, mask_idxs):
        #x, y, z are lists of coordinates of the triangle vertices 
        #simplices are the simplices that define the triangularization;
        #simplices  is a numpy array of shape (no_triangles, 3)
       
        colormap = [np.mean(frame[mask_idxs[0][simplices[i]], mask_idxs[1][simplices[i]]],axis=0)
                    for i in range(len(simplices))]
        facecolor=[map_rgb(r, g, b) for b, g, r in colormap]
        I,J,K=tri_indices(simplices)
    
        triangles=go.Mesh3d(x=x,
                         y=y,
                         z=z,
                         facecolor=facecolor,
                         i=I,
                         j=J,
                         k=K,
                         name=''
                        )
    
        return [triangles]
    
    
    #============
    # plot
    #============
    
    pts_cloud = np.asarray(pts_cloud)
    x = pts_cloud[:,0]
    y = pts_cloud[:,1]
    z = pts_cloud[:,2]
    
    if center_flag == True:
        x = x - x.mean()
        y = y - y.mean()
        z = z - z.mean()
    
    # Triangulate parameter space to determine the triangles
    points2D=np.vstack([mask_idxs[1], mask_idxs[0]]).T
    tri = Delaunay(points2D)
    
    # Plot the surface.  The triangles in parameter space determine which x, y, z
    # points are connected by an edge.
    
    data1=plotly_trisurf(x,y,z, tri.simplices, frame, mask_idxs)

    # set the layout
    if times is not None:
        title = 'object triangulation %.2f-%.2f' %(times[0], times[1])
    else:
        title='object triangulation'
    if axis_flag == True:
        
        axis = dict(
                showbackground=True,
                backgroundcolor="rgb(230, 230,230)",
                gridcolor="rgb(255, 255, 255)",
                zerolinecolor="rgb(255, 255, 255)",
                )
    
        layout = go.Layout(
                title=title,
                width=800,
                height=800,
                scene=dict(
                        xaxis=dict(axis),
                        yaxis=dict(axis),
                        zaxis=dict(axis),
                        aspectratio=dict(
                                x=1,
                                y=1,
                                z=1
                                ),
                                )
                        )
      
    else:
        noaxis=dict(showbackground=False,
                showline=False,  
                zeroline=False,
                showgrid=False,
                showticklabels=False,
                title='' 
              )
        
        layout = go.Layout(
                title=title,
                width=800,
                height=800,
                scene=dict(
                        xaxis=noaxis,
                        yaxis=noaxis,
                        zaxis=noaxis,
                        aspectratio=dict(
                                x=1,
                                y=1,
                                z=1
                                ),
                        camera=dict(eye=dict(x=0, y=0, z= -2))
                                )
                        )
    if show == True:
        fig1 = go.Figure(data=data1, layout=layout)
    
        plot(fig1, filename=psi_path + 'object-trisurfs_%d.html' %cam_id)
    else:
        return data1[0], layout

