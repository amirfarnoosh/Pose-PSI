
"""
Pose extraction by point clouds registation using 
robust iterative closest point (ICP) algorithm:
  https://docs.opencv.org/3.4/dc/d9b/classcv_1_1ppf__match__3d_1_1ICP.html
  https://docs.opencv.org/trunk/d9/d25/group__surface__matching.html#gsc.tab=0
    
The output poses transform the models (srcPC) onto the scene(dstPC).
"""

import cv2

NumNeighbors = 8
FlipViewpoint = False

iterations = 30
tolerence = 0.05
rejectionScale = 2.5
numLevels = 6
sampleType = 0
numMaxCorr = 1
viewpoint = (0,0,1)

def relative_pose_cv2(srcPC, dstPC):
    
    '''Estimating relative pose.
    Args:
        srcPC (2D array): 2D numpy array of size [N,3] 
            containing 3D coordinates of the source point cloud points.
        dstPC (2D array): 2D numpy array of size [N,3] 
            containing 3D coordinates of the destination point cloud points.
    Returns:
        retval (boolean): whether registration was successful or not.
        residual (float): residual error
        pose (2D array): transformation matrix as an array of size [4,4].
    '''

    retval, srcPCNormals = cv2.ppf_match_3d.computeNormalsPC3d(srcPC,
                                                               NumNeighbors,
                                                               FlipViewpoint
                                                               ,viewpoint
                                                               )
    retval, dstPCNormals = cv2.ppf_match_3d.computeNormalsPC3d(dstPC,
                                                               NumNeighbors,
                                                               FlipViewpoint
                                                               ,viewpoint
                                                               )
    
    """ computeNormalsPC3d
    [in]	PC	Input point cloud to compute the normals for.
    [out]	PCNormals	Output point cloud
    [in]	NumNeighbors	Number of neighbors to take into account in a local region
    [in]	FlipViewpoint	Should normals be flipped to a viewing direction?
    [in]	viewpoint
    
    """
    
#    icp = cv2.ppf_match_3d_ICP(iterations,
#                               tolerence,
#                               rejectionScale,
#                               numLevels,
#                               sampleType,
#                               numMaxCorr)
    icp = cv2.ppf_match_3d_ICP()
    
    """
    ICP constructor with default arguments. ppf_match_3d_ICP
    
    Parameters
    [in]	iterations	
    [in]	tolerence	Controls the accuracy of registration at each iteration
        of ICP.
    [in]	rejectionScale	Robust outlier rejection is applied for robustness. 
        This value actually corresponds to the standard deviation coefficient.
        Points with rejectionScale * &sigma are ignored during registration.
    [in]	numLevels	Number of pyramid levels to proceed.
        Deep pyramids increase speed but decrease accuracy.
        Too coarse pyramids might have computational overhead on top of the
        inaccurate registrtaion. This parameter should be chosen to optimize
        a balance. Typical values range from 4 to 10.
    [in]	sampleType	Currently this parameter is ignored and only uniform
        sampling is applied. Leave it as 0.
    [in]	numMaxCorr	Currently this parameter is ignored and only PickyICP
        is applied. Leave it as 1.
    
    const int 	iterations,
    const float 	tolerence = 0.05f,
    const float 	rejectionScale = 2.5f,
    const int 	numLevels = 6,
    const int 	sampleType = ICP::ICP_SAMPLING_TYPE_UNIFORM,
    const int 	numMaxCorr = 1
    """
    
    
    retval, residual, pose	=	icp.registerModelToScene(srcPCNormals, 
                                                         dstPCNormals)
    
    """registerModelToScene
    [in]	srcPC	The input point cloud for the model. Expected to have
        the normals (Nx6). Currently, CV_32F is the only supported data type.
    [in]	dstPC	The input point cloud for the scene. Currently, CV_32F
        is the only supported data type.
    [in,out]	poses	Input poses to start with but also list output of poses.
    """
    
    return retval, residual, pose