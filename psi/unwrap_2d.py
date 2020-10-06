
"""
Unwrapp a phase map using a fast two-dimensional method.
For more information:
https://scikit-image.org/docs/dev/auto_examples/filters/plot_phase_unwrap.html#id2
https://pypi.org/project/unwrap/
"""

import numpy as np 
from matplotlib import pyplot as plt
from skimage.restoration import unwrap_phase

def unwrap_2d(phi, mask_idxs = None, fig_flag = True, wrap = False):
    
    """ 2D phase unwrapping using skimage package.
    Args:
        phi (2D array): A 2D array of angles (phase map) in radians,
            in the range [-pi, pi].
        mask_idxs (2D tuple): A tuple including indexes of desired
            image pixels on which unwrapping will be done. Default is None,
            and unwrapping will be done on the entire image.
        fig_flag (boolean): whether to show unwrapping results or not.
        wrap (boolean): whether to apply x-/y-axis wrap around. 
            Default is False.
    Returns:
        phi_unwrapped (2D array): Unwrapped phase map image 
            with the same size as input. It will be a
            masked ndarray if mask_idxs are provided.
    """
    if mask_idxs is not None:
        # Mask the image
        mask = np.ones_like(phi, dtype=np.bool)
        mask[mask_idxs] = False
        
        phi_masked = np.ma.array(phi, mask=mask)
    else:
        phi_masked = phi
    
    # Unwrap phi without wrap around
    phi_unwrapped = unwrap_phase(phi_masked, wrap_around=(wrap, wrap))
    
    if fig_flag == True:
        
        fig, ax = plt.subplots(2, 1)
        ax1, ax2 = ax.ravel()
        
        fig.colorbar(ax1.imshow(phi_masked,
                     cmap='rainbow',
                     vmin=-np.pi, vmax =np.pi),
                     ax=ax1)
        ax1.set_title('masked phase')
        
        fig.colorbar(ax2.imshow(phi_unwrapped, cmap='rainbow'), ax=ax2)
        ax2.set_title('unwrapped phase')
    
        plt.show()
    
    return phi_unwrapped 