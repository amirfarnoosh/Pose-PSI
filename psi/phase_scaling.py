
import numpy as np

def phase_scaling(phi_unwrapped, p, w_proj, I_c = None, thr_cntr = 1):
    
    h, w = phi_unwrapped.shape
    # --- phase shifting/scaling
    if I_c is not None:
        
        I_c = np.ma.array(I_c, mask=phi_unwrapped.mask)
        idxs = np.where(I_c > thr_cntr * I_c.max())
        offset_c = - phi_unwrapped[idxs] + 2 * np.pi/p * (w_proj // 2)
        offset_c = np.median(offset_c)
        phi_unwrapped = phi_unwrapped + offset_c
        
        ##### rejection check
        x = phi_unwrapped[np.unique(idxs[0])].T.reshape(w, -1)
        y = np.tile(np.arange(w-1, -1, -1), (x.shape[1],1)).T
        y = np.ma.array(y, mask=x.mask)

        corr_coef = (((x-x.mean(axis=0)) * (y-y.mean(axis=0))).mean(axis=0)
                        /(x.std(axis=0) * y.std(axis=0))).mean()
        
    else:
        phi_min = phi_unwrapped.min()
        phi_max = phi_unwrapped.max()
        
        phi_std_min = phi_unwrapped[(phi_unwrapped - phi_min)<2*np.pi/p*(w_proj-1)].std()
        phi_std_max = phi_unwrapped[(phi_unwrapped - phi_max + 2*np.pi/p*(w_proj-1))>0].std()
        
        if phi_std_max >= phi_std_min:
            offset = - phi_max + 2 * np.pi/p * (w_proj-1)
        else:
            offset = -phi_min
        phi_unwrapped = phi_unwrapped + offset
        
        ##### rejection check
        x = phi_unwrapped[h//2]
        y = np.arange(w-1, -1, -1)
        y = np.ma.array(y, mask=x.mask)
        corr_coef = (((x-x.mean()) * (y-y.mean())).mean()
                     /(x.std() * y.std()))
    # -----------------------------------
    
    return phi_unwrapped, corr_coef