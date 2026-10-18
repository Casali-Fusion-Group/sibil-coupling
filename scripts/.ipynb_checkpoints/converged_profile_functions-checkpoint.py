import numpy as np

def end_biased_convergence(var, _N, _L):
    if len(var.shape) == 4:
        var = np.sum(var, axis=3) / var.shape[3]
    (t_all,ix,jy) = var.shape
    N = _N if _N is not None else 1
    L = _L if _L is not None else t_all
    var_tilde = np.zeros((L,ix,jy))
    for l in range(L):
        if l == 0:
            var_tilde[l,:,:] = var[l,:,:]
        elif l < N:
            if l == 0:
                var_tilde[l,:,:] = var[l,:,:]
            else:
                var_tilde[l,:,:] = np.maximum(0,(1/N) * np.sum(var[:l,:,:],axis=0)+((N-l)/N)*var[(l-1),:,:])
        elif l >= N:
            a = 1 - (1/N) 
            A_l = ( 1 - a ) / ( 1 - a**(l+1) )
            
            var_tilde[l,:,:] = np.maximum( 0 , A_l * np.sum( a**(L-l) * var[:l,:,:] , axis=0) )
            
    return var_tilde

def unbiased_convergence(var, _N, _L):
    if len(var.shape) == 4:
        var = np.sum(var, axis=3) / var.shape[3]
    (t_all,ix,jy) = var.shape
    N = _N if _N is not None else 1
    L = _L if _L is not None else t_all
    var_tilde = np.zeros((L,ix,jy))
    for l in range(L):
        if l == 0:
            var_tilde[l,:,:] = var[l,:,:]
        elif l < N:
            if l == 0:
                var_tilde[l,:,:] = var[l,:,:]
            else:
                var_tilde[l,:,:] = np.maximum(0,(1/N) * np.sum(var[:l,:,:],axis=0)+((N-l)/N)*var[(l-1),:,:])
        elif l >= N:
            var_tilde[l,:,:] = np.maximum(0,(1/L) * np.sum(var[:l,:,:],axis=0))
            
    return var_tilde