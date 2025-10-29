import deepxde as dde
import deepxde.backend as bkd
from ..utils import matmul, jacobian

def minmax_scale(x, lb, ub, scale=2.0, offset=1.0):
    """
    min-max scale
    """
    return 1.0/(ub - lb)*scale*(x -lb) - offset

def up_scale(x, lb, ub, scale=0.5, offset=1.0):
    """
    reverse min-max scale
    """
    return lb + scale*(x + offset)*(ub - lb)


def fourier_feature(x, B):
    """
    Apply Fourier Feature Transform
    """
    return bkd.concat([
                      bkd.cos(matmul(x, B)),
                      bkd.sin(matmul(x, B))
                      ], 
                      1)

def default_float_type():
    """ 
    Return the default float type according to the backend used
    """
    return bkd.data_type_dict[dde.config.default_float()]

####################
## output transforms for D-HNN/MC-exact

def transform_MC_HD(x,y):
    """
    transform D,R into u,v,a
    """
    D = y[:,0]
    R = y[:,1]
    H = y[:,2]
    dx_Dx = jacobian(D, x, i=0, j=xid)
    dy_Dy = jacobian(D, x, i=0, j=yid)
    dx_Rx = jacobian(R, x, i=0, j=xid)
    dy_Ry = jacobian(R, x, i=0, j=yid)
    u = (dx_Dx-dy_Ry)/H
    v = (dy_Dy+dx_Rx)/H
    a = dx_Dx + dy_Dy
    return bkd.stack((u,v,a,H),dim=1)

def transform_SSA_HD(x,y):
    """
    transform D,R into u,v,a
    """
    D = y[:,0]
    R = y[:,1]
    s = y[:,2]
    H = y[:,3]
    C = y[:,4]
    dx_Dx = jacobian(D, x, i=0, j=xid)
    dy_Dy = jacobian(D, x, i=0, j=yid)
    dx_Rx = jacobian(R, x, i=0, j=xid)
    dy_Ry = jacobian(R, x, i=0, j=yid)
    u = (dx_Dx-dy_Ry)/H
    v = (dy_Dy+dx_Rx)/H
    a = dx_Dx + dy_Dy
    return bkd.stack((u,v,s,H,a,C),dim=1)