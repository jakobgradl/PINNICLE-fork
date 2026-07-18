import deepxde as dde
import deepxde.backend as bkd
from deepxde.backend import jax, abs
from . import EquationBase, Constants, Physics
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian, slice_function_jax


################
################

class SSAweakEquationParamter(EquationParameter, Constants):
    """default parameters for SSA_weak
    """
    _EQUATION_TYPE = 'SSA_weak'
    def __init__(self, param_dict={}):
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 's', 'H', 'C']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6, 1.0e-8]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1e0,1e0]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08,   # -8 degree C, cuffey
                'm': 3, # exponent of the Weertman friction law
                'rho':917,
                'g':9.81,
                }
class SSA_weak(EquationBase): #{{{
    """ variational (weak-form) SSA on 2D problem with uniform B
    """
    _EQUATION_TYPE = 'SSA_weak'
    def __init__(self, parameters=SSAweakEquationParamter()):
        super().__init__(parameters)
    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ variational (weak-form) physics residual for SSA
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        Cid = self.local_output_var["C"]

        # test function: normalised gaussian kernel
        x = slice_column(nn_input_var, xid)
        y = slice_column(nn_input_var, yid)

        tf1 = bkd.sin(x) + bkd.cos(y)
        tf2 = bkd.sin(y) + bkd.cos(x)

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        tf1_x = jacobian(tf1, nn_input_var, i=0, j=xid)
        tf1_y = jacobian(tf1, nn_input_var, i=0, j=yid)
        tf2_x = jacobian(tf2, nn_input_var, i=0, j=xid)
        tf2_y = jacobian(tf2, nn_input_var, i=0, j=yid)
        

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        H = slice_column(nn_output_var, Hid)
        C = slice_column(nn_output_var, Cid)

        eta = 0.5*self.B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+self.eps)**(0.5*(1.0-self.n)/self.n)
        etaH = eta * H

        # compute the basal stress
        u_norm = (u**2+v**2+self.eps**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        VISC1 = 2*etaH * ( (2*u_x+v_y)*tf1_x + (u_x+v_y)*tf2_y +0.5*(u_y+v_x)*(tf1_y+tf2_x) )
        FRIC1 = (tf1*alpha*u/(u_norm) + tf2*alpha*v/(u_norm))
        GRAV1 = self.rhoi*self.g*H * (tf1*s_x + tf2*s_y)

        VISC2 = 2*etaH * ( (2*u_x+v_y)*tf2_x + (u_x+v_y)*tf1_y +0.5*(u_y+v_x)*(tf2_y+tf1_x) )
        FRIC2 = (tf2*alpha*u/(u_norm) + tf1*alpha*v/(u_norm))
        GRAV2 = self.rhoi*self.g*H * (tf2*s_x + tf1*s_y)

        f1 = VISC1 + FRIC1 + GRAV1
        f2 = VISC2 + FRIC2 + GRAV2

        return [f1,f2] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ 
        """
        pass
#}}}

### SSA action potential for PINN:
class SSAactionEquationParameter(EquationParameter, Constants):
    """ default parameters for SSA_action
    """
    _EQUATION_TYPE = 'SSA_action' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 's', 'H', 'C']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6, 1.0e-8]
        self.residuals = ["f"+self._EQUATION_TYPE+"1"]
        self.pde_weights = [1.0e-11]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08   # -8 degree C, cuffey
                }
class SSA_action(EquationBase): #{{{
    """ SSA action principle on 2D problem with uniform B
    """
    _EQUATION_TYPE = 'SSA_action' 
    def __init__(self, parameters=SSAactionEquationParameter()):
        super().__init__(parameters)
    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of SSA 2D PDEs

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        Cid = self.local_output_var["C"]

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        H = slice_column(nn_output_var, Hid)
        C = slice_column(nn_output_var, Cid)

        sr_eff = (u_x**2 + v_y**2 + 0.25*(u_y+v_x)**2 + (u_x*v_y) + self.eps)**0.5

        u_norm = (u**2+v**2+self.eps**2)**0.5

        VISC = 2*self.n/(self.n+1) * H * self.B * sr_eff**((1/self.n)+1)
        GRAV = self.rhoi * self.g * H * (s_x*u + s_y*v)
        FRIC = self.n/(self.n+1) * C * u_norm**((1/self.n)+1)

        f1 = VISC + FRIC + GRAV

        return [f1] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ don't worry, be happy
        """
        pass
    #}}}
#}}}

