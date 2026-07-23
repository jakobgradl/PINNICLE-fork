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



        a = 1
        # b = 2
        # c = 8
        # d = 32

        # tf1 = [None]*1 #4
        # tf2 = [None]*1 #4

        # tf1[0] = bkd.sin((1/a)*(x-a+1)) + bkd.cos((1/(a*2))*(y-(a*2)+1))
        # tf1[1] = bkd.sin((1/b)*(x-b+1)) + bkd.cos((1/(b*2))*(y-(b*2)+1))
        # tf1[2] = bkd.sin((1/c)*(x-c+1)) + bkd.cos((1/(c*2))*(y-(c*2)+1))
        # tf1[3] = bkd.sin((1/d)*(x-d+1)) + bkd.cos((1/(d*2))*(y-(d*2)+1))

        # tf2[0] = bkd.sin((1/a)*(y-a+1)) + bkd.cos((1/(a*2))*(x-(a*2)+1))
        # tf2[1] = bkd.sin((1/b)*(y-b+1)) + bkd.cos((1/(b*2))*(x-(b*2)+1))
        # tf2[2] = bkd.sin((1/c)*(y-c+1)) + bkd.cos((1/(c*2))*(x-(c*2)+1))
        # tf2[3] = bkd.sin((1/d)*(y-d+1)) + bkd.cos((1/(d*2))*(x-(d*2)+1))

        tf1 = bkd.sin(x) + bkd.cos(y)
        tf2 = bkd.sin(y) + bkd.cos(x)

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        # tf1_x = [None]*1 #4
        # tf1_y = [None]*1 #4
        # tf2_x = [None]*1 #4
        # tf2_y = [None]*1 #4

        # for i in range(4):
        # i = 0
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

        # VISC1 = 0
        # VISC2 = 0
        # FRIC1 = 0
        # FRIC2 = 0
        # GRAV1 = 0
        # GRAV2 = 0

        # # for i in range(4):
        # VISC1 += 2*etaH * ( (2*u_x+v_y)*tf1_x[i] + (u_x+v_y)*tf2_y[i] +0.5*(u_y+v_x)*(tf1_y[i]+tf2_x[i]) )
        # FRIC1 += (tf1[i]*alpha*u/(u_norm) + tf2[i]*alpha*v/(u_norm))
        # GRAV1 += self.rhoi*self.g*H * (tf1[i]*s_x + tf2[i]*s_y)

        # VISC2 += 2*etaH * ( (2*u_x+v_y)*tf2_x[i] + (u_x+v_y)*tf1_y[i] +0.5*(u_y+v_x)*(tf2_y[i]+tf1_x[i]) )
        # FRIC2 += (tf2[i]*alpha*u/(u_norm) + tf1[i]*alpha*v/(u_norm))
        # GRAV2 += self.rhoi*self.g*H * (tf2[i]*s_x + tf1[i]*s_y)

        VISC1 = 2*etaH*(2*u_x+v_y)*tf1_x + etaH*(u_y+v_x)*tf1_y
        FRIC1 = tf1 * alpha*u/(u_norm)
        GRAV1 = self.rhoi*self.g*H*s_x*tf1

        VISC2 = 2*etaH*(u_x+2*v_y)*tf2_y + etaH*(u_y+v_x)*tf2_x
        FRIC2 = tf2 * alpha*v/(u_norm)
        GRAV2 = self.rhoi*self.g*H*s_y*tf2

        f1 = VISC1 + FRIC1 + GRAV1
        f2 = VISC2 + FRIC2 + GRAV2

        return [f1,f2] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ 
        """
        pass
#}}}

class SSAvarBweakEquationParamter(EquationParameter, Constants):
    """default parameters for SSA_weak
    """
    _EQUATION_TYPE = 'SSAvarB_weak'
    def __init__(self, param_dict={}):
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 's', 'H', 'C', 'B']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.output_lb[5] = 0.
        self.output_ub[5] = 3.
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6, 1.0e-8, 1.0e-16]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1e0,1e0]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                # 'B':1.26802073401e+08,   # -8 degree C, cuffey
                'm': 3, # exponent of the Weertman friction law
                'rho':917,
                'g':9.81,
                }
class SSAvarB_weak(EquationBase): #{{{
    """ variational (weak-form) SSA on 2D problem with uniform B
    """
    _EQUATION_TYPE = 'SSAvarB_weak'
    def __init__(self, parameters=SSAvarBweakEquationParamter()):
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
        Bid = self.local_output_var["B"]

        # test function: normalised gaussian kernel
        x = slice_column(nn_input_var, xid)
        y = slice_column(nn_input_var, yid)



        a = 1
        # b = 2
        # c = 8
        # d = 32

        # tf1 = [None]*1 #4
        # tf2 = [None]*1 #4

        # tf1[0] = bkd.sin((1/a)*(x-a+1)) + bkd.cos((1/(a*2))*(y-(a*2)+1))
        # tf1[1] = bkd.sin((1/b)*(x-b+1)) + bkd.cos((1/(b*2))*(y-(b*2)+1))
        # tf1[2] = bkd.sin((1/c)*(x-c+1)) + bkd.cos((1/(c*2))*(y-(c*2)+1))
        # tf1[3] = bkd.sin((1/d)*(x-d+1)) + bkd.cos((1/(d*2))*(y-(d*2)+1))

        # tf2[0] = bkd.sin((1/a)*(y-a+1)) + bkd.cos((1/(a*2))*(x-(a*2)+1))
        # tf2[1] = bkd.sin((1/b)*(y-b+1)) + bkd.cos((1/(b*2))*(x-(b*2)+1))
        # tf2[2] = bkd.sin((1/c)*(y-c+1)) + bkd.cos((1/(c*2))*(x-(c*2)+1))
        # tf2[3] = bkd.sin((1/d)*(y-d+1)) + bkd.cos((1/(d*2))*(x-(d*2)+1))

        tf1 = bkd.sin(x) + bkd.cos(y)
        tf2 = bkd.sin(y) + bkd.cos(x)

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        # tf1_x = [None]*1 #4
        # tf1_y = [None]*1 #4
        # tf2_x = [None]*1 #4
        # tf2_y = [None]*1 #4

        # for i in range(4):
        # i = 0
        tf1_x = jacobian(tf1, nn_input_var, i=0, j=xid)
        tf1_y = jacobian(tf1, nn_input_var, i=0, j=yid)
        tf2_x = jacobian(tf2, nn_input_var, i=0, j=xid)
        tf2_y = jacobian(tf2, nn_input_var, i=0, j=yid)
        
        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        H = slice_column(nn_output_var, Hid)
        C = slice_column(nn_output_var, Cid)
        Bfac = slice_column(nn_output_var, Bid)

        B = 7.469e7 + 7.469e7 * Bfac**2

        eta = 0.5*B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+self.eps)**(0.5*(1.0-self.n)/self.n)
        etaH = eta * H

        # compute the basal stress
        u_norm = (u**2+v**2+self.eps**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        # VISC1 = 0
        # VISC2 = 0
        # FRIC1 = 0
        # FRIC2 = 0
        # GRAV1 = 0
        # GRAV2 = 0

        # # for i in range(4):
        # VISC1 += 2*etaH * ( (2*u_x+v_y)*tf1_x[i] + (u_x+v_y)*tf2_y[i] +0.5*(u_y+v_x)*(tf1_y[i]+tf2_x[i]) )
        # FRIC1 += (tf1[i]*alpha*u/(u_norm) + tf2[i]*alpha*v/(u_norm))
        # GRAV1 += self.rhoi*self.g*H * (tf1[i]*s_x + tf2[i]*s_y)

        # VISC2 += 2*etaH * ( (2*u_x+v_y)*tf2_x[i] + (u_x+v_y)*tf1_y[i] +0.5*(u_y+v_x)*(tf2_y[i]+tf1_x[i]) )
        # FRIC2 += (tf2[i]*alpha*u/(u_norm) + tf1[i]*alpha*v/(u_norm))
        # GRAV2 += self.rhoi*self.g*H * (tf2[i]*s_x + tf1[i]*s_y)

        VISC1 = 2*etaH*(2*u_x+v_y)*tf1_x + etaH*(u_y+v_x)*tf1_y
        FRIC1 = tf1 * alpha*u/(u_norm)
        GRAV1 = self.rhoi*self.g*H*s_x*tf1

        VISC2 = 2*etaH*(u_x+2*v_y)*tf2_y + etaH*(u_y+v_x)*tf2_x
        FRIC2 = tf2 * alpha*v/(u_norm)
        GRAV2 = self.rhoi*self.g*H*s_y*tf2

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

