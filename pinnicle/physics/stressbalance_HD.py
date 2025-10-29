import deepxde as dde
from deepxde.backend import jax, abs
from . import EquationBase, Constants
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian, slice_function_jax
# SSA constant B {{{
class SSAHDEquationParameter(EquationParameter, Constants):
    """ default parameters for SSA_HD
    """
    _EQUATION_TYPE = 'SSA_HD' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['D', 'R', 's', 'H', 'C']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-6, 1.0e-6, 1.0e-8]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08   # -8 degree C, cuffey
                }
class SSA(EquationBase): #{{{
    """ SSA on 2D problem with uniform B
    """
    _EQUATION_TYPE = 'SSA_HD' 
    def __init__(self, parameters=SSAHDEquationParameter()):
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
        aid = self.local_output_var["a"]
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

        eta = 0.5*self.B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+self.eps)**(0.5*(1.0-self.n)/self.n)
        # stress tensor
        etaH = eta * H
        B11 = etaH*(4*u_x + 2*v_y)
        B22 = etaH*(4*v_y + 2*u_x)
        B12 = etaH*(  u_y +   v_x)

        # Getting the other derivatives
        sigma11 = jacobian(B11, nn_input_var, i=0, j=xid)
        sigma12 = jacobian(B12, nn_input_var, i=0, j=yid)

        sigma21 = jacobian(B12, nn_input_var, i=0, j=xid)
        sigma22 = jacobian(B22, nn_input_var, i=0, j=yid)


        # compute the basal stress
        u_norm = (u**2+v**2+self.eps**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        f1 = sigma11 + sigma12 - alpha*u/(u_norm) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm) - self.rhoi*self.g*H*s_y

        return [f1, f2] #}}}
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
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

        # get the spatial derivatives functions
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid, val=1)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid, val=1)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid, val=1)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid, val=1)

        # get variable function
        H_func = lambda x: slice_function_jax(nn_output_var, x, Hid)
        # stress tensor
        etaH = lambda x: 0.5*H_func(x)*self.B *(u_x(x)**2.0 + v_y(x)**2.0 + 0.25*(u_y(x)+v_x(x))**2.0 + u_x(x)*v_y(x)+self.eps)**(0.5*(1.0-self.n)/self.n)

        B11 = lambda x: etaH(x)*(4*u_x(x) + 2*v_y(x))
        B22 = lambda x: etaH(x)*(4*v_y(x) + 2*u_x(x))
        B12 = lambda x: etaH(x)*(  u_y(x) +   v_x(x))

        # Getting the other derivatives
        sigma11 = jacobian((jax.vmap(B11)(nn_input_var), B11), nn_input_var, i=0, j=xid)
        sigma12 = jacobian((jax.vmap(B12)(nn_input_var), B12), nn_input_var, i=0, j=yid)

        sigma21 = jacobian((jax.vmap(B12)(nn_input_var), B12), nn_input_var, i=0, j=xid)
        sigma22 = jacobian((jax.vmap(B22)(nn_input_var), B22), nn_input_var, i=0, j=yid)

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        H = slice_column(nn_output_var, Hid)
        C = slice_column(nn_output_var, Cid)
        # compute the basal stress
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        u_norm = (u**2+v**2+self.eps**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        f1 = sigma11 + sigma12 - alpha*u/(u_norm) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*v/(u_norm) - self.rhoi*self.g*H*s_y

        return [f1, f2] #}}}
    #}}}
#}}}