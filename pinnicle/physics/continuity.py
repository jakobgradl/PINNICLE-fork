import deepxde as dde
import deepxde.backend as bkd
from . import EquationBase, Constants
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian

# static mass conservation {{{
class MCEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'MC' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 'a', 'H']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2, 1.0e-8*self.yts**2, 1.0*self.yts**2, 1.0e-6]
        self.residuals = ["f"+self._EQUATION_TYPE]
        self.pde_weights = [1.0e10]

        # scalar variables: name:value
        self.scalar_variables = {}
class MC(EquationBase): #{{{
    """ MC on 2D problem
    """
    _EQUATION_TYPE = 'MC' 
    def __init__(self, parameters=MCEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        aid = self.local_output_var["a"]
        Hid = self.local_output_var["H"]

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        a = slice_column(nn_output_var, aid)
        H = slice_column(nn_output_var, Hid)

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        H_x = jacobian(nn_output_var, nn_input_var, i=Hid, j=xid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        H_y = jacobian(nn_output_var, nn_input_var, i=Hid, j=yid)

        # residual
        f = H*u_x + H_x*u + H*v_y + H_y*v - a

        return [f] #}}}
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}

# Steady-state mass conservation (a=0) {{{
class MCsteadyEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'MC_steady' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 'H']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2, 1.0e-8*self.yts**2, 1.0e-6]
        self.residuals = ["f"+self._EQUATION_TYPE]
        self.pde_weights = [1.0e10]

        # scalar variables: name:value
        self.scalar_variables = {}
class MC_steady(EquationBase): #{{{
    """ MC on 2D problem with negligible smb and dH/dt
    """
    _EQUATION_TYPE = 'MC_steady' 
    def __init__(self, parameters=MCEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        Hid = self.local_output_var["H"]

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        H = slice_column(nn_output_var, Hid)

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        H_x = jacobian(nn_output_var, nn_input_var, i=Hid, j=xid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        H_y = jacobian(nn_output_var, nn_input_var, i=Hid, j=yid)

        # residual
        f = H*u_x + H_x*u + H*v_y + H_y*v

        return [f] #}}}
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}





# time dependent mass conservation {{{
class ThicknessEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'Mass transport'
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y', 't']
        self.output = ['u', 'v', 'a', 'H']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2, 1.0e-8*self.yts**2, 1.0e-2*self.yts**2, 1.0e-6]
        self.residuals = ["f"+self._EQUATION_TYPE]
        self.pde_weights = [1.0e10]

        # scalar variables: name:value
        self.scalar_variables = {
                }
class Thickness(EquationBase): #{{{
    """ 2D time depenent thickness evolution
    """
    _EQUATION_TYPE = 'Mass transport'
    def __init__(self, parameters=ThicknessEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of 2D thickness evolution PDE

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]
        tid = self.local_input_var["t"]

        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        aid = self.local_output_var["a"]
        Hid = self.local_output_var["H"]

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        a = slice_column(nn_output_var, aid)
        H = slice_column(nn_output_var, Hid)

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        H_x = jacobian(nn_output_var, nn_input_var, i=Hid, j=xid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        H_y = jacobian(nn_output_var, nn_input_var, i=Hid, j=yid)

        # temporal derivative
        H_t = jacobian(nn_output_var, nn_input_var, i=Hid, j=tid)

        # residual
        f = H_t + H*u_x + H_x*u + H*v_y + H_y*v - a

        return [f] #}}}
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}




# D-HNN exact mass conservation {{{
class MCSSAEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'MC_SSA' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['D_smb','D_dH','R', 'H']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0, 1.0, 1.0, 1.0e-3]
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {}
class MC_SSA(EquationBase): #{{{
    """ MC on 2D problem

        for domains with negligible smb and dH/dt

        u,v,a are defined based on two scalar fields D,R
        in a way that automatically satisfies the MC
    """
    _EQUATION_TYPE = 'MC_SSA' 
    def __init__(self, parameters=MCSSAEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ no pde loss required
            use data losses vel_mag_MC, u_MC, v_MC, a_MC

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        Hid = self.local_output_var["H"]
        H = slice_column(nn_output_var, Hid)

        return [] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}




# D-HNN exact mass conservation {{{
class MCSSASteadyEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'MC_SSA_steady' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['R', 'H']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0, 1.0e-3]
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {}
class MC_SSA_steady(EquationBase): #{{{
    """ MC on 2D problem

        for domains with negligible smb and dH/dt

        u,v,a are defined based on two scalar fields D,R
        in a way that automatically satisfies the MC
    """
    _EQUATION_TYPE = 'MC_SSA_steady' 
    def __init__(self, parameters=MCSSASteadyEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ no pde loss required
            use data losses vel_mag_MC, u_MC, v_MC, a_MC

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return [] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}



# D-HNN exact mass conservation resolving vertical velocity profile through MOLHO {{{
class MCMOLHOEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'MC_MOLHO' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['D_smb', 'D_dH', 'R', 'H', 'p']#, 'n']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0]*3 + [1.0e-3, 1.0]#, 1.0]
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {'n': 3.0,
                                 'vub': 200.0/self.yts,
                                 'vlb': 25.0/self.yts,
                                 }
        # self.scalar_variables = {}
class MC_MOLHO(EquationBase): #{{{
    """ MC on 2D problem

        u,v,a are defined based on two scalar fields D,R
        in a way that automatically satisfies the MC

        p describes the relative contributions of basal and shear velocities
        to the surface velocity 

        surface velocity is derived from the depth-averaged velocity according to MOLHO

        vub, vlb are upper and lower velocity bounds for BCs on p
    """
    _EQUATION_TYPE = 'MC_MOLHO' 
    def __init__(self, parameters=MCMOLHOEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ no pde loss required for mass conservation
            use data losses vel_mag_MC, u_MC, v_MC, a_MC

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return [] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}


# D-HNN exact mass conservation resolving vertical velocity profile through MOLHO {{{
class REGUPEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'REGU_P' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['D_smb', 'D_dH', 'R', 'H', 'p']#, 'n']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0]*3 + [1.0e-3, 1.0]#, 1.0]
        self.residuals = ["f"+self._EQUATION_TYPE]
        self.pde_weights = [1.0e0]

        # scalar variables: name:value
        self.scalar_variables = {}
class REGU_P(EquationBase): #{{{
    """ Regularisation for MC_MOLHO
        applies regularisation on along-flow gradient of p (non-negative)
    """
    _EQUATION_TYPE = 'REGU_P' 
    def __init__(self, parameters=REGUPEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ pde loss is used as regularisation on p
            (penalise when p decreases along flow)
            
        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        Dsmbid = self.local_output_var["D_smb"]
        DdHid = self.local_output_var["D_dH"]
        Rid = self.local_output_var["R"]
        pid = self.local_output_var["p"]
        Hid = self.local_output_var["H"]

        H = slice_column(nn_output_var, Hid)

        p1 = slice_column(nn_output_var, pid)
        p = bkd.sigmoid(p1) # p in [0,1]

        u_mf = (jacobian(nn_output_var,nn_input_var,i=Dsmbid,j=xid) + 
                jacobian(nn_output_var,nn_input_var,i=DdHid,j=xid) - 
                jacobian(nn_output_var,nn_input_var,i=Rid,j=yid))
        
        v_mf = (jacobian(nn_output_var,nn_input_var,i=Dsmbid,j=yid) + 
                jacobian(nn_output_var,nn_input_var,i=DdHid,j=yid) - 
                jacobian(nn_output_var,nn_input_var,i=Rid,j=xid))
        
        u_bar = u_mf / H
        v_bar = v_mf / H
        
        u_mag = (u_bar**2 + v_bar**2)**0.5
        
        p_x = jacobian(p, nn_input_var, i=0, j=xid)
        p_y = jacobian(p, nn_input_var, i=0, j=yid)

        # directional derivative of p in direction of velocity vector
        p_dirdev = p_x*(u_mf/u_mag) + p_y*(v_mf/u_mag)

        # return loss if p_dirdev is negative, zero loss otherwise
        f = bkd.relu(-1 * p_dirdev)

        return [f] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}


# D-HNN exact mass conservation resolving vertical velocity profile through MOLHO {{{
class MCMOLHOESteadyquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'MC_MOLHO_steady' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        # self.output = ['D_smb','D_dH','R', 'H']
        self.output = ['R', 'H', 'p']#, 'n']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        # self.data_weights = [1.0, 1.0, 1.0, 1.0e-6]
        self.data_weights = [1.0]*1 + [1.0e-3, 1.0]#, 1.0]
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {'n': 3.0}
        # self.scalar_variables = {}
class MC_MOLHO_steady(EquationBase): #{{{
    """ MC on 2D problem

        u,v,a are defined based on two scalar fields D,R
        in a way that automatically satisfies the MC

        p describes the relative contributions of basal and shear velocities
        to the surface velocity 

        surface velocity is derived from the depth-averaged velocity according to MOLHO
    """
    _EQUATION_TYPE = 'MC_MOLHO_steady' 
    def __init__(self, parameters=MCMOLHOESteadyquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ no pde loss required
            use data losses vel_mag_MC, u_MC, v_MC, a_MC

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        
        return [] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}



# mass conservation with vertical shear {{{
class MC4MOLHOEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation with vertical shear
    """
    _EQUATION_TYPE = 'MC4MOLHO'
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u', 'v', 'u_base', 'v_base', 'a', 'H']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2, 1.0e-8*self.yts**2, 1.0e-8*self.yts**2, 1.0e-8*self.yts**2, 1.0*self.yts**2, 1.0e-6]
        self.residuals = ["f"+self._EQUATION_TYPE]
        self.pde_weights = [1.0e10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n':3.0
                }
class MC4MOLHO(EquationBase): #{{{
    """ MC include vertical shear
    """
    _EQUATION_TYPE = 'MC4MOLHO'
    def __init__(self, parameters=MC4MOLHOEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC PDE, adapted to include vertical shear

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        uid = self.local_output_var["u"]
        vid = self.local_output_var["v"]
        ubid = self.local_output_var["u_base"]
        vbid = self.local_output_var["v_base"]
        aid = self.local_output_var["a"]
        Hid = self.local_output_var["H"]

        # unpacking normalized output
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid)
        ub = slice_column(nn_output_var, ubid)
        vb = slice_column(nn_output_var, vbid)
        a = slice_column(nn_output_var, aid)
        H = slice_column(nn_output_var, Hid)

        # depth-averaged velocities
        ubar = ub + (u - ub)*((self.n+1)/(self.n+2))
        vbar = vb + (v - vb)*((self.n+1)/(self.n+2))

        # calculate ice flux
        Hu = H*ubar
        Hv = H*vbar

       # spatial derivatives
        Dx = jacobian(Hu, nn_input_var, i=0, j=xid)
        Dy = jacobian(Hv, nn_input_var, i=0, j=yid)

        # residual
        f = Dx + Dy - a

        return [f] #}}}
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC PDE with vertical shear, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}
