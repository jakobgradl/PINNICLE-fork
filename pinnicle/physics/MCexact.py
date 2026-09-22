import deepxde as dde
import deepxde.backend as bkd
from ..parameter import PhysicsParameter
# from . import EquationBase
from . import EquationBase, Constants, MC_EXACT
from ..parameter import EquationParameter
import itertools
from ..utils import slice_column, jacobian, ppow, default_float_type
import torch

###################################
### primary function classes

# function for exact solution of mass conservation equation {{{
class MCexactEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'MC_exact' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['Q_x', 'Q_y', 'Hexp']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0]*2 + [1.0e-3] 
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {'n': 3.0,
                                 'p': 0.,
                                 'vub': 200.0/self.yts,
                                 'vlb': 30.0/self.yts,
                                 'nlb': 1.,
                                 'nub': 100.,
                                 'B':1.26802073401e+08,   # -8 degree C, cuffey
                                'm': 3, # exponent of the Weertman friction law
                                'rho':917,
                                'g':9.81,
                                 }
        # self.scalar_variables = {}
class MC_exact(EquationBase): #{{{
    """ MC on 2D problem

        u,v are derived from mass flux Q as Q/H
        a is derived as the residual of the MC equation

        this way, all variables are consistent with the MC equation (exact solution)
    """
    _EQUATION_TYPE = 'MC_exact' 
    def __init__(self, parameters=MCexactEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ no pde loss required for mass conservation
            use data losses u_MC, v_MC, a_MC
        """
        return [] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ 
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}

# function for exact solution of mass conservation equation via Helmholtz decomposition of mass flux vector field {{{
class MCexactHelmholtzEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'MC_exact_Helmholtz' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['D_smb', 'R', 'Hexp']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0]*2 + [1.0e-3] 
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {'n': 3.0,
                                 'p': 0.,
                                 'vub': 200.0/self.yts,
                                 'vlb': 30.0/self.yts,
                                 'nlb': 1.,
                                 'nub': 100.,
                                 'B':1.26802073401e+08,   # -8 degree C, cuffey
                                'm': 3, # exponent of the Weertman friction law
                                'rho':917,
                                'g':9.81,
                                 }
        # self.scalar_variables = {}
class MC_exact_Helmholtz(EquationBase): #{{{
    """ MC on 2D problem

        Helmholtz decompostion of the mass flux vector field into a dissipative field d and a rotational field r
        d,r are defined as the gradient and co-gradient respectively of the scalar potential fields D,R
        mass flux Q is defined as d+r
        u,v are derived from mass flux as Q/H
        a is equal to the divergence of Q and derived from D
    """
    _EQUATION_TYPE = 'MC_exact_Helmholtz' 
    def __init__(self, parameters=MCexactHelmholtzEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ no pde loss required for mass conservation
            use data losses u_MC, v_MC, a_MC
        """
        return [] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ 
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}


##################################
### additional parameters

# plug-flow with parameterised deformation; u_bar = k*u_surf
class ParameterisedDeformationEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'Parameterised_Deformation' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['ubar_ratio']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0]
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {}
class ParameterisedDeformation(EquationBase): #{{{
    """ add sliding parameter p to the output 
    """
    _EQUATION_TYPE = 'Parameterised_Deformation' 
    def __init__(self, parameters=ParameterisedDeformationEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        return [] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}

# sliding parameter for Lliboutry model
class LliboutrySlidingEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'Lliboutry_sliding' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['p']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0]
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {'n': 3.0}
class Lliboutry_sliding(EquationBase): #{{{
    """ add sliding parameter p to the output 
    """
    _EQUATION_TYPE = 'Lliboutry_sliding' 
    def __init__(self, parameters=LliboutrySlidingEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        return [] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}

# deformation parameter for Lliboutry model
class LliboutryShearEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'Lliboutry_shear' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['n']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0]
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {'p': 0.}
class Lliboutry_shear(EquationBase): #{{{
    """ add shear parameter n to the output 
    """
    _EQUATION_TYPE = 'Lliboutry_shear' 
    def __init__(self, parameters=LliboutryShearEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        return [] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}

# additional dissipative field to separate smb and dHdt
class DdHEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'DdH' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['D_dH']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0]
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {}
class DdH(EquationBase): #{{{
    """ add second dissipative potential for dHdt to the output
        this is for non-transient inversions 
    """
    _EQUATION_TYPE = 'DdH' 
    def __init__(self, parameters=DdHEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        return [] #}}}
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}

# add output variables for SSA
class SSAsCEquationParamter(EquationParameter, Constants):
    """default parameters for SSA_exact
    """
    _EQUATION_TYPE = 'SSA_sC'
    def __init__(self, param_dict={}):
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        # self.output = ['b','C']
        self.output = ['s','C']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-3] + [1.0]*1
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08,   # -8 degree C, cuffey
                'm': 3, # exponent of the Weertman friction law
                'rho':917,
                'g':9.81,
                }     
class SSA_sC(EquationBase): #{{{
    """ SSA with weak-form pde loss
    """
    _EQUATION_TYPE = 'SSA_sC'
    def __init__(self, parameters=SSAsCEquationParamter()):
        super().__init__(parameters)
    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ no pde loss required
        """
        return [] 
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ 
        """
        pass
#}}}

class SSAsCBEquationParamter(EquationParameter, Constants):
    """default parameters for SSA_exact
    """
    _EQUATION_TYPE = 'SSA_sCB'
    def __init__(self, param_dict={}):
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        # self.output = ['b','C','B']
        self.output = ['s','C','B']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        # self.output_lb[2] = 7.
        # self.output_ub[2] = 8.
        # self.output_lb[2] = -10.
        # self.output_ub[2] = 0.
        # self.output_lb[2] = 0.
        # self.output_ub[2] = 3.
        self.output_lb[2] = 1e3 # 10.
        self.output_ub[2] = 1e4 # 100.
        self.data_weights = [1.0e-3] + [1.0]*2
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'm': 3, # exponent of the Weertman friction law
                'rho':917,
                'g':9.81,
                'Bmin':7.469e7, # 0 degree C, cuffey
                }     
class SSA_sCB(EquationBase): #{{{
    """ SSA with weak-form pde loss
    """
    _EQUATION_TYPE = 'SSA_sCB'
    def __init__(self, parameters=SSAsCBEquationParamter()):
        super().__init__(parameters)
    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ no pde loss required
        """
        return [] 
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ 
        """
        pass
#}}}

class SSAsBEquationParamter(EquationParameter, Constants):
    """default parameters for SSA_exact
    """
    _EQUATION_TYPE = 'SSA_sB'
    def __init__(self, param_dict={}):
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        # self.output = ['b','C','B']
        self.output = ['s','B']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        # self.output_lb[2] = 7.
        # self.output_ub[2] = 8.
        # self.output_lb[2] = -10.
        # self.output_ub[2] = 0.
        # self.output_lb[2] = 0.
        # self.output_ub[2] = 3.
        self.output_lb[1] = 1e3 # 10.
        self.output_ub[1] = 1e4 # 100.
        self.data_weights = [1.0e-3] + [1.0]
        self.residuals = []
        self.pde_weights = []

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'm': 3, # exponent of the Weertman friction law
                'rho':917,
                'g':9.81,
                'Bmin':7.469e7, # 0 degree C, cuffey
                }     
class SSA_sB(EquationBase): #{{{
    """ SSA with weak-form pde loss
    """
    _EQUATION_TYPE = 'SSA_sB'
    def __init__(self, parameters=SSAsBEquationParamter()):
        super().__init__(parameters)
    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ no pde loss required
        """
        return [] 
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ 
        """
        pass
#}}}

####################################
### regularisation functions

# regularise gradient of sliding parameter in Lliboutry model
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
        self.output = ['D_smb', 'R', 'H', 'p']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.output_lb[2] = -2.
        self.output_ub[2] = 8.5
        self.data_weights = [1.0]*2 + [1.0e-3] + [1.0]
        self.residuals = ["f"+self._EQUATION_TYPE]
        self.pde_weights = [1.0e12]

        # scalar variables: name:value
        self.scalar_variables = {}
class REGU_P(EquationBase,MC_EXACT): #{{{
    """ Regularisation for Lliboutry_Sliding
        applies regularisation on along-flow gradient of p
        gradient must be non-negative (p cannot decrease along flow)
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

        # Dsmbid = self.local_output_var["D_smb"]
        # # DdHid = self.local_output_var["D_dH"]
        # Rid = self.local_output_var["R"]
        # pid = self.local_output_var["p"]
        # Hid = self.local_output_var["H"]

        # H = slice_column(nn_output_var, Hid)

        # p1 = slice_column(nn_output_var, pid)
        # p = bkd.sigmoid(p1) # p in [0,1]

        # u_mf = (jacobian(nn_output_var,nn_input_var,i=Dsmbid,j=xid)
        #         # + jacobian(nn_output_var,nn_input_var,i=DdHid,j=xid)
        #         - jacobian(nn_output_var,nn_input_var,i=Rid,j=yid))
        
        # v_mf = (jacobian(nn_output_var,nn_input_var,i=Dsmbid,j=yid)
        #         # + jacobian(nn_output_var,nn_input_var,i=DdHid,j=yid)
        #         + jacobian(nn_output_var,nn_input_var,i=Rid,j=xid))
        
        # u_bar = u_mf / H
        # v_bar = v_mf / H
        
        u_bar = self.Hu_to_ubar(nn_input_var,nn_output_var)
        v_bar = self.Hv_to_vbar(nn_input_var,nn_output_var)
        p = self.get_p(nn_input_var,nn_output_var)

        u_mag = (u_bar**2 + v_bar**2 + (1e-15)**2)**0.5
        
        p_x = jacobian(p, nn_input_var, i=0, j=xid)
        p_y = jacobian(p, nn_input_var, i=0, j=yid)

        # directional derivative of p in direction of velocity vector
        p_dirdev = p_x*(u_bar/u_mag) + p_y*(v_bar/u_mag)

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

# regularise gradient of deformation parameter in Lliboutry model
class REGUNEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'REGU_N' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['D_smb', 'R', 'H', 'n']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.output_lb[2] = -2.
        self.output_ub[2] = 8.5
        self.data_weights = [1.0]*2 + [1.0e-3] + [1.0]
        self.residuals = ["f"+self._EQUATION_TYPE]
        self.pde_weights = [1.0e12]

        # scalar variables: name:value
        self.scalar_variables = {}
class REGU_N(EquationBase,MC_EXACT): #{{{
    """ Regularisation for Lliboutry_Shear
        applies regularisation on along-flow gradient of n
        gradient must be non-negative (n cannot decrease along flow)
    """
    _EQUATION_TYPE = 'REGU_N' 
    def __init__(self, parameters=REGUNEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ pde loss is used as regularisation on n
            (penalise when n decreases along flow)
            
        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        # Dsmbid = self.local_output_var["D_smb"]
        # # DdHid = self.local_output_var["D_dH"]
        # Rid = self.local_output_var["R"]
        # nid = self.local_output_var["n"]
        # Hid = self.local_output_var["H"]

        # H = slice_column(nn_output_var, Hid)

        # n1 = slice_column(nn_output_var, nid)
        # a = 3 
        # b = 18 
        # n = (b-a) * bkd.sigmoid(n1) + a

        # u_mf = (jacobian(nn_output_var,nn_input_var,i=Dsmbid,j=xid) 
        #         # + jacobian(nn_output_var,nn_input_var,i=DdHid,j=xid)
        #         - jacobian(nn_output_var,nn_input_var,i=Rid,j=yid))
        
        # v_mf = (jacobian(nn_output_var,nn_input_var,i=Dsmbid,j=yid)
        #         # + jacobian(nn_output_var,nn_input_var,i=DdHid,j=yid)
        #         + jacobian(nn_output_var,nn_input_var,i=Rid,j=xid))
        
        # u_bar = u_mf / H
        # v_bar = v_mf / H

        u_bar = self.Hu_to_ubar(nn_input_var,nn_output_var)
        v_bar = self.Hv_to_vbar(nn_input_var,nn_output_var)
        n = self.get_n(nn_input_var,nn_output_var)
        
        u_mag = (u_bar**2 + v_bar**2 + (1e-15)**2)**0.5
        
        n_x = jacobian(n, nn_input_var, i=0, j=xid)
        n_y = jacobian(n, nn_input_var, i=0, j=yid)

        # directional derivative of n in direction of velocity vector
        n_dirdev = n_x*(u_bar/u_mag) + n_y*(v_bar/u_mag)

        # return loss if n_dirdev is negative, zero loss otherwise
        f = bkd.relu(-1 * n_dirdev)

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

# regularise magnitude of deformation parameter in Lliboutry model
class REGUNmagEquationParameter(EquationParameter, Constants):
    """ default parameters for mass conservation
    """
    _EQUATION_TYPE = 'REGU_Nmag' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['n']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0]
        self.residuals = ["f"+self._EQUATION_TYPE]
        self.pde_weights = [1.0e-4]

        # scalar variables: name:value
        self.scalar_variables = {}
class REGU_Nmag(EquationBase): #{{{
    """ Regularisation for Lliboutry_Shear
        applies regularisation on n
        prefer small n
    """
    _EQUATION_TYPE = 'REGU_Nmag' 
    def __init__(self, parameters=REGUNmagEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ 
        """
        nid = self.local_output_var["n"]
        
        n1 = slice_column(nn_output_var, nid)
        a = 3 
        b = 18 
        n = (b-a) * bkd.sigmoid(n1) + a

        f = n

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

