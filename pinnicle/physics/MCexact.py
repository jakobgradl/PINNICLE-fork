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

