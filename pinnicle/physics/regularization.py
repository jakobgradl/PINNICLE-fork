import deepxde as dde
from deepxde.backend import jax, abs
from . import EquationBase, Constants
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian, slice_function_jax

# {{{
class ReguRangaCBEquationParameter(EquationParameter, Constants):
    """ default parameters for ReguRangeCB regularization 
    """
    _EQUATION_TYPE = 'ReguRangaCB' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['u','v','C','B']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0e-8*self.yts**2.0, 1.0e-8*self.yts**2.0, 1.0e-8, 1e-16]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2",
                          "f"+self._EQUATION_TYPE+"3", "f"+self._EQUATION_TYPE+"4",
                          "f"+self._EQUATION_TYPE+"5", "f"+self._EQUATION_TYPE+"6"]
        self.pde_weights = [1.0e2]+ 2*[1.0e0]+ [2.5e-1]+ 2*[1.0e3] # based on suplement to Ranganathan et al. (2021)

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08,  # -8 degree C, cuffey
                'C':1.e4
                }
class ReguRangaCB(EquationBase): #{{{
    """ Strain-rate informed regularization of simulatneous B,C inverions.
        This implementation is valid for SSA and MOLHO flow models.
        Ranganathan, Meghana, Brent Minchew, Colin R. Meyer, and G. Hilmar Gudmundsson. 
        “A New Approach to Inferring Basal Drag and Ice Rheology in Ice Streams, with Applications to West Antarctic Ice Streams.” 
        Journal of Glaciology 67, no. 262 (2021): 229-42. 
        https://doi.org/10.1017/jog.2020.95.
    """
    _EQUATION_TYPE = 'ReguRangaCB' 
    def __init__(self, parameters=ReguRangaCBEquationParameter()):
        super().__init__(parameters)

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of ReguRangaCB regularisation

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        Bid = self.local_output_var["B"]
        Cid = self.local_output_var["C"]

        uid = self.local_output_var["u"] 
        vid = self.local_output_var["v"]


        # unpacking normalized output
        B = slice_column(nn_output_var, Bid)
        C = slice_column(nn_output_var, Cid)

        # spatial derivatives
        u_x = jacobian(nn_output_var, nn_input_var, i=uid, j=xid)
        v_y = jacobian(nn_output_var, nn_input_var, i=vid, j=yid)
        u_y = jacobian(nn_output_var, nn_input_var, i=uid, j=yid)
        v_x = jacobian(nn_output_var, nn_input_var, i=vid, j=xid)

        sr_eff = ( u_x**2 + v_y**2 + 0.25*(u_y+v_x)**2 + u_x*v_y )**0.5

        k = 1.0e-2
        f = 1.0     # this is a floatation mask in the original study, 1 for grounded ice
        
        fac = 1.0 / (k*f - (1-f)*sr_eff)

        # residual
        f1 = (self.B-B)
        f2 = jacobian(f1, nn_input_var, i=0, j=xid)
        f3 = jacobian(f1, nn_input_var, i=0, j=yid)
        f4 = (self.C-C)
        f5 = jacobian(f2, nn_input_var, i=0, j=xid)
        f6 = jacobian(f2, nn_input_var, i=0, j=yid)

        return [fac*f1, fac*f2, fac*f3, f4,f5,f6] #}}}

    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MC 2D PDE, jax version

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        return self._pde(nn_input_var, nn_output_var) #}}}
    #}}}
#}}}