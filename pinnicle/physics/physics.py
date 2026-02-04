import deepxde as dde
import deepxde.backend as bkd
from ..parameter import PhysicsParameter
from . import EquationBase
import itertools
from ..utils import slice_column, jacobian, ppow, default_float_type

class Physics:
    """ All the physics in used as constraint in the PINN
    """
    def __init__(self, parameters=PhysicsParameter()):
        self.parameters = parameters

        # add all physics 
        self.equations = [EquationBase.create(eq, parameters=self.parameters.equations[eq])  for eq in self.parameters.equations] 

        # update (global) input, output variable list from local_input_var and local_output_var of each equations
        self.input_var = self._update_global_variables([p.local_input_var for p in self.equations])
        self.output_var = self._update_global_variables([p.local_output_var for p in self.equations])

        # update the index in each of physics
        for p in self.equations:
            p.update_id(self.input_var, self.output_var)

        # find the min and max of the lb and ub of the output_var among all physics
        self.output_lb = []
        self.output_ub = []
        self.data_weights = []
        for k in self.output_var:
            self.output_lb.append(min([p.output_lb[k] for p in self.equations if k in p.output_lb]))
            self.output_ub.append(max([p.output_ub[k] for p in self.equations if k in p.output_ub]))
            self.data_weights.append(max([p.data_weights[k] for p in self.equations if k in p.data_weights]))
        
        # manualy update data weights
        if self.parameters.manual_data_weights is not None:
             for k in self.parameters.manual_data_weights:
                 if k in self.output_var:
                     kid = self.output_var.index(k)
                     self.data_weights[kid] = self.parameters.manual_data_weights[k]

        # update residual list
        self.residuals = list(itertools.chain.from_iterable([p.residuals for p in self.equations]))
        self.pde_weights = list(itertools.chain.from_iterable([p.pde_weights for p in self.equations]))

    def _update_global_variables(self, local_var_list):
        """ Update global variables based on a list of local variables,
            find all unqiue keys, then put in one single List

        Args: 
            local_var_list: list of local variables in the equation
        """
        # merge all dict, get all unique keys
        global_var = {}
        for d in local_var_list:
            global_var.update(d)

        return list(global_var.keys())

    def pdes(self, nn_input_var, nn_output_var):
        """ a wrapper of all the equations used in the PINN, Args need to follow the requirment by deepxde

        Args: 
            nn_input_var:  input tensor to the nn
            nn_output_var: output tensor from the nn
        """
        eq = []
        for p in self.equations:
            eq += p.pde(nn_input_var, nn_output_var) 
        return eq

    def vel_mag(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde

        Args: 
            nn_input_var:  input tensor to the nn
            nn_output_var: output tensor from the nn
            X:  NumPy array of the collocation points defined on the boundary, required by deepxde
        """
        uid = self.output_var.index('u')
        vid = self.output_var.index('v')
        u = slice_column(nn_output_var, uid)
        v = slice_column(nn_output_var, vid) 
        vel = ppow((bkd.square(u) + bkd.square(v) + 1.0e-30), 0.5)
        return vel

    def surf_x(self, nn_input_var, nn_output_var, X):
        """dsdx
        """
        sid = self.output_var.index('s')
        xid = self.input_var.index('x')
        dsdx = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        return dsdx

    def surf_y(self, nn_input_var, nn_output_var, X):
        """dsdy
        """
        sid = self.output_var.index('s')
        yid = self.input_var.index('y')
        dsdy = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)
        return dsdy

    def user_defined_gradient(self, output_var, input_var):
        """ compute the gradient of output_var with respect to the input_var, return a function wrapper for PointSetOperatorBC
            TODO: implement jax version

        Args: 
            input_var: string name of input variable (independent variable)
            output_var: string name of output variable (dependent variable)
        """
        def _wrapper(nn_input_var, nn_output_var, X):
            yid = self.output_var.index(output_var)
            xid = self.input_var.index(input_var)
            dydx = jacobian(nn_output_var, nn_input_var, i=yid, j=xid)
            return dydx

        return _wrapper

    def calving_front(self, nn_input_var, nn_output_var, X):
        """ calculate the calving front boundary condition

        Args:             
            nn_input_var:  input tensor to the nn
            nn_output_var: output tensor from the nn
            X:  NumPy array of the collocation points defined on the boundary, required by deepxde
        """
        eqind = next((i for i,p in enumerate(self.equations) if p._EQUATION_TYPE.upper() == "CALVINGFRONT"), None)
        return self.equations[eqind]._bc(nn_input_var, nn_output_var)

    def operator(self, pname):
        """ grab the pde operator, used for testing the pdes and plotting

        Args:
            pname : pde operator name (string), case insensitive
        """
        for p in self.equations:
            if p._EQUATION_TYPE.lower() == pname.lower():
                return p.pde



### transform functions for DHNN:

## 1) velocity

    def DR_xy(self, nn_input_var, nn_output_var, boundary):
        """ transform D,R scalar fields of mass-conserving stressbalance
            to u,v,dHdt,smb
        """

        xid = self.input_var.index('x')
        yid = self.input_var.index('y')
        
        Rid = self.output_var.index('R')
        R_x = jacobian(nn_output_var, nn_input_var, i=Rid, j=xid)
        R_y = jacobian(nn_output_var, nn_input_var, i=Rid, j=yid)
        
        if "D_dH" in self.output_var and boundary==False:
            # for MC_exact
            Did = self.output_var.index('D_dH')
            DdH_x = jacobian(nn_output_var, nn_input_var, i=Did, j=xid)
            DdH_y = jacobian(nn_output_var, nn_input_var, i=Did, j=yid)
        else:
            # for MCSteady_exact
            DdH_x = R_x*1e-32
            DdH_y = R_y*1e-32

        if "D_smb" in self.output_var and boundary==False:
            # for MC_exact
            Did = self.output_var.index('D_smb')
            Dsmb_x = jacobian(nn_output_var, nn_input_var, i=Did, j=xid)
            Dsmb_y = jacobian(nn_output_var, nn_input_var, i=Did, j=yid)
        else:
            # for MCSteady_exact
            Dsmb_x = R_x*1e-32
            Dsmb_y = R_y*1e-32

        return [Dsmb_x, Dsmb_y, DdH_x, DdH_y, R_x, R_y]
    
    def DR_to_Hu(self, nn_input_var, nn_output_var, boundary):
        """ recover mass-flux (Hu) from scalar fields D,R
        """
        Dsmb_x, _, DdH_x, _, _, R_y = self.DR_xy(nn_input_var,nn_output_var, boundary)
        Hu = (Dsmb_x + DdH_x - R_y)
        return Hu

    def DR_to_Hv(self, nn_input_var, nn_output_var, boundary):
        """ recover mass-flux (Hv) from scalar fields D,R
        """
        _, Dsmb_y, _, DdH_y, R_x, _ = self.DR_xy(nn_input_var,nn_output_var, boundary)
        Hv = (Dsmb_y + DdH_y + R_x)
        return Hv
    
    def Hu_to_ubar(self, nn_input_var, nn_output_var, boundary):
        """ get depth-averaged velocity (ubar) from mass flux
        """
        Hid = self.output_var.index('H')
        H = slice_column(nn_output_var, Hid)
        Hu = self.DR_to_Hu(nn_input_var,nn_output_var, boundary)
        ubar = Hu / H
        return ubar
    
    def Hv_to_vbar(self, nn_input_var, nn_output_var, boundary):
        """ get depth-averaged velocity (vbar) from mass flux
        """
        Hid = self.output_var.index('H')
        H = slice_column(nn_output_var, Hid)
        Hv = self.DR_to_Hv(nn_input_var,nn_output_var, boundary)
        vbar = Hv / H
        return vbar
    
    def u_MC(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        u = self.Hu_to_ubar(nn_input_var,nn_output_var,boundary=False)
        return u
    
    def v_MC(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        v = self.Hv_to_vbar(nn_input_var,nn_output_var,boundary=False)
        return v

    def vel_mag_MC(self, nn_input_var, nn_output_var, X):
        """ compute surface velocity magnitude (SSA)
        """
        u = self.u_MC(nn_input_var,nn_output_var,X)
        v = self.v_MC(nn_input_var,nn_output_var,X)
        vel = ppow((bkd.square(u) + bkd.square(v) + 1.0e-30), 0.5)
        return vel
    
    def u_MOLHO(self, nn_input_var, nn_output_var, boundary=False):
        """ compute MOLHO surface velocity from depth-averaged velocity
        """
        p = self.p_to_01(nn_input_var,nn_output_var)
        n = self.get_n(nn_input_var,nn_output_var)
        ubar = self.Hu_to_ubar(nn_input_var,nn_output_var,boundary)
        q = 1. - p
        f = (n+1.)/(n+2.)
        usurf = ubar * (p+f*q)**-1.
        return usurf
    
    def v_MOLHO(self, nn_input_var, nn_output_var, boundary=False):
        """ compute MOLHO surface velocity from depth-averaged velocity
        """
        p = self.p_to_01(nn_input_var,nn_output_var)
        n = self.get_n(nn_input_var,nn_output_var)
        vbar = self.Hv_to_vbar(nn_input_var,nn_output_var,boundary)
        q = 1. - p
        f = (n+1.)/(n+2.)
        vsurf = vbar * (p+f*q)**-1.
        return vsurf
    
    def u_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        return self.u_MOLHO(nn_input_var, nn_output_var, boundary=False)
    
    def v_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        return self.v_MOLHO(nn_input_var, nn_output_var, boundary=False)
    
    def u_MC_MOLHO_BC(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        return self.u_MOLHO(nn_input_var, nn_output_var, boundary=True)
    
    def v_MC_MOLHO_BC(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        return self.v_MOLHO(nn_input_var, nn_output_var, boundary=True)
    
    def vel_mag_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """ compute surface velocity magnitude (MOLHO)
        """
        u = self.u_MC_MOLHO(nn_input_var,nn_output_var, X)
        v = self.v_MC_MOLHO(nn_input_var,nn_output_var, X)
        vel = ppow((bkd.square(u) + bkd.square(v) + 1.0e-30), 0.5)
        return vel
    
    def u_base_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """compute basal velocity u component (MOLHO)
        """
        p = self.p_to_01(nn_input_var,nn_output_var)
        usurf = self.u_MC_MOLHO(nn_input_var, nn_output_var, X)
        ubase = usurf * p
        return ubase
    
    def v_base_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """compute basal velocity v component (MOLHO)
        """
        p = self.p_to_01(nn_input_var,nn_output_var)
        vsurf = self.v_MC_MOLHO(nn_input_var, nn_output_var, X)
        vbase = vsurf * p
        return vbase
    
    def vel_base_mag_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """ compute basal velocity magnitude (MOLHO)
        """
        # p = self.p_to_01(nn_input_var,nn_output_var)
        # usurf = self.u_MC_MOLHO(nn_input_var, nn_output_var, X)
        # vsurf = self.v_MC_MOLHO(nn_input_var, nn_output_var, X)
        # ubase = usurf * p
        # vbase = vsurf * p
        ubase = self.u_base_MC_MOLHO(nn_input_var, nn_output_var, X)
        vbase = self.v_base_MC_MOLHO(nn_input_var, nn_output_var, X)
        vel_base = ppow((bkd.square(ubase) + bkd.square(vbase) + 1.0e-30), 0.5)
        return vel_base
    
    def u_shear_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """compute shear velocity u component (MOLHO)
        """
        p = self.p_to_01(nn_input_var,nn_output_var)
        usurf = self.u_MC_MOLHO(nn_input_var, nn_output_var, X)
        ushear = usurf * (1.-p)
        return ushear
    
    def v_shear_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """compute shear velocity v component (MOLHO)
        """
        p = self.p_to_01(nn_input_var,nn_output_var)
        vsurf = self.v_MC_MOLHO(nn_input_var, nn_output_var, X)
        vshear = vsurf * (1.-p)
        return vshear
    
    def vel_shear_mag_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """ compute shear velocity magnitude (MOLHO)
        """
        # p = self.p_to_01(nn_input_var,nn_output_var)
        # usurf = self.u_MC_MOLHO(nn_input_var, nn_output_var, X)
        # vsurf = self.v_MC_MOLHO(nn_input_var, nn_output_var, X)
        # ushear = usurf * (1.-p)
        # vshear = vsurf * (1.-p)
        ushear = self.u_shear_MC_MOLHO(nn_input_var, nn_output_var, X)
        vshear = self.v_shear_MC_MOLHO(nn_input_var, nn_output_var, X)
        vel_shear = ppow((bkd.square(ushear) + bkd.square(vshear) + 1.0e-30), 0.5)
        return vel_shear
    
## 2) surface mass balance (smb)
    
    def DR_to_smb(self, nn_input_var, nn_output_var):
        """ recover smb from scalar fields D,R
        """
        xid = self.input_var.index('x')
        yid = self.input_var.index('y')
        Dsmb_x, Dsmb_y, _, _, _, _ = self.DR_xy(nn_input_var,nn_output_var,boundary=False)
        Dsmb_xx = jacobian(Dsmb_x, nn_input_var, i=0, j=xid)
        Dsmb_yy = jacobian(Dsmb_y, nn_input_var, i=0, j=yid)
        smb = Dsmb_xx + Dsmb_yy ## == div(Hv)
        return smb
    
    def smb_MC(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        smb = self.DR_to_smb(nn_input_var,nn_output_var)
        return smb
    
## 3) thickness change (dHdt)

    def DR_to_dH(self, nn_input_var, nn_output_var):
        """ recover dHdt from scalar fields D,R
        """
        xid = self.input_var.index('x')
        yid = self.input_var.index('y')
        _, _, DdH_x, DdH_y, _, _ = self.DR_xy(nn_input_var,nn_output_var,boundary=False)
        DdH_xx = jacobian(DdH_x, nn_input_var, i=0, j=xid)
        DdH_yy = jacobian(DdH_y, nn_input_var, i=0, j=yid)
        dH = -1. * (DdH_xx + DdH_yy) ## == div(Hv)
        return dH
    
    def dH_MC(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        dH = self.DR_to_dH(nn_input_var,nn_output_var)
        return dH
    
## 4) utils

    def p_to_01(self, nn_input_var, nn_output_var):
        """constrain p to [0,1]
        """
        pid = self.output_var.index('p')
        p1 = slice_column(nn_output_var, pid)
        p = bkd.sigmoid(p1) # p in [0,1]
        return p

    def get_n(self, nn_input_var, nn_output_var):
        """get n from nn_output or scalar_variables
        """
        if 'n' in self.output_var:
            n = self.n_to_range(nn_input_var,nn_output_var)
        else:
            n = self.equations[0].parameters.scalar_variables['n']
        return n
    
    def n_to_range(self, nn_input_var, nn_output_var):
        """constrain n to [1.8, 5.0]
        """
        nid = self.output_var.index('n')
        n = slice_column(nn_output_var, nid)
        a = 5.
        b = 1.8
        return (a-b) * bkd.sigmoid(n) + b
    
    def mf_mag(self, nn_input_var, nn_output_var,X):
        """compute the mass flux magnitude
        """
        # vel_mag_MC * H
        Hu = self.DR_to_Hu(nn_input_var,nn_output_var,boundary=False)
        Hv = self.DR_to_Hv(nn_input_var,nn_output_var,boundary=False)
        return ppow((bkd.square(Hu) + bkd.square(Hv) + 1.0e-30), 0.5)