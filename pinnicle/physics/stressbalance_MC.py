import deepxde as dde
import deepxde.backend as bkd
from deepxde.backend import jax, abs
from . import EquationBase, Constants, Physics
from ..parameter import EquationParameter
from ..utils import slice_column, jacobian, slice_function_jax


################
################

class SSAweakEquationParamter(EquationParameter, Constants):
    """default parameters for SSA_exact
    """
    _EQUATION_TYPE = 'SSA_weak'
    def __init__(self, param_dict={}):
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['b','C','B']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        # self.output_lb[2] = 7.
        # self.output_ub[2] = 8.
        # self.output_lb[2] = -10.
        # self.output_ub[2] = 0.
        self.output_lb[2] = 0.
        self.output_ub[2] = 3.
        self.data_weights = [1.0e-3] + [1.0]*2
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
        
class SSA_weak(EquationBase):
    """ SSA with weak-form pde loss
    """
    _EQUATION_TYPE = 'SSA_weak'
    def __init__(self, parameters=SSAweakEquationParamter()):
        super().__init__(parameters)
    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ no pde loss required
        """
        return [] 
    
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ 
        """
        pass






# MOLHO-MC


# MOLHO constant B{{{
class MOLHOMCEquationParameter(EquationParameter, Constants):
    """ default parameters for MOLHO
    """
    _EQUATION_TYPE = 'MOLHO_MC' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['D_smb', 'R', 'H', 's', 'C', 'p', 'n']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0]*3 + [1.0e-6, 1.0e-6, 1.0e-8, 1.0]
        self.residuals = ["f"+self._EQUATION_TYPE+" 1", "f"+self._EQUATION_TYPE+" 2", "f"+self._EQUATION_TYPE+" base 1", "f"+self._EQUATION_TYPE+" base 2"]
        self.pde_weights = [1.0e-10, 1.0e-10, 1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08   # -8 degree C, cuffey
                }

class MOLHO_MC(EquationBase): #{{{
    """ MOLHO on 2D problem with uniform B
    """
    _EQUATION_TYPE = 'MOLHO_MC' 
    def __init__(self, parameters=MOLHOMCEquationParameter()):
        super().__init__(parameters)

        # gauss points for integration
        self.constants = {"gauss_x":[0.5, 0.23076534494715845, 0.7692346550528415, 0.04691007703066802, 0.9530899229693319],
                "gauss_weights":[0.5688888888888889,0.4786286704993665,0.4786286704993665,0.2369268850561891,0.2369268850561891]}

    def _pde(self, nn_input_var, nn_output_var): #{{{
        """ residual of MOLHO 2D PDEs

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
       # get the ids
        xid = self.local_input_var["x"]
        yid = self.local_input_var["y"]

        #
        Rid = self.local_output_var["R"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        Cid = self.local_output_var["C"]
        pid = self.local_output_var["p"]

        # unpacking normalized output
        H = slice_column(nn_output_var, Hid)
        C = slice_column(nn_output_var, Cid)
        p1 = slice_column(nn_output_var, pid)
        p = bkd.sigmoid(p1) # p in [0,1]

        if 'n' in self.local_output_var:
            nid = self.local_output_var['n']
            n = slice_column(nn_output_var, nid)
            a = 5.
            b = 1.8
            n = (a-b) * bkd.sigmoid(n) + b
        else:
            n = self.n

        # recovering u,v
        R_x = jacobian(nn_output_var, nn_input_var, i=Rid, j=xid)
        R_y = jacobian(nn_output_var, nn_input_var, i=Rid, j=yid)

        if 'D_smb' in self.local_output_var:
            Dsmbid = self.local_output_var["D_smb"]
            Dsmb_x = jacobian(nn_output_var, nn_input_var, i=Dsmbid, j=xid)
            Dsmb_y = jacobian(nn_output_var, nn_input_var, i=Dsmbid, j=yid)
        else:
            Dsmb_x = 1e-30 * R_x
            Dsmb_y = 1e-30 * R_y

        if 'D_dH' in self.local_output_var:
            DdHid = self.local_output_var["D_dH"]
            DdH_x = jacobian(nn_output_var, nn_input_var, i=DdHid, j=xid)
            DdH_y = jacobian(nn_output_var, nn_input_var, i=DdHid, j=yid)
        else:
            DdH_x = 1e-30 * R_x
            DdH_y = 1e-30 * R_y

        # a = D_x + D_y ## == div(Hv)
        ubar = (Dsmb_x + DdH_x - R_y) / H
        vbar = (Dsmb_y + DdH_y + R_x) / H

        q = 1. - p
        f = (n+1.)/(n+2.)
        usurf = ubar * (p+f*q)**-1.
        vsurf = vbar * (p+f*q)**-1.

        ubase = usurf * p
        vbase = vsurf * p

        ushear = usurf * (1.-p)
        vshear = vsurf * (1.-p)

        # spatial derivatives
        u_x = jacobian(usurf, nn_input_var, i=0, j=xid)
        v_x = jacobian(vsurf, nn_input_var, i=0, j=xid)
        ub_x = jacobian(ubase, nn_input_var, i=0, j=xid)
        vb_x = jacobian(vbase, nn_input_var, i=0, j=xid)
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)

        u_y = jacobian(usurf, nn_input_var, i=0, j=yid)
        v_y = jacobian(vsurf, nn_input_var, i=0, j=yid)
        ub_y = jacobian(ubase, nn_input_var, i=0, j=yid)
        vb_y = jacobian(vbase, nn_input_var, i=0, j=yid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        # compute mus
        mu1 = 0.0
        mu2 = 0.0
        mu3 = 0.0
        mu4 = 0.0

        for i,zeta in enumerate(self.constants["gauss_x"]):
            shear_comp = 1.0 - zeta**(n+1.0)
            epsilon_eff2 = (ub_x + (u_x-ub_x)*shear_comp)**2.0 + (vb_y + (v_y-vb_y)*shear_comp)**2.0 + (0.5*(ub_y+vb_x+(u_y-ub_y+v_x-vb_x)*shear_comp))**2.0 \
                    + (0.5*(n+1)/H*(ushear)*(1-shear_comp))**2.0 + (0.5*(n+1)/H*(vshear)*(1-shear_comp))**2.0 + (ub_x + (u_x-ub_x)*shear_comp)*(vb_y + (v_y-vb_y)*shear_comp)

            mu = 0.5*self.B*(epsilon_eff2 + self.eps)**(0.5*(1.0-n)/n)
            mu1 += 0.5*H*mu*self.constants["gauss_weights"][i]
            mu2 += 0.5*H*mu*self.constants["gauss_weights"][i]*(shear_comp)
            mu3 += 0.5*H*mu*self.constants["gauss_weights"][i]*(shear_comp**2.0)
            mu4 += 0.5*H*mu*self.constants["gauss_weights"][i]*(((n+1.0)/H*zeta**n)**2.0)

        # stress tensor
        B11 = mu1*(4.0*ub_x+2.0*vb_y) + mu2*(4.0*(u_x-ub_x)+2.0*(v_y-vb_y))
        B12 = mu1*(ub_y+vb_x) + mu2*(u_y-ub_y+v_x-vb_x)
        # B21 = B12
        B22 = mu1*(2.0*ub_x+4.0*vb_y) + mu2*(2.0*(u_x-ub_x)+4.0*(v_y-vb_y))
        B31 = mu2*(4.0*ub_x+2.0*vb_y) + mu3*(4.0*(u_x-ub_x)+2.0*(v_y-vb_y))
        B32 = mu2*(ub_y+vb_x) + mu3*(u_y-ub_y+v_x-vb_x)
        #B41 = B32
        B42 = mu2*(2.0*ub_x+4.0*vb_y) + mu3*(2.0*(u_x-ub_x)+4.0*(v_y-vb_y))

        # Getting the other derivatives
        sigma11 = jacobian(B11, nn_input_var, i=0, j=xid)
        sigma12 = jacobian(B12, nn_input_var, i=0, j=yid)

        sigma21 = jacobian(B12, nn_input_var, i=0, j=xid)
        sigma22 = jacobian(B22, nn_input_var, i=0, j=yid)

        sigma31 = jacobian(B31, nn_input_var, i=0, j=xid)
        sigma32 = jacobian(B32, nn_input_var, i=0, j=yid)

        sigma41 = jacobian(B32, nn_input_var, i=0, j=xid)
        sigma42 = jacobian(B42, nn_input_var, i=0, j=yid)

        # compute the basal stress
        u_norm = (ubase**2+vbase**2+self.eps**2)**0.5
        alpha = C**2 * (u_norm)**(1.0/self.n)

        f1 = sigma11 + sigma12 - alpha*ubase/(u_norm) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - alpha*vbase/(u_norm) - self.rhoi*self.g*H*s_y
        f3 = sigma31 + sigma32 + mu4*ushear - self.rhoi*self.g*H*s_x*(n+1.0)/(n+2.0)
        f4 = sigma41 + sigma42 + mu4*vshear - self.rhoi*self.g*H*s_y*(n+1.0)/(n+2.0)

        return [f1, f2, f3, f4] #}}}
    def _pde_jax(self, nn_input_var, nn_output_var): #{{{
        """ residual of MOLHO 2D PDEs

        Args:
            nn_input_var: global input to the nn
            nn_output_var: global output from the nn
        """
        pass
    #}}}
#}}}
#}}}






################
################
# SSA-MC



# generalised SSA_MC
class SSAMCEquationParameter(EquationParameter, Constants):
    """ default parameters for SSA
    """
    _EQUATION_TYPE = 'SSA_MC' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        # self.output = ['D_smb', 'D_dH', 'R', 's', 'H', 'C', 'B']
        self.output = ['D_dH', 'R', 's', 'H', 'C']#, 'B']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        # self.data_weights = [1.0, 1.0, 1.0, 1.0e-6, 1.0e-6, 1.0e-8, 1.0e-16]
        self.data_weights = [1.0]*2 + [1.0e-6, 1.0e-6, 1.0e-8]#, 1.0e-16]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08   # -8 degree C, cuffey
                }
        
class SSA_MC(EquationBase): #{{{
    """ SSA on 2D problem with uniform B, no friction law, but use taub=-beta*u
    """
    _EQUATION_TYPE = 'SSA_MC' 
    def __init__(self, parameters=SSAMCEquationParameter()):
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

        Rid = self.local_output_var["R"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        Cid = self.local_output_var["C"]

        # unpacking normalized output
        # s = slice_column(nn_output_var, sid)
        H = slice_column(nn_output_var, Hid)
        C = slice_column(nn_output_var, Cid)

        if 'B' in self.local_output_var:
            Bid = self.local_output_var["B"]
            B = slice_column(nn_output_var, Bid)
        else:
            B = self.B

        if 'n' in self.local_output_var:
            nid = self.local_output_var['n']
            n = slice_column(nn_output_var, nid)
            a = 5.
            b = 1.8
            n = (a-b) * bkd.sigmoid(n) + b
        else:
            n = self.n
        
        # recovering u,v
        R_x = jacobian(nn_output_var, nn_input_var, i=Rid, j=xid)
        R_y = jacobian(nn_output_var, nn_input_var, i=Rid, j=yid)

        if 'D_smb' in self.local_output_var:
            Dsmbid = self.local_output_var["D_smb"]
            Dsmb_x = jacobian(nn_output_var, nn_input_var, i=Dsmbid, j=xid)
            Dsmb_y = jacobian(nn_output_var, nn_input_var, i=Dsmbid, j=yid)
        else:
            Dsmb_x = 1e-30 * R_x
            Dsmb_y = 1e-30 * R_y

        if 'D_dH' in self.local_output_var:
            DdHid = self.local_output_var["D_dH"]
            DdH_x = jacobian(nn_output_var, nn_input_var, i=DdHid, j=xid)
            DdH_y = jacobian(nn_output_var, nn_input_var, i=DdHid, j=yid)
        else:
            DdH_x = 1e-30 * R_x
            DdH_y = 1e-30 * R_y

        # a = D_x + D_y ## == div(Hv)
        u = (Dsmb_x + DdH_x - R_y) / H
        v = (Dsmb_y + DdH_y + R_x) / H

        # spatial derivatives
        u_x = jacobian(u, nn_input_var, i=0, j=xid)
        v_x = jacobian(v, nn_input_var, i=0, j=xid)
        u_y = jacobian(u, nn_input_var, i=0, j=yid)
        v_y = jacobian(v, nn_input_var, i=0, j=yid)
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        eta = 0.5*B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+self.eps)**(0.5*(1.0-self.n)/self.n)
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
        # Weertman sliding
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
        pass
    #}}}
#}}}






# # mass-conserving SSA with taub
# class SSAMCTauEquationParameter(EquationParameter, Constants):
#     """ default parameters for SSA Taub
#     """
#     _EQUATION_TYPE = 'SSA_MC Taub' 
#     def __init__(self, param_dict={}):
#         # load necessary constants
#         Constants.__init__(self)
#         super().__init__(param_dict)

#     def set_default(self):
#         self.input = ['x', 'y']
#         self.output = ['D', 'R', 's', 'H', 'taub']
#         self.output_lb = [self.variable_lb[k] for k in self.output]
#         self.output_ub = [self.variable_ub[k] for k in self.output]
#         self.data_weights = [1.0, 1.0, 1.0e-6, 1.0e-6, 1.0e-10]#, 1.0e-2*self.yts**2]
#         self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
#         self.pde_weights = [1.0e-10, 1.0e-10]

#         # scalar variables: name:value
#         self.scalar_variables = {
#                 'n': 3.0,               # exponent of Glen's flow law
#                 'B':1.26802073401e+08   # -8 degree C, cuffey
#                 }
# class SSA_MC_Taub(EquationBase): #{{{
#     """ SSA on 2D problem with uniform B, no friction law, but use taub=-beta*u
#     """
#     _EQUATION_TYPE = 'SSA_MC Taub' 
#     def __init__(self, parameters=SSAMCTauEquationParameter()):
#         super().__init__(parameters)
#     def _pde(self, nn_input_var, nn_output_var): #{{{
#         """ residual of SSA 2D PDEs

#         Args:
#             nn_input_var: global input to the nn
#             nn_output_var: global output from the nn
#         """
#         # get the ids
#         xid = self.local_input_var["x"]
#         yid = self.local_input_var["y"]

#         Did = self.local_output_var["D"]
#         Rid = self.local_output_var["R"]
#         sid = self.local_output_var["s"]
#         Hid = self.local_output_var["H"]
#         taubid = self.local_output_var["taub"]

#         # unpacking normalized output
#         H = slice_column(nn_output_var, Hid)
#         taub = slice_column(nn_output_var, taubid)
        
#         # recovering u,v,a
#         D_x = jacobian(nn_output_var, nn_input_var, i=Did, j=xid)
#         D_y = jacobian(nn_output_var, nn_input_var, i=Did, j=yid)
#         R_x = jacobian(nn_output_var, nn_input_var, i=Rid, j=xid)
#         R_y = jacobian(nn_output_var, nn_input_var, i=Rid, j=yid)

#         # a = D_x + D_y ## == div(Hv)
#         u = (D_x - R_y) / H
#         v = (D_y + R_x) / H

#         # u = Physics.DR_to_u(self, nn_input_var, nn_output_var)
#         # v = Physics.DR_to_v(self, nn_input_var, nn_output_var)

#         # spatial derivatives
#         u_x = jacobian(u, nn_input_var, i=0, j=xid)
#         v_x = jacobian(v, nn_input_var, i=0, j=xid)
#         u_y = jacobian(u, nn_input_var, i=0, j=yid)
#         v_y = jacobian(v, nn_input_var, i=0, j=yid)
#         s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
#         s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

#         eta = 0.5*self.B *(u_x**2.0 + v_y**2.0 + 0.25*(u_y+v_x)**2.0 + u_x*v_y+self.eps)**(0.5*(1.0-self.n)/self.n)
#         # stress tensor
#         etaH = eta * H
#         B11 = etaH*(4*u_x + 2*v_y)
#         B22 = etaH*(4*v_y + 2*u_x)
#         B12 = etaH*(  u_y +   v_x)

#         # Getting the other derivatives
#         sigma11 = jacobian(B11, nn_input_var, i=0, j=xid)
#         sigma12 = jacobian(B12, nn_input_var, i=0, j=yid)

#         sigma21 = jacobian(B12, nn_input_var, i=0, j=xid)
#         sigma22 = jacobian(B22, nn_input_var, i=0, j=yid)


#         # compute the basal stress
#         u_norm = (u**2+v**2+self.eps**2)**0.5

#         f1 = sigma11 + sigma12 - abs(taub)*u/(u_norm) - self.rhoi*self.g*H*s_x
#         f2 = sigma21 + sigma22 - abs(taub)*v/(u_norm) - self.rhoi*self.g*H*s_y

#         return [f1, f2] #}}}
#     def _pde_jax(self, nn_input_var, nn_output_var): #{{{
#         """ residual of SSA 2D PDEs

#         Args:
#             nn_input_var: global input to the nn
#             nn_output_var: global output from the nn
#         """
#         # get the ids
#         xid = self.local_input_var["x"]
#         yid = self.local_input_var["y"]

#         Did = self.local_output_var["D"]
#         Rid = self.local_output_var["R"]
#         sid = self.local_output_var["s"]
#         Hid = self.local_output_var["H"]
#         taubid = self.local_output_var["taub"]

#         # unpacking normalized output
#         H = slice_column(nn_output_var, Hid)
#         taub = slice_column(nn_output_var, taubid)
        
#         # recovering u,v,a
#         D_x = jacobian(nn_output_var, nn_input_var, i=Did, j=xid)
#         D_y = jacobian(nn_output_var, nn_input_var, i=Did, j=yid)
#         R_x = jacobian(nn_output_var, nn_input_var, i=Rid, j=xid)
#         R_y = jacobian(nn_output_var, nn_input_var, i=Rid, j=yid)

#         # a = D_x + D_y ## == div(Hv)
#         u = (D_x - R_y) / H
#         v = (D_y + R_x) / H

#         # get the spatial derivatives functions
#         u_x = jacobian(u, nn_input_var, i=0, j=xid, val=1)
#         v_x = jacobian(v, nn_input_var, i=0, j=xid, val=1)
#         u_y = jacobian(u, nn_input_var, i=0, j=yid, val=1)
#         v_y = jacobian(v, nn_input_var, i=0, j=yid, val=1)

#         # get variable function
#         H_func = lambda x: slice_function_jax(nn_output_var, x, Hid)
#         # stress tensor
#         etaH = lambda x: 0.5*H_func(x)*self.B *(u_x(x)**2.0 + v_y(x)**2.0 + 0.25*(u_y(x)+v_x(x))**2.0 + u_x(x)*v_y(x)+self.eps)**(0.5*(1.0-self.n)/self.n)

#         B11 = lambda x: etaH(x)*(4*u_x(x) + 2*v_y(x))
#         B22 = lambda x: etaH(x)*(4*v_y(x) + 2*u_x(x))
#         B12 = lambda x: etaH(x)*(  u_y(x) +   v_x(x))

#         # Getting the other derivatives
#         sigma11 = jacobian((jax.vmap(B11)(nn_input_var), B11), nn_input_var, i=0, j=xid)
#         sigma12 = jacobian((jax.vmap(B12)(nn_input_var), B12), nn_input_var, i=0, j=yid)

#         sigma21 = jacobian((jax.vmap(B12)(nn_input_var), B12), nn_input_var, i=0, j=xid)
#         sigma22 = jacobian((jax.vmap(B22)(nn_input_var), B22), nn_input_var, i=0, j=yid)

#         # compute the basal stress
#         s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
#         s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

#         u_norm = (u**2+v**2+self.eps**2)**0.5

#         f1 = sigma11 + sigma12 - abs(taub)*u/(u_norm) - self.rhoi*self.g*H*s_x
#         f2 = sigma21 + sigma22 - abs(taub)*v/(u_norm) - self.rhoi*self.g*H*s_y

#         return [f1, f2] #}}}
#     #}}}
# #}}}








# mass-conserving SSA with taub
# for regions with negligible smb and dH/dt
class SSAMCsteadyTauEquationParameter(EquationParameter, Constants):
    """ default parameters for SSA Taub
    """
    _EQUATION_TYPE = 'SSA_MCsteady Taub' 
    def __init__(self, param_dict={}):
        # load necessary constants
        Constants.__init__(self)
        super().__init__(param_dict)

    def set_default(self):
        self.input = ['x', 'y']
        self.output = ['R', 's', 'H', 'taub']
        self.output_lb = [self.variable_lb[k] for k in self.output]
        self.output_ub = [self.variable_ub[k] for k in self.output]
        self.data_weights = [1.0, 1.0e-5, 1.0e-5, 1.0e-10]
        self.residuals = ["f"+self._EQUATION_TYPE+"1", "f"+self._EQUATION_TYPE+"2"]
        self.pde_weights = [1.0e-10, 1.0e-10]

        # scalar variables: name:value
        self.scalar_variables = {
                'n': 3.0,               # exponent of Glen's flow law
                'B':1.26802073401e+08   # -8 degree C, cuffey
                }
class SSA_MCsteady_Taub(EquationBase): #{{{
    """ SSA on 2D problem with uniform B, no friction law, but use taub
    """
    _EQUATION_TYPE = 'SSA_MCsteady Taub' 
    def __init__(self, parameters=SSAMCsteadyTauEquationParameter()):
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

        Rid = self.local_output_var["R"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        taubid = self.local_output_var["taub"]

        # unpacking normalized output
        H = slice_column(nn_output_var, Hid)
        taub = slice_column(nn_output_var, taubid)
        
        # recovering u,v,a
        R_x = jacobian(nn_output_var, nn_input_var, i=Rid, j=xid)
        R_y = jacobian(nn_output_var, nn_input_var, i=Rid, j=yid)

        # a = D_x + D_y ## == div(Hv)
        u = -1. * R_y / H
        v = R_x / H

        # spatial derivatives
        u_x = jacobian(u, nn_input_var, i=0, j=xid)
        v_x = jacobian(v, nn_input_var, i=0, j=xid)
        u_y = jacobian(u, nn_input_var, i=0, j=yid)
        v_y = jacobian(v, nn_input_var, i=0, j=yid)
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

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

        f1 = sigma11 + sigma12 - abs(taub)*u/(u_norm) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - abs(taub)*v/(u_norm) - self.rhoi*self.g*H*s_y

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

        Rid = self.local_output_var["R"]
        sid = self.local_output_var["s"]
        Hid = self.local_output_var["H"]
        taubid = self.local_output_var["taub"]

        # unpacking normalized output
        H = slice_column(nn_output_var, Hid)
        taub = slice_column(nn_output_var, taubid)
        
        # recovering u,v,a
        R_x = jacobian(nn_output_var, nn_input_var, i=Rid, j=xid)
        R_y = jacobian(nn_output_var, nn_input_var, i=Rid, j=yid)

        # a = D_x + D_y ## == div(Hv)
        u = -1. * R_y / H
        v = R_x / H

        # get the spatial derivatives functions
        u_x = jacobian(u, nn_input_var, i=0, j=xid, val=1)
        v_x = jacobian(v, nn_input_var, i=0, j=xid, val=1)
        u_y = jacobian(u, nn_input_var, i=0, j=yid, val=1)
        v_y = jacobian(v, nn_input_var, i=0, j=yid, val=1)

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

        # compute the basal stress
        s_x = jacobian(nn_output_var, nn_input_var, i=sid, j=xid)
        s_y = jacobian(nn_output_var, nn_input_var, i=sid, j=yid)

        u_norm = (u**2+v**2+self.eps**2)**0.5

        f1 = sigma11 + sigma12 - abs(taub)*u/(u_norm) - self.rhoi*self.g*H*s_x
        f2 = sigma21 + sigma22 - abs(taub)*v/(u_norm) - self.rhoi*self.g*H*s_y

        return [f1, f2] #}}}
    #}}}
#}}}