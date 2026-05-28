import deepxde as dde
import deepxde.backend as bkd
from ..parameter import PhysicsParameter
from . import EquationBase
import itertools
from ..utils import slice_column, jacobian, ppow, default_float_type
import torch

class MC_EXACT:
    """ all the physics used for the helmholtz decomposition of the mass-flux vector field
    """

    ## 1) flux to depth-avg. velocity

    def DR_xy(self, nn_input_var, nn_output_var):
        """ transform D,R scalar fields of mass-conserving stressbalance
            to u,v,dHdt,smb
        """

        xid = self.input_var.index('x')
        yid = self.input_var.index('y')
        
        Rid = self.output_var.index('R')
        R_x = jacobian(nn_output_var, nn_input_var, i=Rid, j=xid)
        R_y = jacobian(nn_output_var, nn_input_var, i=Rid, j=yid)
        
        if "D_dH" in self.output_var:
            # for MC_exact
            Did = self.output_var.index('D_dH')
            DdH_x = jacobian(nn_output_var, nn_input_var, i=Did, j=xid)
            DdH_y = jacobian(nn_output_var, nn_input_var, i=Did, j=yid)
        else:
            # for MCSteady_exact
            DdH_x = 0.
            DdH_y = 0.

        if "D_smb" in self.output_var:
            # for MC_exact
            Did = self.output_var.index('D_smb')
            Dsmb_x = jacobian(nn_output_var, nn_input_var, i=Did, j=xid)
            Dsmb_y = jacobian(nn_output_var, nn_input_var, i=Did, j=yid)
        else:
            # for MCSteady_exact
            Dsmb_x = 0.
            Dsmb_y = 0.

        return [Dsmb_x, Dsmb_y, DdH_x, DdH_y, R_x, R_y]
    
    def DR_to_Hu(self, nn_input_var, nn_output_var):
        """ recover mass-flux (Hu) from scalar fields D,R
        """
        Dsmb_x, _, DdH_x, _, _, R_y = self.DR_xy(nn_input_var,nn_output_var)
        Hu = (Dsmb_x + DdH_x - R_y)
        return Hu

    def DR_to_Hv(self, nn_input_var, nn_output_var):
        """ recover mass-flux (Hv) from scalar fields D,R
        """
        _, Dsmb_y, _, DdH_y, R_x, _ = self.DR_xy(nn_input_var,nn_output_var)
        Hv = (Dsmb_y + DdH_y + R_x)
        return Hv
    
    def Hu_to_ubar(self, nn_input_var, nn_output_var):
        """ get depth-averaged velocity (ubar) from mass flux
        """
        # Hid = self.output_var.index('H')
        # H = slice_column(nn_output_var, Hid)
        H = self.get_H(nn_input_var, nn_output_var)
        if 'Q_x' in self.output_var:
            QXid = self.output_var.index('Q_x')
            Hu = slice_column(nn_output_var, QXid)
        else:
            Hu = self.DR_to_Hu(nn_input_var,nn_output_var)
        ubar = Hu / H
        return ubar
    
    def Hv_to_vbar(self, nn_input_var, nn_output_var):
        """ get depth-averaged velocity (vbar) from mass flux
        """
        # Hid = self.output_var.index('H')
        # H = slice_column(nn_output_var, Hid)
        H = self.get_H(nn_input_var, nn_output_var)
        if 'Q_y' in self.output_var:
            QYid = self.output_var.index('Q_y')
            Hv = slice_column(nn_output_var, QYid)
        else:
            Hv = self.DR_to_Hv(nn_input_var,nn_output_var)
        vbar = Hv / H
        return vbar

    
    ## 2) depth-avg. velocity to surface velocity
    
    ### 2.1) plug-flow
    def u_MC(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        u = self.Hu_to_ubar(nn_input_var,nn_output_var)
        return u
    
    def v_MC(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        v = self.Hv_to_vbar(nn_input_var,nn_output_var)
        return v

    def vel_mag_MC(self, nn_input_var, nn_output_var, X):
        """ compute surface velocity magnitude (SSA)
        """
        u = self.u_MC(nn_input_var,nn_output_var,X)
        v = self.v_MC(nn_input_var,nn_output_var,X)
        vel = ppow((bkd.square(u) + bkd.square(v) + 1.0e-30), 0.5)
        return vel


    ### 2.2) flow with Parameterised Deformation and Sliding (PDS)
    def u_MC_pds(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        p = self.get_p(nn_input_var,nn_output_var)
        ubar = self.Hu_to_ubar(nn_input_var,nn_output_var)
        return ubar/p
    
    def v_MC_pds(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        p = self.get_p(nn_input_var,nn_output_var)
        vbar = self.Hv_to_vbar(nn_input_var,nn_output_var)
        return vbar/p

    def vel_mag_MC_pds(self, nn_input_var, nn_output_var, X):
        """ compute surface velocity magnitude (SSA)
        """
        u = self.u_MC_pds(nn_input_var,nn_output_var,X)
        v = self.v_MC_pds(nn_input_var,nn_output_var,X)
        vel = ppow((bkd.square(u) + bkd.square(v) + 1.0e-30), 0.5)
        return vel
    

    ### 2.3) Lliboutry model with sliding
    def u_MOLHO(self, nn_input_var, nn_output_var):
        """ compute MOLHO surface velocity from depth-averaged velocity
        """
        p = self.get_p(nn_input_var,nn_output_var)
        n = self.get_n(nn_input_var,nn_output_var)
        ubar = self.Hu_to_ubar(nn_input_var,nn_output_var)
        q = 1. - p
        f = (n+1.)/(n+2.)
        usurf = ubar * (p+f*q)**-1.
        return usurf
    
    def v_MOLHO(self, nn_input_var, nn_output_var):
        """ compute MOLHO surface velocity from depth-averaged velocity
        """
        p = self.get_p(nn_input_var,nn_output_var)
        n = self.get_n(nn_input_var,nn_output_var)
        vbar = self.Hv_to_vbar(nn_input_var,nn_output_var)
        q = 1. - p
        f = (n+1.)/(n+2.)
        vsurf = vbar * (p+f*q)**-1.
        return vsurf
    
    def u_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        return self.u_MOLHO(nn_input_var, nn_output_var)
    
    def v_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        return self.v_MOLHO(nn_input_var, nn_output_var)
    
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
        p = self.get_p(nn_input_var,nn_output_var)
        usurf = self.u_MC_MOLHO(nn_input_var, nn_output_var, X)
        ubase = usurf * p
        return ubase
    
    def v_base_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """compute basal velocity v component (MOLHO)
        """
        p = self.get_p(nn_input_var,nn_output_var)
        vsurf = self.v_MC_MOLHO(nn_input_var, nn_output_var, X)
        vbase = vsurf * p
        return vbase
    
    def vel_base_mag_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """ compute basal velocity magnitude (MOLHO)
        """
        ubase = self.u_base_MC_MOLHO(nn_input_var, nn_output_var, X)
        vbase = self.v_base_MC_MOLHO(nn_input_var, nn_output_var, X)
        vel_base = ppow((bkd.square(ubase) + bkd.square(vbase) + 1.0e-30), 0.5)
        return vel_base
    
    def u_shear_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """compute shear velocity u component (MOLHO)
        """
        p = self.get_p(nn_input_var,nn_output_var)
        usurf = self.u_MC_MOLHO(nn_input_var, nn_output_var, X)
        ushear = usurf * (1.-p)
        return ushear
    
    def v_shear_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """compute shear velocity v component (MOLHO)
        """
        p = self.get_p(nn_input_var,nn_output_var)
        vsurf = self.v_MC_MOLHO(nn_input_var, nn_output_var, X)
        vshear = vsurf * (1.-p)
        return vshear
    
    def vel_shear_mag_MC_MOLHO(self, nn_input_var, nn_output_var, X):
        """ compute shear velocity magnitude (MOLHO)
        """
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
        Dsmb_x, Dsmb_y, _, _, _, _ = self.DR_xy(nn_input_var,nn_output_var)
        Dsmb_xx = jacobian(Dsmb_x, nn_input_var, i=0, j=xid)
        Dsmb_yy = jacobian(Dsmb_y, nn_input_var, i=0, j=yid)
        smb = Dsmb_xx + Dsmb_yy ## == div(Hv)
        return smb
    
    def divQ_to_smb(self, nn_input_var, nn_output_var):
        """ recover smb from divergence of mass flux
        """
        xid = self.input_var.index('x')
        yid = self.input_var.index('y')
        Qxid = self.output_var.index('Q_x')
        Qyid = self.output_var.index('Q_y')
        Qx = slice_column(nn_output_var, Qxid)
        Qy = slice_column(nn_output_var, Qyid)
        Qx_x = jacobian(Qx, nn_input_var, i=0, j=xid)
        Qy_y = jacobian(Qy, nn_input_var, i=0, j=yid)
        smb = Qx_x + Qy_y
        return smb
    
    def smb_MC(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        if 'Q_x' in self.output_var:
            smb = self.divQ_to_smb(nn_input_var,nn_output_var)
        else:
            smb = self.DR_to_smb(nn_input_var,nn_output_var)
        return smb

    def mb_MC(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        vel = self.vel_mag_MC_MOLHO(nn_input_var, nn_output_var, X)
        threshold = 200. / 3.1536e7
        smb = self.smb_MC(nn_input_var, nn_output_var, X)
        return torch.where(vel<threshold,1.,0.) * smb

    
    ## 3) thickness change (dHdt)

    def DR_to_dH(self, nn_input_var, nn_output_var):
        """ recover dHdt from scalar fields D,R
        """
        xid = self.input_var.index('x')
        yid = self.input_var.index('y')
        _, _, DdH_x, DdH_y, _, _ = self.DR_xy(nn_input_var,nn_output_var)
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

    def H_MC(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        return self.get_H(nn_input_var, nn_output_var)

    def get_H(self, nn_input_var, nn_output_var):
        """ define H as exp(h) (positive definite)"""
        hid = self.output_var.index('H')
        h = slice_column(nn_output_var, hid)
        return bkd.exp(h)

    def p_MC(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        return self.get_p(nn_input_var, nn_output_var)

    def get_p(self, nn_input_var, nn_output_var):
        """ Sliding ratio:
            u_base = p*u_surf
            u_shear = (1-p)*u_surf
            p in [0,1]

            get p from nn_output or scalar_variables
        """
        if 'p' in self.output_var:
            p = self.p_to_range(nn_input_var,nn_output_var)
        else:
            p = self.equations[0].parameters.scalar_variables['p']
        return p

    def p_to_range(self, nn_input_var, nn_output_var):
        """ constrain p to [0,1]
        """
        lb = 0.8
        pid = self.output_var.index('p')
        p1 = slice_column(nn_output_var, pid)
        p = (1.0-lb) * bkd.sigmoid(p1) + lb # p in [lb,1]
        return p

    def get_n(self, nn_input_var, nn_output_var):
        """ get n from nn_output or scalar_variables
        """
        if 'n' in self.output_var:
            n = self.n_to_range(nn_input_var,nn_output_var)
        else:
            n = self.equations[0].parameters.scalar_variables['n']
        return n
    
    def n_to_range(self, nn_input_var, nn_output_var):
        """ constrain n to interval [a,b]
        """
        nid = self.output_var.index('n')
        n = slice_column(nn_output_var, nid)
        # a = self.equations[0].parameters.scalar_variables['nlb']
        # b = self.equations[0].parameters.scalar_variables['nub']
        # return (b-a) * bkd.sigmoid(n) + a
        return 1. + bkd.exp(n)
    
    def mf_mag(self, nn_input_var, nn_output_var,X):
        """ compute the mass flux magnitude
        """
        # vel_mag_MC * H
        Hu = self.DR_to_Hu(nn_input_var,nn_output_var)
        Hv = self.DR_to_Hv(nn_input_var,nn_output_var)
        return ppow((bkd.square(Hu) + bkd.square(Hv) + 1.0e-30), 0.5)
    
    def get_mu(self, nn_input_var, nn_output_var):
        """ define mu as exp(MU) (positive definite)
        """
        MUid = self.output_var.index('mu')
        MU = slice_column(nn_output_var, MUid)
        return bkd.exp(MU)
    
    def get_k(self, nn_input_var, nn_output_var):
        """ basal stress ratio: 
            tau_b = -k*tau_d
            k in [0,1]

            get k from nn_output or scalar_variables
        """
        if 'k' in self.output_var:
            k = self.k_to_range(nn_input_var,nn_output_var)
        else:
            k = self.equations[0].parameters.scalar_variables['k']
        return k

    def k_to_range(self, nn_input_var, nn_output_var):
        """constrain k to [0,1]
        """
        kid = self.output_var.index('k')
        k1 = slice_column(nn_output_var, kid)
        k = bkd.sigmoid(k1) # k in [0,1]
        return k

    def get_u_base(self, nn_input_var, nn_output_var):
        """ get the correct basal velocity for the sliding laws
        """
        if 'n' in self.output_var:
            u_base = self.u_base_MC_MOLHO(nn_input_var,nn_output_var,None)
            vel_base_mag = self.vel_base_mag_MC_MOLHO(nn_input_var,nn_output_var,None)
        else:
            u_base = self.u_MC(nn_input_var,nn_output_var,None)
            vel_base_mag = self.vel_mag_MC(nn_input_var,nn_output_var,None)
        return u_base, vel_base_mag

    ## 5) boundary conditions

    def p_BC(self, nn_input_var, nn_output_var, X):
        """BC on p
        """
        f1 = self.p_vub(nn_input_var,nn_output_var,X)
        f2 = self.p_vlb(nn_input_var,nn_output_var,X)

        return f1 + f2
    
    def p_vub(self, nn_input_var, nn_output_var, X):
        """BC on p
           promote p=0 below specified velocity (vlb) on boundary
        """
        p = self.get_p(nn_input_var,nn_output_var) 
        vel = self.vel_mag_MC_MOLHO(nn_input_var, nn_output_var, X)

        vub = self.equations[0].parameters.scalar_variables['vub']

        return bkd.relu(vel-vub) * (1. - p)
    
    def p_vlb(self, nn_input_var, nn_output_var, X):
        """BC on p
           promote p=0 below specified velocity (vlb) on boundary
        """
        p = self.get_p(nn_input_var,nn_output_var) 
        vel = self.vel_mag_MC_MOLHO(nn_input_var, nn_output_var, X)

        vlb = self.equations[0].parameters.scalar_variables['vlb']

        return bkd.relu(-1.*(vel-vlb)) * p
    

    ## 6) stress balances (2d)

    ### 6.1) general
    def get_B(self, nn_input_var, nn_output_var):
        """ define B as exp(B) (positive definite)"""
        if 'B' in self.output_var:
            Bid = self.output_var.index('B')
            B = slice_column(nn_output_var, Bid)
            B = bkd.exp(B)
        else:
            B = self.equations[0].parameters.scalar_variables['B']
        return B
    
    def get_C(self, nn_input_var, nn_output_var):
        """ define C as C**2 (positive definite)"""
        Cid = self.output_var.index('C')
        C = slice_column(nn_output_var, Cid)
        return C**2
    
    def get_b(self, nn_input_var, nn_output_var):
        """ get basal elevation b """
        bid = self.output_var.index('b')
        b = slice_column(nn_output_var, bid)
        return b
    
    def get_s(self, nn_input_var, nn_output_var):
        """ get surface elevation s (bed elevation plus thickness)"""
        H = self.get_H(nn_input_var, nn_output_var)
        b = self.get_b(nn_input_var, nn_output_var)
        return b + H

    def s_MC(self, nn_input_var, nn_output_var, X):
        return self.get_s(nn_input_var, nn_output_var)
        
    def s_x(self, nn_input_var, nn_output_var):
        xid = self.input_var.index('x')
        s = self.get_s(nn_input_var, nn_output_var)
        s_x = jacobian(nn_output_var, nn_input_var, i=0, j=xid)
        return s_x

    def s_y(self, nn_input_var, nn_output_var):
        yid = self.input_var.index('y')
        s = self.get_s(nn_input_var, nn_output_var)
        s_y = jacobian(nn_output_var, nn_input_var, i=0, j=yid)
        return s_y
    
    def tau_d_x(self, nn_input_var, nn_output_var):
        rho = self.equations[0].parameters.scalar_variables['rho']
        g = self.equations[0].parameters.scalar_variables['g']
        H = self.get_H(nn_input_var,nn_output_var)
        s_x = self.s_x(nn_input_var,nn_output_var)
        return rho*g*H*s_x

    def tau_d_y(self, nn_input_var, nn_output_var):
        rho = self.equations[0].parameters.constants['rhoi']
        g = self.equations[0].parameters.constants['g']
        H = self.get_H(nn_input_var,nn_output_var)
        s_y = self.s_y(nn_input_var,nn_output_var)
        return rho*g*H*s_y

    def tau_b_x(self, nn_input_var, nn_output_var):
        k = self.get_k(nn_input_var,nn_output_var)
        tau_d_x = self.tau_d_x(nn_input_var,nn_output_var)
        return -1.*k*tau_d_x

    def tau_b_y(self, nn_input_var, nn_output_var):
        k = self.get_k(nn_input_var,nn_output_var)
        tau_d_y = self.tau_d_y(nn_input_var,nn_output_var)
        return -1.*k*tau_d_y

    def effective_strain_rate_SSA(self, nn_input_var, nn_output_var):
        xid = self.input_var.index('x')
        yid = self.input_var.index('y')
        u = self.u_MC(nn_input_var, nn_output_var, None)
        v = self.v_MC(nn_input_var, nn_output_var, None)
        dux = jacobian(u, nn_input_var, i=0, j=xid)
        dvy = jacobian(v, nn_input_var, i=0, j=yid)
        duy = jacobian(u, nn_input_var, i=0, j=yid)
        dvx = jacobian(v, nn_input_var, i=0, j=xid)
        return dux**2 + dvy**2 + 0.25*(duy+dvx)**2 + (dux*dvy)

    ### 6.2) sliding laws

    def C_Weertman(self, nn_input_var, nn_output_var):
        """ get Weertman friction coefficient 
        """
        tau_b = self.tau_b_x(nn_input_var,nn_output_var)
        u_base, vel_base_mag = self.get_u_base(nn_input_var,nn_output_var)
        m = self.equations[0].parameters.scalar_variables['Weertman_friction_exponent']
        return tau_b * vel_base_mag**(1.-m) * u_base 


    ### 6.3) rheology

    def B_Glen(self, nn_input_var, nn_output_var):
        mu = self.get_mu(nn_input_var,nn_output_var)
        n = self.equations[0].parameters.scalar_variables['n']
        exponent = (1-n)/2*n
        sr_eff = self.eff_strain_rate_SSA(nn_input_var,nn_output_var)
        return 2.*mu * sr_eff**exponent

    ### 6.4) SSA

    def action_SSA(self, nn_input_var, nn_output_var, X):
        return self.SSA_weak(nn_input_var, nn_output_var)
    
    def SSA_weak(self, nn_input_var, nn_output_var):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
            weak-form pde loss for SSA flow model
        """
        ## need in output: B,C,s
        ## need functions: get_B, get_C, get_s

        n = self.equations[0].parameters.scalar_variables['n']
        rho = self.equations[0].parameters.scalar_variables['rho']
        g = self.equations[0].parameters.scalar_variables['g']
        m = self.equations[0].parameters.scalar_variables['m']

        H = self.get_H(nn_input_var,nn_output_var)
        B = self.get_B(nn_input_var,nn_output_var)
        C = self.get_C(nn_input_var,nn_output_var)
        s = self.get_s(nn_input_var,nn_output_var)
        u = self.u_MC(nn_input_var,nn_output_var,None)
        v = self.v_MC(nn_input_var,nn_output_var,None)

        sx = self.s_x(nn_input_var,nn_output_var)
        sy = self.s_y(nn_input_var,nn_output_var)
        u_mag = self.vel_mag_MC(nn_input_var,nn_output_var,None)
        sr_eff = self.effective_strain_rate_SSA(nn_input_var, nn_output_var)
        sr_term = sr_eff**((n+1)/2*n)

        VISC = 2*n/(n+1) * H * B * sr_term
        GRAV = rho * g * H * (sx*u + sy*v)
        FRIC = m/(m+1) * C * u_mag**((1/m)+1)

        return VISC + FRIC + GRAV