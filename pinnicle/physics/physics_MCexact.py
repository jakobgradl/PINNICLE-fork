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
    
    ### plug-flow
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


    ### flow with Parameterised Deformation and Sliding (PDS)
    def u_MC_pds(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        p = self.get_p(nn_input_var,nn_output_var)
        u = self.Hu_to_ubar(nn_input_var,nn_output_var)
        return u/p
    
    def v_MC_pds(self, nn_input_var, nn_output_var, X):
        """ a wrapper for PointSetOperatorBC func call, Args need to follow the requirment by deepxde
        """
        p = self.get_p(nn_input_var,nn_output_var)
        v = self.Hv_to_vbar(nn_input_var,nn_output_var)
        return v/p

    def vel_mag_MC_pds(self, nn_input_var, nn_output_var, X):
        """ compute surface velocity magnitude (SSA)
        """
        u = self.u_MC_pds(nn_input_var,nn_output_var,X)
        v = self.v_MC_pds(nn_input_var,nn_output_var,X)
        vel = ppow((bkd.square(u) + bkd.square(v) + 1.0e-30), 0.5)
        return vel
    

    ### Lliboutry model with sliding
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
        hid = self.output_var.index('H')
        h = slice_column(nn_output_var, hid)
        return bkd.exp(h)

    def get_p(self, nn_input_var, nn_output_var):
        """get p from nn_output or scalar_variables
        """
        if 'p' in self.output_var:
            p = self.p_to_range(nn_input_var,nn_output_var)
        else:
            p = self.equations[0].parameters.scalar_variables['p']
        return p

    def p_to_range(self, nn_input_var, nn_output_var):
        """constrain p to [0,1]
        """
        pid = self.output_var.index('p')
        p1 = slice_column(nn_output_var, pid)
        p = 0.5 * bkd.sigmoid(p1) + 0.5 # p in [0,1]
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
        """constrain n to interval [a,b]
        """
        nid = self.output_var.index('n')
        n = slice_column(nn_output_var, nid)
        # a = self.equations[0].parameters.scalar_variables['nlb']
        # b = self.equations[0].parameters.scalar_variables['nub']
        # return (b-a) * bkd.sigmoid(n) + a
        return 1. + bkd.exp(n)
    
    def mf_mag(self, nn_input_var, nn_output_var,X):
        """compute the mass flux magnitude
        """
        # vel_mag_MC * H
        Hu = self.DR_to_Hu(nn_input_var,nn_output_var)
        Hv = self.DR_to_Hv(nn_input_var,nn_output_var)
        return ppow((bkd.square(Hu) + bkd.square(Hv) + 1.0e-30), 0.5)


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
    


