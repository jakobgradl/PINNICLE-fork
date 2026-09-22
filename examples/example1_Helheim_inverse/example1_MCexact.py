# import torch
# import numpy as np
# import deepxde as dde
# import deepxde.backend as bkd
import pinnicle

# General parameters
hp = {}
epochs = 700_000
hp["epochs"] = epochs

hp['loss_functions'] = []
# hp['loss_weights'] = []

hp['learning_rate'] = 1e-3
hp['decay_rate'] = 0

hp['mini_batch'] = 2000

hp['is_save'] = False
hp['is_plot'] = False

# NN
hp["activation"] = "SiLU"
hp["initializer"] = "Glorot uniform"
hp["num_neurons"] = 50 
hp["num_layers"] = 8

# domain
hp["shapefile"] = "Helheim.exp"
# hp["num_collocation_points"] = 0

# physics
SSA_sCB = {}
SSA_sCB["scalar_variables"] = {'Bmin':1e7}
hp["equations"] = {"MC_exact_Helmholtz":{},"SSA_sC":{}}
# hp["equations"] = {"MC_exact_Helmholtz":{},"SSA_sCB":SSA_sCB}

# data
HELHEIM = {}
HELHEIM["name_map"] = {"u_MC":'u', "v_MC":'v', "s_MC":'s', "smb_MC":'a', "H_MC":'H'}
HELHEIM["data_size"] = {"u_MC":'MAX', "v_MC":'MAX', "s_MC":'MAX', "smb_MC":'MAX', "H_MC":'MAX'}
HELHEIM["data_path"] = "HelheimMCexact.mat"
HELHEIM["source"] = 'mat'

RES_SSA = {}
RES_SSA["data_path"] = "HelheimMCexact.mat"
RES_SSA["name_map"] = {"SSAx_weak":'u', "SSAy_weak":'v'}
RES_SSA["data_size"] = {'SSAx_weak':'MAX', 'SSAy_weak':'MAX'}
RES_SSA["source"] = 'mat'

hp['data'] = {'HELHEIM':HELHEIM,'RES_SSA':RES_SSA}


# losses
u_loss = {}
u_loss['name'] = "u"
u_loss['function'] = "MAE"
u_loss['weight'] = 1.0e8 

v_loss = {}
v_loss['name'] = "v"
v_loss['function'] = "MAE"
v_loss['weight'] = 1.0e8

H_loss = {}
H_loss['name'] = "H"
H_loss['function'] = "MSE"
H_loss['weight'] = 1.0e-3

s_loss = {}
s_loss['name'] = "s"
s_loss['function'] = "MSE"
s_loss['weight'] = 1.0e-2

a_loss = {}
a_loss['name'] = "a"
a_loss['function'] = "MSE"
a_loss['weight'] = 1.0e21 

ssax_loss = {}
ssax_loss['name'] = "SSAx"
ssax_loss['function'] = "MAE"
ssax_loss['weight'] = 1e-12

ssay_loss = {}
ssay_loss['name'] = "SSAy"
ssay_loss['function'] = "MAE"
ssay_loss['weight'] = 1e-12

hp["additional_loss"] = {"u_MC":u_loss,
                         "v_MC":v_loss,
                         "H_MC":H_loss,
                         "s_MC":s_loss,
                         "smb_MC":a_loss,
                         "SSAx_weak":ssax_loss,
                         "SSAy_weak":ssay_loss}

# create experiment
experiment = pinnicle.PINN(hp)
print(experiment.params)

experiment.compile()

# Train
experiment.train()

fname = f'example1_MCexact_model_sC_230826'
experiment.save_model(name=fname)
