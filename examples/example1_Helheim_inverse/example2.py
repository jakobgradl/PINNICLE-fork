import deepxde as dde
import pinnicle

dde.config.set_default_float('float32')

# General parameters
hp = {}
hp["epochs"] = 100_000

hp['learning_rate'] = 1e-3 
hp['decay_rate'] = 0
hp['decay_steps'] = 0 

# NN
hp["num_neurons"] = 20
hp["num_layers"] = 6

# domain
hp["shapefile"] = "Helheim.exp"
hp["num_collocation_points"] = 9000

# physics
hp["equations"] = {"SSA_VB":{},
                   "ReguRangaCB":{}
                   }

# data
issm = {}
issm["data_size"] = {"u":4000, "v":4000, "s":4000, "H":4000, "C":None, "B":None}
issm["data_path"] = "Helheim.mat"
hp["data"] = {"ISSM":issm}

hp['is_save'] = False

# create experiment
experiment = pinnicle.PINN(hp)
experiment.compile()

# Train
experiment.train()
experiment.save_model(name='test')
