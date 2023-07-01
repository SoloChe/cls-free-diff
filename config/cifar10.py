import ml_collections
import torch


def get_config():
  config = ml_collections.ConfigDict()

  # data
  config.data = data = ml_collections.ConfigDict()
  data.data_dir = './data'
  data.dataset = 'CIFAR10'
  data.image_size = 32
  data.random_flip = True
  data.centered = False
  data.uniform_dequantization = False
  data.num_channels = 3
  data.num_classes = 10

  # training
  config.training = training = ml_collections.ConfigDict()
  training.lr = 2e-4
  training.grad_clip = 1. 
  training.total_steps = 30001
  training.warmup = 5000
  training.batch_size = 64 
  training.num_workers = 4
  training.ema_decay = 0.9999
  training.parallel = False
  training.threshold = 0.1
  

  # Gaussian Diffusion
  training.beta_1 = 1e-4 
  training.beta_T = 0.02
  training.T = 1000
  training.mean_type = 'epsilon'
  training.var_type =  'fixedlarge'

  config.training.ddim_eta = 0
  config.training.num_ddim_steps = 20
  
  # UNet
  config.model = model = ml_collections.ConfigDict()
  model.in_ch = 3
  model.mod_ch = 64
  model.out_ch = 3
  model.num_res_blocks = 2
  model.cdim = 10 
  model.ch_mult = [1, 2, 2, 2]
  model.attn = [1] # add attention to these levels
  model.num_res_blocks = 2 # resblock in each level
  model.dropout = 0.1

  # evaluation
  config.evaluate = evaluate = ml_collections.ConfigDict()
  evaluate.sample_size = 5
  evaluate.sample_step = 10000
  evaluate.sampler = 'ddim' # ddim or ddpm
  evaluate.save_step = 10000
  evaluate.eval_step = 0 # frequency of evaluating model, 0 to disable during training
  evaluate.w = 1.8 # class-conditional sampling temperature
  

  return config
