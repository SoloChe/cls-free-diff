import torch
import torch.nn as nn
import torch.nn.functional as F




class Diffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        # parameters initialization
        self.T = config.training.T
        self.img_size = config.data.image_size

        # linear noise schedule
        self.betas_ = torch.linspace(config.training.beta_1, config.training.beta_T, config.training.T).double()
        self.alphas_ = 1. - self.betas_
        self.alphas_bar_ = torch.cumprod(self.alphas_, dim=0)

        self.sqrt_alphas_bar_ = torch.sqrt(self.alphas_bar_)
        self.sqrt_one_minus_alphas_bar_ = torch.sqrt(1. - self.alphas_bar_)

        if config.evaluate.sampler == 'ddpm':
            self.var_type = config.training.var_type
            self.alphas_bar_prev_ = F.pad(self.alphas_bar_, [1, 0], value=1)[:config.training.T] # add 1 to the beginning and remove the last element

            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.sqrt_recip_alphas_bar_ =  torch.sqrt(1. / self.alphas_bar_)
            self.sqrt_recipm1_alphas_bar_ =  torch.sqrt(1. / self.alphas_bar_ - 1)
            
        elif config.evaluate.sampler == 'ddim':
            self.ddim_eta = config.training.ddim_eta

            # ddim steps 
            c = self.T // config.training.num_ddim_steps
            self.ddim_steps_ = torch.tensor(list(range(0, self.T, c))) + 1

            self.ddim_alpha_ = self.alphas_bar_[self.ddim_steps_].clone()
            self.ddim_alpha_sqrt_ = torch.sqrt(self.ddim_alpha_)
            self.ddim_alpha_prev_ = torch.cat([self.alphas_bar_[0:1], self.alphas_bar_[self.ddim_steps_[:-1]]])

            self.ddim_sigma_ = (self.ddim_eta *
                                ((1 - self.ddim_alpha_prev_) / (1 - self.ddim_alpha_) *
                                (1 - self.ddim_alpha_ / self.ddim_alpha_prev_)) ** .5)
        
            self.ddim_sqrt_one_minus_alpha_ = (1. - self.ddim_alpha_) ** .5

            # calculations for diffusion q(x_t | x_{t-1}) and others
            self.ddim_sqrt_recip_alphas_bar_ =  torch.sqrt(1. / self.ddim_alpha_)
            self.ddim_sqrt_recipm1_alphas_bar_ =  torch.sqrt(1. / self.ddim_alpha_ - 1)

        else:
            print('Please specify the sampler type (ddpm or ddim) in the config file.')
            raise NotImplementedError
             
            
        
    @staticmethod
    def extract(v, t, x_shape):
        """
        Extract some coefficients at specified timesteps, then reshape to
        [batch_size, 1, 1, 1] for broadcasting purposes.
        """
        # [batch_size, channels, height, width]
        out = torch.gather(v, index=t, dim=0).float()
        return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))

    
    
    
   

