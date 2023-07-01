import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusion_base import Diffusion
from utils import ConditionalEmbedding
import numpy as np


class GaussianDiffusionTrainer(Diffusion):
    def __init__(self, model, config):
        super().__init__(config)

        self.model = model
        self.config = config
        # reggister buffer
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_alphas_bar', self.sqrt_alphas_bar_)
        self.register_buffer(
            'sqrt_one_minus_alphas_bar', self.sqrt_one_minus_alphas_bar_)


    def forward(self, x_0, cemb):
        """
        Algorithm 1. with label embedding
        """
       

        # x_0.shape = [batch_size, channels, height, width]
        # select a batch of timesteps
        t = torch.randint(self.T, size=(x_0.shape[0], ), device=x_0.device)
        noise = torch.randn_like(x_0)
        x_t = (
            self.extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
            self.extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise) 
        loss = F.mse_loss(self.model(x_t, t, cemb), noise, reduction='none')
        return loss


class DDPM_Sampler(Diffusion):
    def __init__(self, model, config):
        
        super().__init__(config)

        self.model = model
        self.w = config.evaluate.w

        self.register_buffer('betas', self.betas_)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'sqrt_recip_alphas_bar', self.sqrt_recip_alphas_bar_)
        self.register_buffer(
            'sqrt_recipm1_alphas_bar', self.sqrt_recipm1_alphas_bar_)
        
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer(
            'posterior_var',
            self.betas_ * (1. - self.alphas_bar_prev_) / (1. - self.alphas_bar_))
        
        # below: log calculation clipped because the posterior variance is 0 at
        # the beginning of the diffusion chain
        self.register_buffer(
            'posterior_log_var_clipped',
            torch.log(
                torch.cat([self.posterior_var[1:2], self.posterior_var[1:]]))) # replace the first element with the second element
        
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(self.alphas_bar_prev_) * self.betas_ / (1. - self.alphas_bar_))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(self.alphas_) * (1. - self.alphas_bar_prev_) / (1. - self.alphas_bar_))
        
    def predict_x0_from_eps(self, x_t, t, eps):
            assert x_t.shape == eps.shape
            return (
                self.extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
                self.extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps
            )

    def q_mean_variance(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = self.extract(self.posterior_log_var_clipped, t, x_t.shape)
        return posterior_mean, posterior_log_var_clipped

    
    def p_sample(self, x_t, t, cemb): # p_theta(x_{t-1} | x_t)
        # below: only log_variance is used in the KL computations
        model_log_var = {
            # for fixedlarge, we set the initial (log-)variance like so to
            # get a better decoder log likelihood
            'fixedlarge': torch.log(torch.cat([self.posterior_var[1:2],
                                               self.betas[1:]])),
            'fixedsmall': self.posterior_log_var_clipped,
        }[self.var_type]

        model_log_var = self.extract(model_log_var, t, x_t.shape)

        # Mean parameterization
        eps_cond = self.model(x_t, t, cemb)
        nu_emb = torch.zeros(cemb.shape, device = eps_cond.device)
        eps_uncond = self.model(x_t, t, nu_emb)
        eps = (1+self.w)*eps_cond - self.w*eps_uncond

        x_0 = self.predict_x0_from_eps(x_t, t, eps=eps)
        mean, log_var = self.q_mean_variance(x_0, x_t, t)
       
        # x_0 = torch.clip(x_0, -1., 1.)
        # no noise when t == 0
        time_step = t[0]
        if time_step > 0:
            noise = torch.randn_like(x_t) # for a batch of images
        else:
            noise = 0

        x_prev = mean + torch.exp(0.5 * log_var) * noise
        return x_prev
    
    def forward(self, x_T, cemb):
        """
        Algorithm 2.
        """
        x_t = x_T
        for time_step in reversed(range(self.T)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step 
            x_t = self.p_sample(x_t, t, cemb)
        x_0 = x_t
        return torch.clip(x_0, -1, 1)
    
class DDIM_Sampler(Diffusion):
    def __init__(self, model, config):
      
        super().__init__(config)

        self.model = model
        self.w = config.evaluate.w

        self.register_buffer('ddim_sigma', self.ddim_sigma_)

        self.register_buffer('ddim_steps', self.ddim_steps_)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer(
            'ddim_sqrt_recip_alphas_bar', self.ddim_sqrt_recip_alphas_bar_)
        self.register_buffer(
            'ddim_sqrt_recipm1_alphas_bar', self.ddim_sqrt_recipm1_alphas_bar_)
    
        self.register_buffer(
            'posterior_mean_coef1',
            torch.sqrt(self.ddim_alpha_prev_))
        self.register_buffer(
            'posterior_mean_coef2',
            torch.sqrt(1-self.ddim_alpha_prev_-self.ddim_sigma_**2))

    def predict_x0_from_eps(self, x_t, idx, eps):
        assert x_t.shape == eps.shape
        return (
            self.extract(self.ddim_sqrt_recip_alphas_bar, idx, x_t.shape) * x_t -
            self.extract(self.ddim_sqrt_recipm1_alphas_bar, idx, x_t.shape) * eps
        )

    def q_mean_variance(self, x_0, x_t, idx, eps):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, idx, x_t.shape) * x_0 +
            self.extract(self.posterior_mean_coef2, idx, x_t.shape) * eps
        )
        # return sigma here not variance
        posterior_sigma = self.extract(self.ddim_sigma, idx, x_t.shape)
        return posterior_mean, posterior_sigma
    
    def p_sample(self, x_t, t, idx, cemb):

        eps_cond = self.model(x_t, t, cemb)
        nu_emb = torch.zeros(cemb.shape, device = eps_cond.device)
        eps_uncond = self.model(x_t, t, nu_emb)
        eps = (1+self.w)*eps_cond - self.w*eps_uncond

        x_0 = self.predict_x0_from_eps(x_t, idx, eps)

        mean, sigma = self.q_mean_variance(x_0, x_t, idx, eps)
        
        x_prev = mean + sigma * eps
        
        return x_prev

    def forward(self, x_T, cemb):
        
        x_t = x_T
        for idx, time_step in enumerate(reversed(self.ddim_steps)):
            t = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * time_step 
            idx = x_t.new_ones([x_T.shape[0], ], dtype=torch.long) * (len(self.ddim_steps) - idx - 1)
            x_t = self.p_sample(x_t, t, idx, cemb)
        x_0 = x_t
        return torch.clip(x_0, -1, 1)

        


       

        