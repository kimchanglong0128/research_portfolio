# forward process 

import torch 
import torch.nn as nn


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DiffusionScheduler(nn.Module):
    """
    DDPM-style diffusion scheduler implementation.
    - linear beta schedule
    - q_sample(x0, t) ->  q(x_t | noise)
    """

    def __init__(self, timesteps:int=1000, beta_start:float=1e-4, beta_end:float=0.02, device:str=device):
        super(DiffusionScheduler, self).__init__()
        self.timesteps = timesteps
        
        # Create linear beta schedule
        betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32, device=device)
        
        # Precompute alphas and their cumulative products
        alphas = 1.0 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)      # ᾱ_t

        self.register_buffer("betas", betas.to(device))
        self.register_buffer("alphas", alphas.to(device))
        self.register_buffer("alpha_cumprod", alpha_cumprod.to(device))

    def q_sample(self, x0:torch.Tensor, t:torch.Tensor, noise:torch.Tensor=None):
        """
        x0 = [B, C, H, W]
        t = [B]  (timesteps)
        x_t = sqrt(ᾱ_t) * x0 + sqrt(1 - ᾱ_t) * noise
        noise ~ N(0, I) -> ε
        """
        if noise is None:
            noise = torch.randn_like(x0)

        device = x0.device
        alpha_cumprod = self.alpha_cumprod.to(device)
        t= t.to(device)

        a_bar = alpha_cumprod[t].view(-1, 1, 1, 1)  # [B, 1, 1, 1
        x_t = a_bar.sqrt() * x0 + (1 - a_bar).sqrt() * noise
        return x_t, noise