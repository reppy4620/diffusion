import torch

class SDE:
    def sde(self, x, t):
        pass

    def reverse_sde(self, score, x, t):
        drift, diffusion = self.sde(x, t)
        drift = drift - (diffusion ** 2)[:, None, None, None] * score
        return drift, diffusion

    def probability_flow(self, score, x, t):
        drift, diffusion = self.sde(x, t)
        drift = drift - 0.5 * (diffusion ** 2)[:, None, None, None] * score
        diffusion = torch.zeros_like(diffusion)
        return drift, diffusion
    
class VESDE(SDE):
    def __init__(self, sigma_min=0.01, sigma_max=50.):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def sde(self, x, t):
        drift = torch.zeros_like(x)
        sigma_t = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma_t * torch.sqrt(2 * (torch.log(self.sigma_max) - torch.log(self.sigma_min)))
        return drift, diffusion
    
    def marginal_prob(self, x, t):
        mean = x
        std = self.sigma_min ** 2 * (self.sigma_max / self.sigma_min) ** (2 * t)
        return mean, std
    
class VPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20):
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def sde(self, x, t):
        beta_t = self.beta_0 + (self.beta_1 - self.beta_0) * t
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)
        return drift, diffusion
    
    def marginal_prob(self, x, t):
        beta_int = self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t ** 2
        log_mean_coeff = -0.5 * beta_int
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff))
        return mean, std

class SubVPSDE(SDE):
    def __init__(self, beta_min=0.1, beta_max=20):
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def sde(self, x, t):
        beta_t = self.beta_0 + (self.beta_1 - self.beta_0) * t
        drift = -0.5 * beta_t[:, None, None, None] * x
        beta_int = self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t ** 2
        diffusion = torch.sqrt(beta_t * (1. - torch.exp(-2. * beta_int)))
        return drift, diffusion
    
    def marginal_prob(self, x, t):
        beta_int = self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t ** 2
        log_mean_coeff = -0.5 * beta_int
        mean = torch.exp(log_mean_coeff)[:, None, None, None] * x
        std = 1. - torch.exp(2. * log_mean_coeff)
        return mean, std
    
