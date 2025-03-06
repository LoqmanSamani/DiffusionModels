import torch

class Config:
    def __init__(self, method=None, start=None, end=None, max_steps=None, sigma_min=None, sigma_max=None, beta_range=None):
        self.method = method or "smld"
        self.start = start or 0
        self.end = end or 1
        self.max_steps = max_steps or 1000
        self.step_size = (self.end - self.start) / self.max_steps  
        self.sigma_min = sigma_min or 0.01
        self.sigma_max = sigma_max or 1.0
        self.beta_range = beta_range or (1e-4, 0.02)
        
        # compute hyperparameters
        self.sigmas, self.betas, self.t = self.compute_params()

    def compute_params(self):
        t = torch.linspace(self.start, self.end, self.max_steps)  # time steps
        sigmas = self.sigma_min * (self.sigma_max / self.sigma_min) ** t  # geometric variance schedule
        betas = torch.linspace(self.beta_range[0], self.beta_range[1], self.max_steps)
        return sigmas, betas, t