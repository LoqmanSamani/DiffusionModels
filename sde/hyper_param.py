import torch
import torch.nn as nn




class HyperParamsSDE(nn.Module):
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02, trainable_beta=False, beta_method="linear",
                 sigma_start=1e-3, sigma_end=10.0, start=0.0, end=1.0):
        super().__init__()
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.trainable_beta = trainable_beta
        self.beta_method = beta_method
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.start = start
        self.end = end

        if not (0 < self.beta_start < self.beta_end):
            raise ValueError(f"beta_start ({self.beta_start}) and beta_end ({self.beta_end}) must satisfy 0 < start < end")
        if not (0 < self.sigma_start < self.sigma_end):
            raise ValueError(f"sigma_start ({self.sigma_start}) and sigma_end ({self.sigma_end}) must satisfy 0 < start < end")
        if self.num_steps <= 0:
            raise ValueError(f"num_steps ({self.num_steps}) must be positive")

        beta_range = (beta_start, beta_end)
        betas_init = self.compute_beta_schedule(beta_range, num_steps, beta_method)
        self.time = torch.linspace(self.start, self.end, self.num_steps, dtype=torch.float32)
        self.dt = (self.end - self.start) / self.num_steps

        if trainable_beta:
            self.betas = nn.Parameter(betas_init)
        else:
            self.register_buffer('betas', betas_init)
            self.register_buffer('cum_betas', torch.cumsum(betas_init, dim=0) * self.dt)
            self.register_buffer("sigmas", self.sigma_start * (self.sigma_end / self.sigma_start) ** self.time)

    def compute_beta_schedule(self, beta_range, num_steps, method):
        beta_min, beta_max = beta_range
        if method == "sigmoid":
            x = torch.linspace(-6, 6, num_steps)
            beta = torch.sigmoid(x) * (beta_max - beta_min) + beta_min
        elif method == "quadratic":
            x = torch.linspace(beta_min ** 0.5, beta_max ** 0.5, num_steps)
            beta = x ** 2
        elif method == "constant":
            beta = torch.full((num_steps,), beta_max)
        elif method == "inverse_time":
            beta = 1.0 / torch.linspace(num_steps, 1, num_steps)
            beta = beta_min + (beta_max - beta_min) * (beta - beta.min()) / (beta.max() - beta.min())
        elif method == "linear":
            beta = torch.linspace(beta_min, beta_max, num_steps)
        else:
            raise ValueError(f"Unknown beta_method: {method}. Supported: linear, sigmoid, quadratic, constant, inverse_time")
        beta = torch.clamp(beta, min=beta_min, max=beta_max)
        return beta

    def constrain_betas(self):
        if self.trainable_beta:
            with torch.no_grad():
                self.betas.clamp_(min=self.beta_start, max=self.beta_end)

    def get_variance(self, time_steps, method):
        if method == "ve":
            return self.sigmas[time_steps] ** 2
        elif method == "vp":
            return 1 - torch.exp(-self.cum_betas[time_steps])
        elif method == "sub-vp":
            return 1 - torch.exp(-2 * self.cum_betas[time_steps])
        else:
            raise ValueError(f"Unknown method: {method}")