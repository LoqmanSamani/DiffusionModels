import torch
import torch.nn as nn




class ReverseDDIM(nn.Module):
    """
    implements one step of the reverse process in Denoising Diffusion Implicit Models (DDIM).
    computes the previous noisy image (xt_prev) and the denoised image (x0) given the current noisy image (x),
    the predicted noise (p_noise), the current timestep (t), and the previous timestep (prev_t).

        :param config: configuration object containing model parameters like eta, alpha_tau, etc.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config


    def forward(self, x, p_noise, t, prev_t):
        """
        performs one step of the reverse process in DDIM framework to compute the previous noisy image (xt_prev)
        and the denoised image (x0).

            :param x: tensor representing the current noisy image at timestep t
            :param p_noise: tensor representing the predicted noise at timestep t
            :param t: tensor representing the current timestep
            :param prev_t: tensor representing the previous timestep
            :return: tuple containing:
                - xt_prev: tensor representing the previous noisy image
                - x0: tensor representing the denoised image
        """

        eta = self.config.eta
        at_sqrt = self.config.alpha_tau_sqrt.to(x.device)[t]
        at_prev_sqrt = self.config.alpha_tau_sqrt.to(x.device)[prev_t]

        x0 = x - p_noise * torch.sqrt(1 - at_sqrt) / at_sqrt
        c1 = eta * ((1 - at_sqrt / at_prev_sqrt) * (1 - at_prev_sqrt) / torch.clamp((1 - at_sqrt), min=1e-8))
        c2 = torch.clamp((1 - at_prev_sqrt) - c1 ** 2, min=1e-8)
        xt_prev = at_prev_sqrt * x0 + c1 * torch.randn_like(x) + c2 * p_noise

        return xt_prev, x0