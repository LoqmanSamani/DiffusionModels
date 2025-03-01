import torch



class ReverseDDPM:
    """
    reverse diffusion process of the "Denoising Diffusion Probabilistic Model (DDPM)".
    the class removes noise from an image that has undergone a diffusion process.
    it predicts the denoised image at the previous time step (`xt-1`).
    """

    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):

        params = self.compute_params(num_steps, beta_start, beta_end)
        self.betas = params[0]  # noise variances
        self.alphas = params[1]  # (1 - betas)
        self.alpha_bars = params[2]  # cumulative product of alphas
        self.num_steps = num_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

    @staticmethod
    def compute_params(num_steps, beta_start, beta_end):
        """
        computes the noise schedule parameters used in the diffusion process.
        """
        betas = torch.linspace(start=beta_start, end=beta_end, steps=num_steps)  # Linear noise schedule
        alphas = 1 - betas  # Noise reduction factors
        alpha_bars = torch.cumprod(alphas, dim=0)  # Cumulative product of alphas

        return betas, alphas, alpha_bars

    def remove_noise(self, batch_t, predicted_noise, time_step):
        """
        performs one step of the reverse diffusion process to estimate the original image
        and also predict the previous time step image.
        """
        # estimate the original clean image x0 using the DDPM formula (it is skipped in the original paper!!!)
        # batch0 = (batch_t - (torch.sqrt(1 - self.alpha_bars.to(batch_t.device)[time_step])) * predicted_noise) / \
        #          (torch.sqrt(self.alpha_bars.to(batch_t.device)[time_step]))
        # batch0 = torch.clamp(batch0, min=-1.0, max=1.0)  # clamp values to [-1,1]

        # used to calculate x_t-1
        pred = (batch_t - ((1 - self.alphas.to(batch_t.device)[time_step]) * predicted_noise) /
                     (torch.sqrt(1 - self.alpha_bars.to(batch_t.device)[time_step]))) / \
                     (torch.sqrt(self.alphas.to(batch_t.device)[time_step]))

        # if t=0, return x0 since we don’t predict earlier steps
        if time_step == 0:
            return pred #, batch0

        # compute the variance term for adding noise
        var = (1 - self.alpha_bars.to(batch_t.device)[time_step - 1]) / (1 - self.alpha_bars.to(batch_t.device)[time_step])
        var = var * self.betas.to(batch_t.device)[time_step]
        std = var ** 0.5  # Standard deviation

        # sample Gaussian noise
        z = torch.randn(batch_t.shape).to(batch_t.device)

        # predict the next image x_{t-1}
        predicted = pred + std * z

        return predicted #, batch0