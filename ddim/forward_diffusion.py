import torch.nn as nn




class ForwardDDIM(nn.Module):
    def __init__(self, config):
        """
        implements the forward diffusion process for ddim (denoising diffusion implicit models).
        this class adds controlled noise to input images over time, simulating the diffusion process.

            :param config: configuration object containing precomputed diffusion parameters
        """
        super().__init__()
        self.config = config  # store configuration object

    def forward(self, x, noise, t):
        """
        applies the forward diffusion step by adding noise to the input images.
        the function computes a weighted sum of the clean images and the noise based on
        the diffusion step `t`, using precomputed alpha values.

        :param x: batch of input images tensor of shape (batch_size, channels, height, width)
        :param noise: gaussian noise tensor of the same shape as x
        :param t: diffusion timestep tensor of shape (batch_size,), indicating how much noise to add
        :return: noisy images tensor of shape (batch_size, channels, height, width)
        """

        # extract the square root of alpha values at timestep t (controls signal strength)
        at_sqrt = self.config.alpha_sqrt.to(x.device)[t]  # signal rate
        com_at_sqrt = 1 - at_sqrt
        # reshape for broadcasting across image dimensions
        at_sqrt = at_sqrt[:, None, None, None]
        com_at_sqrt = com_at_sqrt[:, None, None, None]
        # compute the noisy image: weighted combination of original image and noise
        noisy_x = at_sqrt * x + com_at_sqrt * noise

        return noisy_x