import torch
import torch.nn as nn
from reverse_ddim import ReverseDDIM




class Generate(nn.Module):
    def __init__(self, config):
        """
        performs image generation using a trained ddim (denoising diffusion implicit model).
        this class initializes a reverse diffusion process, starting from random noise and iteratively
        denoising it step by step to generate a clean image.

            :param config: configuration object containing necessary parameters such as:
               - num_tau_steps: number of generation steps (should be ≤ training steps)
               - device: computation device (cuda or cpu)
               - in_channels: number of image channels
               - image_shape: dimensions of the output image
        """
        super().__init__()
        self.config = config
        self.num_steps = config.num_tau_steps  # generation steps, should be ≤ training steps
        self.device = config.device
        self.in_channels = config.in_channels
        self.image_shape = config.image_shape

    @torch.no_grad()
    def forward(self):
        """
        performs the reverse diffusion process to generate an image.

        the function starts with random noise and iteratively removes noise using a trained model,
        storing intermediate outputs for visualization.

        :return: a tensor of generated images over time, shape (num_time_steps, in_channels, image_size, image_size)
        """

        # tensor to store intermediate image frames (for visualization purposes)
        frames = torch.zeros(self.num_steps, self.in_channels, self.image_shape, self.image_shape).to(self.device)

        # initialize the reverse diffusion process
        reverse = ReverseDDIM(self.config)

        # load the trained model
        model = torch.load(self.config.model_path, weights_only=False).to(self.device)
        model.eval()  # set model to evaluation mode

        # start from pure gaussian noise
        noisy_sample = torch.randn(1, self.in_channels, self.image_shape, self.image_shape).to(self.device)

        # iteratively remove noise in reverse order of time steps
        f = 0  # frame index counter
        for t in reversed(range(1, self.num_steps)):
            # predict noise at time step t using the trained model
            pred_noise = model(noisy_sample, torch.as_tensor(t).unsqueeze(0).to(self.device))

            # compute the previous time step (t-1)
            t = torch.as_tensor(t).to(self.device)
            t_prev = torch.as_tensor(t - 1).to(self.device)

            # remove noise to obtain a less noisy image for the previous time step
            xt_prev, x0 = reverse(noisy_sample, pred_noise, t, t_prev)

            # store the predicted x0 (denoised image) at this step
            frames[f, :, :, :] = x0
            f += 1

        # normalize output images to range [0,1] for visualization
        output = torch.clamp(frames, -1., 1.).detach().cpu()
        output = (output + 1) / 2  # scale from [-1,1] to [0,1]

        return output