import torch
import torch.nn as nn
from reverse_ddpm import ReverseDDPM




class Generate(nn.Module):
    """image generation using trained ddpm model"""
    def __init__(self, config):
        super().__init__()
        self.model_path = config.model_path # path, where the trained model is stored!
        self.num_steps = config.num_steps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.device = config.device
        self.in_channels = config.in_channels
        self.image_shape = config.image_shape
        self.num_time_steps = config.num_time_steps

    def forward(self):

        # initialize diffusion reverse process
        reverse = ReverseDDPM(
            num_steps=self.num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end
        )
        model = torch.load(self.model_path).to(self.device)
        model.eval()
        noisy_sample = torch.randn(1, self.in_channels, self.image_shape, self.image_shape).to(self.device)

        # denoising the noise sample step by step to generate X0 (an image without noise)
        with torch.no_grad():
            for t in reversed(range(self.num_time_steps)):
                predicted_noise = model(batch=noisy_sample, t=torch.as_tensor(t).unsqueeze(0).to(self.device))
                # predicted_t is the generated image in time step t, predicted_0 is the predicted image in time step 0
                predicted_t, predicted_0 = reverse.remove_noise(
                    batch_t=noisy_sample,
                    predicted_noise=predicted_noise,
                    time_step=torch.as_tensor(t).to(self.device)
                )

        generated_img = torch.clamp(predicted_t, -1., 1.).detach().cpu()
        generated_img = (generated_img + 1) / 2

        return generated_img
    
