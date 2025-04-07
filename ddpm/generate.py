import torch
import torch.nn as nn
from reverse_diffusion import ReverseDDPM  # Adjusted to match previous naming





class Generate(nn.Module):
    """Image generation using a trained DDPM model."""
    def __init__(self, config):
        super().__init__()
        self.model_path = config.model_path  # Path to the trained model
        self.num_steps = config.num_steps  # Total diffusion steps
        self.beta_start = config.beta_start
        self.beta_end = config.beta_end
        self.device = config.device
        self.in_channels = config.in_channels
        self.image_shape = config.image_shape  # Assuming square images (height = width)
        self.num_time_steps = config.num_time_steps  # Should equal num_steps
        self.batch_size = getattr(config, 'batch_size', 1)  # Default to 1 if not specified

        # Validate num_time_steps matches num_steps
        if self.num_time_steps != self.num_steps:
            raise ValueError(f"num_time_steps ({self.num_time_steps}) must equal num_steps ({self.num_steps})")

    def forward(self):
        """Generates a batch of images by denoising random noise step-by-step."""
        # Initialize reverse diffusion process
        reverse = ReverseDDPM(
            num_steps=self.num_steps,
            beta_start=self.beta_start,
            beta_end=self.beta_end
        ).to(self.device)

        # Load trained model
        model = torch.load(self.model_path, map_location=self.device)
        model.eval()

        # Initialize noisy samples (batch_size images)
        noisy_samples = torch.randn(
            self.batch_size, self.in_channels, self.image_shape, self.image_shape
        ).to(self.device)

        # Denoise step-by-step
        with torch.no_grad():
            xt = noisy_samples
            for t in reversed(range(self.num_steps)):
                # Create timestep tensor for the batch
                time_steps = torch.full((self.batch_size,), t, device=self.device, dtype=torch.long)

                # Predict noise using the model
                predicted_noise = model(batch=xt, t=time_steps)

                # Step backwards in the reverse diffusion process
                xt = reverse(xt, predicted_noise, time_steps)

        # Post-process generated images
        generated_imgs = torch.clamp(xt, -1., 1.).detach().cpu()
        generated_imgs = (generated_imgs + 1) / 2  # Rescale from [-1, 1] to [0, 1]

        return generated_imgs


# Example config class for testing
class Config:
    model_path = "path/to/model.pth" #TODO: fix this
    num_steps = 1000
    beta_start = 1e-4
    beta_end = 0.02
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_channels = 3
    image_shape = 32
    num_time_steps = 1000
    batch_size = 4  # Generate 4 images at once


# Example usage
if __name__ == "__main__":
    config = Config()
    generator = Generate(config)
    generated_images = generator()
    print("Generated images shape:", generated_images.shape)  # Should be [4, 3, 32, 32]