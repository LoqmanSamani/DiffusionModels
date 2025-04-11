import torch
import torch.nn as nn
from reverse_ddpm import ReverseDDPM





class GenerateDDPM(nn.Module):
    """Image generation using a trained DDPM model."""
    def __init__(self, noise_predictor, hyper_params_model, image_shape, conditions=None, conditional_model=None,
                 batch_size=1, in_channels=3, device=None):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_predictor = noise_predictor.to(self.device)
        self.hyper_params_model = hyper_params_model.to(self.device)
        self.conditions = conditions  # tensor of conditions (e.g., tokenized text)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.in_channels = in_channels
        self.image_shape = image_shape  # tuple (height, width)
        self.batch_size = batch_size

        # validate inputs
        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def forward(self):
        """Generates a batch of images by denoising random noise step-by-step."""
        # initialize reverse diffusion process
        reverse = ReverseDDPM(hyper_params=self.hyper_params_model).to(self.device)

        # initialize noisy samples
        noisy_samples = torch.randn(
            self.batch_size, self.in_channels, self.image_shape[0], self.image_shape[1]
        ).to(self.device)

        # set models to evaluation mode
        self.noise_predictor.eval()
        if self.conditional_model is not None:
            self.conditional_model.eval()

        # denoise step-by-step
        with torch.no_grad():
            xt = noisy_samples
            for t in reversed(range(self.hyper_params_model.num_steps)):
                time_steps = torch.full((self.batch_size,), t, device=self.device, dtype=torch.long)

                # predict noise with or without conditions
                if self.conditional_model is not None and self.conditions is not None:
                    y = self.conditional_model(self.conditions)  # encode conditions
                    predicted_noise = self.noise_predictor(xt, time_steps, y)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps)

                # reverse diffusion step (use positional args for compatibility)
                xt = reverse(xt, predicted_noise, time_steps)

        # post-process generated images
        generated_imgs = torch.clamp(xt, -1., 1.).detach().cpu()
        generated_imgs = (generated_imgs + 1) / 2  # rescale from [-1, 1] to [0, 1]

        return generated_imgs

