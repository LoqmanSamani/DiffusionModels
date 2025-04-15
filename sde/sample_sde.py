import torch
import torch.nn as nn
from reverse_sde import ReverseSDE





class SampleSDE(nn.Module):

    def __init__(self, method, noise_predictor, hyper_params_model, image_shape, conditions=None, conditional_model=None,
                 batch_size=1, in_channels=3, device=None, output_range=(-1, 1)):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.method = method
        self.noise_predictor = noise_predictor.to(self.device)
        self.hyper_params_model = hyper_params_model.to(self.device)
        self.conditions = conditions.to(self.device) if isinstance(conditions, torch.Tensor) else conditions
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.in_channels = in_channels
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.output_range = output_range

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

    def forward(self, normalize_output=True):

        reverse = ReverseSDE(hyper_params=self.hyper_params_model, method=self.method).to(self.device)
        noisy_samples = torch.randn(
            self.batch_size, self.in_channels, self.image_shape[0], self.image_shape[1]
        ).to(self.device)
        self.noise_predictor.eval()
        if self.conditional_model is not None:
            self.conditional_model.eval()

        with torch.no_grad():
            xt = noisy_samples
            for t in reversed(range(self.hyper_params_model.num_steps)):
                noise = torch.randn_like(xt) if self.method != "ode" else None
                time_steps = torch.full((self.batch_size,), t, device=self.device, dtype=torch.long)

                if self.conditional_model is not None and self.conditions is not None:
                    y = self.conditional_model(self.conditions)
                    predicted_noise = self.noise_predictor(xt, time_steps, y)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps)

                xt = reverse(xt, noise, predicted_noise, time_steps)

            generated_imgs = torch.clamp(xt, min=self.output_range[0], max=self.output_range[1])
            if normalize_output:
                generated_imgs = (generated_imgs - self.output_range[0]) / (self.output_range[1] - self.output_range[0])

        return generated_imgs