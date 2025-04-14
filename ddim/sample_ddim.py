import torch
import torch.nn as nn
from reverse_ddim import ReverseDDIM



class SampleDDIM(nn.Module):
    def __init__(self, noise_predictor, hyper_params_model, image_shape, conditions=None, conditional_model=None,
                 batch_size=1, in_channels=3, device=None, output_range=(-1, 1)):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_predictor = noise_predictor.to(self.device)
        self.hyper_params_model = hyper_params_model.to(self.device)
        self.conditions = conditions.to(self.device) if isinstance(conditions, torch.Tensor) else conditions
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.in_channels = in_channels
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.output_range = output_range  # tuple (min, max) for clamping


        if not isinstance(hyper_params_model, nn.Module) or not hasattr(hyper_params_model, 'tau_num_steps'):
            raise ValueError("hyper_params_model must be a HyperParams instance with tau_num_steps")
        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if not isinstance(output_range, (tuple, list)) or len(output_range) != 2 or output_range[0] >= output_range[1]:
            raise ValueError("output_range must be a tuple (min, max) with min < max")

    def forward(self, normalize_output=True):
        reverse = ReverseDDIM(hyper_params=self.hyper_params_model).to(self.device)
        noisy_samples = torch.randn(self.batch_size, self.in_channels, self.image_shape[0], self.image_shape[1]).to(self.device)

        self.noise_predictor.eval()
        if self.conditional_model:
            self.conditional_model.eval()

        with torch.no_grad():
            xt = noisy_samples
            for t in reversed(range(self.hyper_params_model.tau_num_steps)):
                time_steps = torch.full((self.batch_size,), t, device=self.device, dtype=torch.long)
                prev_time_steps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device, dtype=torch.long)

                if self.conditional_model:
                    if self.conditions is None:
                        raise ValueError("Conditions must be provided for conditional model")
                    y = self.conditional_model(self.conditions)
                    predicted_noise = self.noise_predictor(xt, time_steps, y)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps)

                xt, _ = reverse(xt, predicted_noise, time_steps, prev_time_steps)

            generated_imgs = torch.clamp(xt, min=self.output_range[0], max=self.output_range[1])
            if normalize_output:
                generated_imgs = (generated_imgs - self.output_range[0]) / (self.output_range[1] - self.output_range[0])

        return generated_imgs