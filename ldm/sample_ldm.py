import torch
import torch.nn as nn
from transformers import BertTokenizer


class SampleLDM(nn.Module):

    def __init__(self, model, reverse_diffusion, noise_predictor, compressor_model, image_shape, conditional_model=None,
                 tokenizer="bert-base-uncased", batch_size=1, in_channels=3, device=None, max_length=77, output_range=(-1, 1)):
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.noise_predictor = noise_predictor.to(self.device)
        self.reverse = reverse_diffusion.to(self.device)
        self.compressor = compressor_model.to(self.device)
        self.conditional_model = conditional_model.to(self.device) if conditional_model else None
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer)
        self.in_channels = in_channels
        self.image_shape = image_shape
        self.batch_size = batch_size
        self.max_length = max_length
        self.output_range = output_range

        if not isinstance(image_shape, (tuple, list)) or len(image_shape) != 2 or not all(isinstance(s, int) and s > 0 for s in image_shape):
            raise ValueError("image_shape must be a tuple of two positive integers (height, width)")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if not isinstance(output_range, (tuple, list)) or len(output_range) != 2 or output_range[0] >= output_range[1]:
            raise ValueError("output_range must be a tuple (min, max) with min < max")

    def tokenize(self, prompts):

        if isinstance(prompts, str):
            prompts = [prompts]
        elif not isinstance(prompts, list) or not all(isinstance(p, str) for p in prompts):
            raise TypeError("prompts must be a string or list of strings")

        encoded = self.tokenizer(
            prompts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return encoded["input_ids"].to(self.device), encoded["attention_mask"].to(self.device)

    def forward(self, conditions=None, normalize_output=True):

        if conditions is not None and self.conditional_model is None:
            raise ValueError("Conditions provided but no conditional model specified")
        if conditions is None and self.conditional_model is not None:
            raise ValueError("Conditions must be provided for conditional model")

        noisy_samples = torch.randn(self.batch_size, self.in_channels, self.image_shape[0], self.image_shape[1]).to(self.device)

        self.noise_predictor.eval()
        self.compressor.eval()
        self.reverse.eval()
        if self.conditional_model:
            self.conditional_model.eval()

        with torch.no_grad():
            xt = noisy_samples
            xt, _ = self.compressor.encode(xt)

            if self.model == "ddim":
                num_steps = self.reverse.hyper_params.tau_num_steps
            elif self.model == "ddpm" or self.model == "sde":
                num_steps = self.reverse.hyper_params.num_steps
            else:
                raise ValueError(f"Unknown model: {self.model}. Supported: ddpm, ddim, sde")

            for t in reversed(range(num_steps)):
                time_steps = torch.full((self.batch_size,), t, device=self.device, dtype=torch.long)
                prev_time_steps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device, dtype=torch.long)

                if self.model == "sde":
                    noise = torch.randn_like(xt) if getattr(self.reverse, "method", None) != "ode" else None

                if self.conditional_model is not None and conditions is not None:
                    input_ids, attention_masks = self.tokenize(conditions)
                    key_padding_mask = (attention_masks == 0)
                    y = self.conditional_model(input_ids, key_padding_mask)
                    predicted_noise = self.noise_predictor(xt, time_steps, y)
                else:
                    predicted_noise = self.noise_predictor(xt, time_steps)

                if self.model == "sde":
                    xt = self.reverse(xt, noise, predicted_noise, time_steps)
                elif self.model == "ddim":
                    xt, _ = self.reverse(xt, predicted_noise, time_steps, prev_time_steps)
                elif self.model == "ddpm":
                    xt = self.reverse(xt, predicted_noise, time_steps)
                else:
                    raise ValueError(f"Unknown model: {self.model}. Supported: ddpm, ddim, sde")

            x = self.compressor.decode(xt)
            generated_imgs = torch.clamp(x, min=self.output_range[0], max=self.output_range[1])
            if normalize_output:
                generated_imgs = (generated_imgs - self.output_range[0]) / (self.output_range[1] - self.output_range[0])

        return generated_imgs

    def to(self, device):
        self.device = device
        self.noise_predictor.to(device)
        self.reverse.to(device)
        self.compressor.to(device)
        if self.conditional_model:
            self.conditional_model.to(device)
        return super().to(device)