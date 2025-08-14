import torch
import torch.nn as nn
from typing import Optional, Union, List, Tuple

class SampleUnCLIP(nn.Module):
    def __init__(
            self,
            prior_model: nn.Module,
            decoder_model: nn.Module,
            clip_model: nn.Module,
            first_upsampler_model: nn.Module,
            second_upsampler_model: Optional[nn.Module],
            device: Optional[Union[torch.device, str]],
            prior_guidance_scale: float = 4.0,
            batch_size: int = 1,
            normalize: bool = True,
            reduce_dim: bool = True,
            image_size: Tuple[int, int, int] = (3, 64, 64)


    ) -> None:

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

        self.prior_model = prior_model.to(self.device)
        self.decoder_model = decoder_model.to(self.device)
        self.clip_model = clip_model.to(self.device)
        self.first_upsampler_model = first_upsampler_model.to(self.device)
        self.second_upsampler_model = second_upsampler_model.to(self.device) if second_upsampler_model else None

        self.prior_guidance_scale = prior_guidance_scale
        self.batch_size = batch_size
        self.normalize = normalize
        self.reduce_dim = reduce_dim
        self.image_size = image_size

        super().__init__()

    def forward(self, prompts: Optional[Union[str, List]] = None):

        noise = torch.randn((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]), device=self.device)

        with torch.no_grad():

            # embedd the input prompt using prior model
            prompt_embeddings = self.clip_model(data=prompts, data_type="text", normalize=self.normalize)
            image_embeddings = self.clip_model(data=noise, data_type="img", normalize=self.normalize)
            if self.reduce_dim:
                prompt_embeddings = self.prior_model.text_projection(prompt_embeddings)
            for t in reversed(range(self.prior_model.forward_diffusion.variance_scheduler.tau_num_steps)):
                time_steps = torch.full((self.batch_size,), t, device=self.device)
                prev_time_steps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)
                prev_image_embeddings = image_embeddings
                predicted_embeddings = self.prior_model(prompt_embeddings, image_embeddings, time_steps)
                if self.reduce_dim:
                    predicted_embeddings = self.prior_model.image_projection.inverse_transform(predicted_embeddings)
                # compute guided prediction
                guided_embeddings = self.compute_prior_guided_prediction(predicted_embeddings, prompt_embeddings, image_embeddings, time_steps)
                image_embeddings, _ = self.prior_model.reverse_diffusion(prev_image_embeddings, guided_embeddings, time_steps, prev_time_steps)

            # Decode the output of prior model (embedd input prompts)
            decoder_noise = torch.randn((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]), device=self.device)
            # project z_i to 4 tokens
            c = self.decoder_model.decoder_projection(image_embeddings)
            # encode text with GLIDE
            encoded_prompts = self.decoder_model._encode_text_with_glide(prompts)
            # concatenate embeddings
            context = self.decoder_model._concatenate_embeddings(encoded_prompts, c)
            for t in reversed(range(self.decoder_model_model.forward_diffusion.variance_scheduler.tau_num_steps)):
                time_steps = torch.full((self.batch_size,), t, device=self.device)
                prev_time_steps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)
                prev_image_embeddings = image_embeddings
                # noise predictor internally concatenate time_steps and context
                predicted_noise = self.decoder_model.noise_predictor(decoder_noise, time_steps, None, context)
                guided_embeddings = self.compute_decoder_guided_prediction(predicted_noise, decoder_noise, time_steps, context)
                image_embeddings, _ = self.decoder_model.reverse_diffusion(prev_image_embeddings, guided_embeddings, time_steps, prev_time_steps)

            # implement upsamplers !

            pass


    def compute_prior_guided_prediction(self, predicted_embeddings: torch.Tensor, prompt_embeddings: torch.Tensor, image_embeddings: torch.Tensor, time_steps: torch.Tensor):

        zero_image_embeddings = torch.zeros_like(image_embeddings)

        return (1.0 + self.prior_guidance_scale) * predicted_embeddings - self.prior_guidance_scale * self.prior_model(prompt_embeddings, zero_image_embeddings, time_steps)

    def compute_decoder_guided_prediction(self, predicted_noise: torch.Tensor, decoder_noise: torch.Tensor, time_steps: torch.Tensor, context: torch.Tensor):

        zero_context = torch.zeros_like(context)

        return (1.0 + self.decoder_guidance_scale) * predicted_noise - self.decoder_guidance_scale * self.decoder_model.noise_predictor(decoder_noise, time_steps, None, zero_context)

