import torch
import torch.nn as nn
import torchvision
from typing import Optional, Union, List, Tuple
import os


class SampleUnCLIP(nn.Module):
    def __init__(
            self,
            prior_model: nn.Module,
            decoder_model: nn.Module,
            clip_model: nn.Module,
            first_upsampler_model: nn.Module,
            second_upsampler_model: Optional[nn.Module] = None,
            device: Optional[Union[torch.device, str]] = None,
            prior_guidance_scale: float = 4.0,
            decoder_guidance_scale: float = 8.0,
            batch_size: int = 1,
            normalize: bool = True,
            reduce_dim: bool = True,
            image_size: Tuple[int, int, int] = (3, 64, 64),
            use_second_upsampler: bool = True,
            output_range: Tuple[float, float] = (-1.0, 1.0)
    ) -> None:
        super().__init__()

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
        self.decoder_guidance_scale = decoder_guidance_scale
        self.batch_size = batch_size
        self.normalize = normalize
        self.reduce_dim = reduce_dim
        self.image_size = image_size
        self.use_second_upsampler = use_second_upsampler
        self.output_range = output_range
        self.images_256 = None
        self.images_1024 = None

    def forward(self, prompts: Optional[Union[str, List]] = None, normalize_output: bool = True,
                save_images: bool = True, save_path: str = "unclip_generated"):

        # initialize noise for prior sampling (image embedding space)
        embedding_noise = torch.randn((self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]),
                                      device=self.device)

        with torch.no_grad():
            # ====== PRIOR STAGE: Generate image embeddings from text ======
            # encode text prompt using CLIP
            text_embeddings = self.clip_model(data=prompts, data_type="text", normalize=self.normalize)
            current_embeddings = self.clip_model(data=embedding_noise, data_type="img", normalize=self.normalize)

            # Optionally reduce dimensionality
            if self.reduce_dim:
                text_embeddings = self.prior_model.text_projection(text_embeddings)

            # prior diffusion sampling loop
            for t in reversed(range(self.prior_model.forward_diffusion.variance_scheduler.tau_num_steps)):
                timesteps = torch.full((self.batch_size,), t, device=self.device)
                prev_timesteps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)

                # predict embeddings
                predicted_embeddings = self.prior_model(text_embeddings, current_embeddings, timesteps)

                if self.reduce_dim:
                    predicted_embeddings = self.prior_model.image_projection.inverse_transform(predicted_embeddings)

                # apply guidance
                guided_embeddings = self.compute_prior_guided_prediction(
                    predicted_embeddings, text_embeddings, current_embeddings, timesteps
                )

                # update embeddings using reverse diffusion
                current_embeddings, _ = self.prior_model.reverse_diffusion(
                    current_embeddings, guided_embeddings, timesteps, prev_timesteps
                )

            final_image_embeddings = current_embeddings

            # ====== DECODER STAGE: Generate 64x64 images from embeddings ======
            # initialize noise for decoder sampling
            decoder_noise = torch.randn(
                (self.batch_size, self.image_size[0], self.image_size[1], self.image_size[2]),
                device=self.device
            )

            # project image embeddings to 4 tokens
            projected_embeddings = self.decoder_model.decoder_projection(final_image_embeddings)

            # encode text with GLIDE
            glide_text_embeddings = self.decoder_model._encode_text_with_glide(prompts)

            # concatenate embeddings for context
            context = self.decoder_model._concatenate_embeddings(glide_text_embeddings, projected_embeddings)

            current_images = decoder_noise
            # decoder diffusion sampling loop
            for t in reversed(range(self.decoder_model.forward_diffusion.variance_scheduler.tau_num_steps)):
                timesteps = torch.full((self.batch_size,), t, device=self.device)
                prev_timesteps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)

                # predict noise
                predicted_noise = self.decoder_model.noise_predictor(current_images, timesteps, None, context)

                # apply guidance
                guided_noise = self.compute_decoder_guided_prediction(
                    predicted_noise, current_images, timesteps, context
                )

                # update images using reverse diffusion
                current_images, _ = self.decoder_model.reverse_diffusion(
                    current_images, guided_noise, timesteps, prev_timesteps
                )

            generated_64x64 = current_images

            # ====== FIRST UPSAMPLER: 64x64 -> 256x256 ======
            upsampled_256_noise = torch.randn((self.batch_size, self.image_size[0], 256, 256), device=self.device)
            current_256_images = upsampled_256_noise

            for t in reversed(range(self.first_upsampler_model.forward_diffusion.variance_scheduler.tau_num_steps)):
                timesteps = torch.full((self.batch_size,), t, device=self.device)
                prev_timesteps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)

                # predict noise for upsampling (conditioned on low-res image)
                predicted_noise = self.first_upsampler_model(current_256_images, timesteps, generated_64x64)

                # update using reverse diffusion
                current_256_images, _ = self.first_upsampler_model.reverse_diffusion(
                    current_256_images, predicted_noise, timesteps, prev_timesteps
                )

            self.images_256 = current_256_images

            # ====== SECOND UPSAMPLER: 256x256 -> 1024x1024 (if enabled) ======
            if self.use_second_upsampler and self.second_upsampler_model:
                upsampled_1024_noise = torch.randn((self.batch_size, self.image_size[0], 1024, 1024),
                                                   device=self.device)
                current_1024_images = upsampled_1024_noise

                for t in reversed(
                        range(self.second_upsampler_model.forward_diffusion.variance_scheduler.tau_num_steps)):
                    timesteps = torch.full((self.batch_size,), t, device=self.device)
                    prev_timesteps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)

                    # predict noise for upsampling (conditioned on 256x256 image)
                    predicted_noise = self.second_upsampler_model(current_1024_images, timesteps, self.images_256)

                    # update using reverse diffusion
                    current_1024_images, _ = self.second_upsampler_model.reverse_diffusion(
                        current_1024_images, predicted_noise, timesteps, prev_timesteps
                    )

                self.images_1024 = current_1024_images

            # ====== POST-PROCESSING ======
            # normalize output to [0, 1] range if requested
            if normalize_output:
                final_256 = (self.images_256 - self.output_range[0]) / (self.output_range[1] - self.output_range[0])
                final_1024 = None
                if self.images_1024 is not None:
                    final_1024 = (self.images_1024 - self.output_range[0]) / (
                                self.output_range[1] - self.output_range[0])
            else:
                final_256 = self.images_256
                final_1024 = self.images_1024

            # save images if requested
            if save_images:
                os.makedirs(save_path, exist_ok=True)
                os.makedirs(os.path.join(save_path, "images_256"), exist_ok=True)
                if final_1024 is not None:
                    os.makedirs(os.path.join(save_path, "images_1024"), exist_ok=True)

                for i in range(self.batch_size):
                    img_path_256 = os.path.join(save_path, "images_256", f"image_{i}.png")
                    torchvision.utils.save_image(final_256[i], img_path_256)

                    if final_1024 is not None:
                        img_path_1024 = os.path.join(save_path, "images_1024", f"image_{i}.png")
                        torchvision.utils.save_image(final_1024[i], img_path_1024)

        # return final images
        if final_1024 is not None:
            return final_1024
        else:
            return final_256

    def compute_prior_guided_prediction(
            self,
            predicted_embeddings: torch.Tensor,
            text_embeddings: torch.Tensor,
            current_embeddings: torch.Tensor,
            timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Compute classifier-free guidance for prior model."""
        zero_embeddings = torch.zeros_like(current_embeddings)
        unconditioned_pred = self.prior_model(text_embeddings, zero_embeddings, timesteps)

        # CFG formula: (1 + guidance_scale) * conditioned - guidance_scale * unconditioned
        return (1.0 + self.prior_guidance_scale) * predicted_embeddings - self.prior_guidance_scale * unconditioned_pred

    def compute_decoder_guided_prediction(
            self,
            predicted_noise: torch.Tensor,
            current_images: torch.Tensor,
            timesteps: torch.Tensor,
            context: torch.Tensor
    ) -> torch.Tensor:
        """Compute classifier-free guidance for decoder model."""
        zero_context = torch.zeros_like(context)
        unconditioned_noise = self.decoder_model.noise_predictor(current_images, timesteps, None, zero_context)

        # CFG formula: (1 + guidance_scale) * conditioned - guidance_scale * unconditioned
        return (1.0 + self.decoder_guidance_scale) * predicted_noise - self.decoder_guidance_scale * unconditioned_noise