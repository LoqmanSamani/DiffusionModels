import torch
import torch.nn as nn
import torchvision
from typing import Optional, Union, List, Tuple, Self
import os


class SampleUnCLIP(nn.Module):
    """Generates images using the UnCLIP model pipeline.

    Combines a prior model, decoder model, CLIP model, and upsampler models to generate
    images from text prompts or noise. Performs diffusion-based sampling with classifier-free
    guidance in both prior and decoder stages, followed by upsampling to higher resolutions.

    Parameters
    ----------
    `prior_model` : nn.Module
        The UnCLIP prior model for generating image embeddings from text.
    `decoder_model` : nn.Module
        The UnCLIP decoder model for generating low-resolution images from embeddings.
    `clip_model` : nn.Module
        CLIP model for encoding text prompts into embeddings.
    `low_res_upsampler` : nn.Module
        First upsampler model for scaling images from 64x64 to 256x256.
    `high_res_upsampler` : nn.Module, optional
        Second upsampler model for scaling images from 256x256 to 1024x1024, default None.
    `device` : Union[torch.device, str], optional
        Device for computation (default: CUDA if available, else CPU).
    `clip_embedding_dim` : int, optional
        Dimensionality of CLIP embeddings (default: 512).
    `prior_guidance_scale` : float, optional
        Classifier-free guidance scale for the prior model (default: 4.0).
    `decoder_guidance_scale` : float, optional
        Classifier-free guidance scale for the decoder model (default: 8.0).
    `batch_size` : int, optional
        Number of images to generate per batch (default: 1).
    `normalize` : bool, optional
        Whether to normalize CLIP embeddings (default: True).
    `prior_dim_reduction` : bool, optional
        Whether to apply dimensionality reduction in the prior model (default: True).
    `image_size` : Tuple[int, int, int], optional
        Size of the initial generated images (default: (3, 64, 64) for RGB 64x64).
    `use_high_res_upsampler` : bool, optional
        Whether to use the second upsampler for 1024x1024 output (default: True).
    `image_output_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    """
    def __init__(
            self,
            prior_model: nn.Module,
            decoder_model: nn.Module,
            clip_model: nn.Module,
            low_res_upsampler: nn.Module,
            high_res_upsampler: Optional[nn.Module] = None,
            device: Optional[Union[torch.device, str]] = None,
            clip_embedding_dim: int = 512,  # CLIP embedding dimension
            prior_guidance_scale: float = 4.0,
            decoder_guidance_scale: float = 8.0,
            batch_size: int = 1,
            normalize_clip_embeddings: bool = True,
            prior_dim_reduction: bool = True,
            initial_image_size: Tuple[int, int, int] = (3, 64, 64),
            use_high_res_upsampler: bool = True,
            image_output_range: Tuple[float, float] = (-1.0, 1.0)
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
        self.low_res_upsampler = low_res_upsampler.to(self.device)
        self.high_res_upsampler = high_res_upsampler.to(self.device) if high_res_upsampler else None

        self.prior_guidance_scale = prior_guidance_scale
        self.decoder_guidance_scale = decoder_guidance_scale
        self.batch_size = batch_size
        self.normalize_clip_embeddings = normalize_clip_embeddings
        self.prior_dim_reduction = prior_dim_reduction
        self.clip_embedding_dim = clip_embedding_dim
        self.initial_image_size = initial_image_size
        self.use_high_res_upsampler = use_high_res_upsampler
        self.image_output_range = image_output_range
        self.images_256 = None
        self.images_1024 = None

    def forward(
            self,
            prompts: Optional[Union[str, List]] = None,
            normalize_output: bool = True,
                save_images: bool = True,
            save_path: str = "unclip_generated"
    ):
        """Generates images from text prompts or noise using the UnCLIP pipeline.

        Executes the full UnCLIP generation process: prior model generates image embeddings,
        decoder model generates 64x64 images, first upsampler scales to 256x256, and optional
        second upsampler scales to 1024x1024. Supports classifier-free guidance and saves
        generated images if requested.

        Parameters
        ----------
        `prompts` : Union[str, List], optional
            Text prompt(s) for conditional generation, default None (unconditional).
        `normalize_output` : bool, optional
            Whether to normalize output images to [0, 1] range (default: True).
        `save_images` : bool, optional
            Whether to save generated images to disk (default: True).
        `save_path` : str, optional
            Directory to save generated images (default: "unclip_generated").

        Returns
        -------
        final_images : torch.Tensor
            Generated images, shape (batch_size, channels, height, width), either 256x256
            or 1024x1024 depending on use_second_upsampler.
        """
        # initialize noise for prior sampling (image embedding space)
        embedding_noise = torch.randn((self.batch_size, self.clip_embedding_dim), device=self.device)
        print("embedding noise: ", embedding_noise.size())

        with torch.no_grad():
            # ====== PRIOR STAGE: generate image embeddings from text ======
            print("############################################################")
            print("                           prior model                      ")
            print("############################################################")
            # encode text prompt using CLIP
            text_embeddings = self.clip_model(data=prompts, data_type="text", normalize=self.normalize_clip_embeddings)
            print("text embedding : ", text_embeddings.size())

            current_embeddings = embedding_noise.clone()

            # optionally reduce dimensionality for prior model
            if self.prior_dim_reduction:
                text_embeddings_reduced = self.prior_model.text_projection(text_embeddings)
                current_embeddings_reduced = self.prior_model.image_projection(current_embeddings)
                print("text embedding reduced: ", text_embeddings_reduced.size())
                print("current embedding reduced: ", current_embeddings_reduced.size())
            else:
                text_embeddings_reduced = text_embeddings
                current_embeddings_reduced = current_embeddings
                print("text embedding reduced: ", text_embeddings_reduced.size())
                print("current embedding reduced: ", current_embeddings_reduced.size())

            # prior diffusion sampling loop
            t_counter = 0
            for t in reversed(range(self.prior_model.forward_diffusion.variance_scheduler.tau_num_steps)):
                timesteps = torch.full((self.batch_size,), t, device=self.device)
                prev_timesteps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)

                # predict embeddings
                predicted_embeddings = self.prior_model(text_embeddings_reduced, current_embeddings_reduced, timesteps)
                if t == 10:
                    print("predicted embeddings: ", predicted_embeddings.size())

                # apply guidance
                guided_embeddings = self.compute_prior_guided_prediction(
                    predicted_embeddings, text_embeddings_reduced, current_embeddings_reduced, timesteps
                )
                if t == 10:
                    print("guided embeddings: ", guided_embeddings.size())

                # update embeddings using reverse diffusion
                current_embeddings_reduced, _ = self.prior_model.reverse_diffusion(
                    current_embeddings_reduced, guided_embeddings, timesteps, prev_timesteps
                )
                if t == 10:
                    print("current embedding reduced: ", current_embeddings_reduced.size())

            # convert back to full embedding dimension if needed
            if self.prior_dim_reduction:
                final_image_embeddings = self.prior_model.image_projection.inverse_transform(current_embeddings_reduced)
                print("final image embeddings: ", final_image_embeddings.size())
            else:
                final_image_embeddings = current_embeddings_reduced
                print("final image embeddings: ", final_image_embeddings.size())

                t_counter += 1
            print("number of iters in prior model: ", t_counter)

            # ====== DECODER STAGE: generate 64x64 images from embeddings ======

            print("############################################################")
            print("                         decoder model                      ")
            print("############################################################")

            # initialize noise for decoder sampling
            decoder_noise = torch.randn((self.batch_size, self.initial_image_size[0], self.initial_image_size[1], self.initial_image_size[2]), device=self.device)
            print("decoder noise: ", decoder_noise.size())

            # project image embeddings to 4 tokens
            projected_embeddings = self.decoder_model.decoder_projection(final_image_embeddings)
            print("projected embeddings: ", projected_embeddings.size())

            # encode text with GLIDE/decoder's text encoder
            glide_text_embeddings = self.decoder_model._encode_text_with_glide(prompts)
            print("glide text embeddings: ", glide_text_embeddings.size())

            # concatenate embeddings for context
            context = self.decoder_model._concatenate_embeddings(glide_text_embeddings, projected_embeddings)
            print("context: ", context.size())

            current_images = decoder_noise
            # decoder diffusion sampling loop
            t_counter = 0
            for t in reversed(range(self.decoder_model.forward_diffusion.variance_scheduler.tau_num_steps)):
                timesteps = torch.full((self.batch_size,), t, device=self.device)
                prev_timesteps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)

                # Predict noise
                predicted_noise = self.decoder_model.noise_predictor(current_images, timesteps, context, None)
                if t ==  10:
                    print("predicted noise: ", predicted_noise.size())

                # apply guidance
                guided_noise = self.compute_decoder_guided_prediction(
                    predicted_noise, current_images, timesteps, context
                )
                if t == 10:
                    print("guided noise: ", guided_noise.size())

                # update images using reverse diffusion
                current_images, _ = self.decoder_model.reverse_diffusion(
                    current_images, guided_noise, timesteps, prev_timesteps
                )
                if t == 10:
                    print("current image: ", current_images.size())
                t_counter += 1

            generated_64x64 = current_images
            print("  number of iters of decoder model: ", t_counter)

            # ====== FIRST UPSAMPLER: 64x64 -> 256x256 ======
            print("############################################################")
            print("                         first upsampler                      ")
            print("############################################################")
            upsampled_256_noise = torch.randn((self.batch_size, self.initial_image_size[0], 256, 256), device=self.device)
            current_256_images = upsampled_256_noise
            print("upsampled 256 noise: ", upsampled_256_noise.size())

            t_counter = 0
            for t in reversed(range(self.low_res_upsampler.forward_diffusion.variance_scheduler.tau_num_steps)):
                timesteps = torch.full((self.batch_size,), t, device=self.device)
                prev_timesteps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)

                # predict noise for upsampling (conditioned on low-res image)
                predicted_noise = self.low_res_upsampler(current_256_images, timesteps, generated_64x64)
                if t == 10:
                    print("predicted noise: ", predicted_noise.size())

                # update using reverse diffusion
                current_256_images, _ = self.low_res_upsampler.reverse_diffusion(
                    current_256_images, predicted_noise, timesteps, prev_timesteps
                )
                if t == 10:
                    print("current 256 images: ", current_256_images.size())
                t_counter += 1
            print("number of iters in upsampler one:", t_counter)

            self.images_256 = current_256_images

            # ====== SECOND UPSAMPLER: 256x256 -> 1024x1024 (if enabled) ======
            print("############################################################")
            print("                         second upsampler                   ")
            print("############################################################")
            if self.use_high_res_upsampler and self.high_res_upsampler:
                upsampled_1024_noise = torch.randn((self.batch_size, self.initial_image_size[0], 1024, 1024), device=self.device)
                current_1024_images = upsampled_1024_noise

                t_counter = 0
                for t in reversed(range(self.high_res_upsampler.forward_diffusion.variance_scheduler.tau_num_steps)):
                    timesteps = torch.full((self.batch_size,), t, device=self.device)
                    prev_timesteps = torch.full((self.batch_size,), max(t - 1, 0), device=self.device)

                    # predict noise for upsampling (conditioned on 256x256 image)
                    predicted_noise = self.high_res_upsampler(current_1024_images, timesteps, self.images_256)
                    if t == 10:
                        print("predicted noise: ", predicted_noise.size())

                    # update using reverse diffusion
                    current_1024_images, _ = self.high_res_upsampler.reverse_diffusion(
                        current_1024_images, predicted_noise, timesteps, prev_timesteps
                    )
                    if t == 10:
                        print("current 1024 images: ", current_1024_images.size())
                    t_counter += 1
                print("number of iters in upsampler two:", t_counter)

                self.images_1024 = current_1024_images

            # ====== POST-PROCESSING ======
            # normalize output to [0, 1] range if requested
            if normalize_output:
                final_256 = (self.images_256 - self.image_output_range[0]) / (self.image_output_range[1] - self.image_output_range[0])
                final_1024 = None
                if self.images_1024 is not None:
                    final_1024 = (self.images_1024 - self.image_output_range[0]) / (
                            self.image_output_range[1] - self.image_output_range[0])
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
        """Computes classifier-free guidance for the prior model.

        Combines conditioned and unconditioned predictions using the classifier-free guidance
        formula to enhance the quality of generated image embeddings.

        Parameters
        ----------
        `predicted_embeddings` : torch.Tensor
            Conditioned predicted embeddings, shape (batch_size, embedding_dim).
        `text_embeddings` : torch.Tensor
            Text embeddings from CLIP, shape (batch_size, embedding_dim).
        `current_embeddings` : torch.Tensor
            Current noisy embeddings, shape (batch_size, embedding_dim).
        `timesteps` : torch.Tensor
            Timestep indices, shape (batch_size,).

        Returns
        -------
        guided_embeddings : torch.Tensor
            Guided embeddings, shape (batch_size, embedding_dim).
        """
        # use zero embeddings for unconditional generation
        zero_text_embeddings = torch.zeros_like(text_embeddings)
        unconditioned_pred = self.prior_model(zero_text_embeddings, current_embeddings, timesteps)

        # CFG formula: (1 + guidance_scale) * conditioned - guidance_scale * unconditioned
        return (1.0 + self.prior_guidance_scale) * predicted_embeddings - self.prior_guidance_scale * unconditioned_pred

    def compute_decoder_guided_prediction(
            self,
            predicted_noise: torch.Tensor,
            current_images: torch.Tensor,
            timesteps: torch.Tensor,
            context: torch.Tensor
    ) -> torch.Tensor:
        """Computes classifier-free guidance for the decoder model.

        Combines conditioned and unconditioned noise predictions using the classifier-free
        guidance formula to enhance the quality of generated images.

        Parameters
        ----------
        `predicted_noise` : torch.Tensor
            Conditioned predicted noise, shape (batch_size, channels, height, width).
        `current_images` : torch.Tensor
            Current noisy images, shape (batch_size, channels, height, width).
        `timesteps` : torch.Tensor
            Timestep indices, shape (batch_size,).
        `context` : torch.Tensor
            Context embeddings (concatenated GLIDE text and projected image embeddings),
            shape (batch_size, seq_len, embedding_dim).

        Returns
        -------
        guided_noise : torch.Tensor
            Guided noise prediction, shape (batch_size, channels, height, width).
        """
        zero_context = torch.zeros_like(context)
        unconditioned_noise = self.decoder_model.noise_predictor(current_images, timesteps, zero_context, None)

        # CFG formula: (1 + guidance_scale) * conditioned - guidance_scale * unconditioned
        return (1.0 + self.decoder_guidance_scale) * predicted_noise - self.decoder_guidance_scale * unconditioned_noise

    def to(self, device: Union[torch.device, str]) -> Self:
        """Moves the module and all its components to the specified device.

        Updates the device attribute and moves all sub-models (prior, decoder, CLIP,
        and upsamplers) to the specified device.

        Parameters
        ----------
        device : Union[torch.device, str]
            Target device for the module and its components.

        Returns
        -------
        SampleUnCLIP
            The module moved to the specified device.
        """
        if isinstance(device, str):
            device = torch.device(device)

        self.device = device

        # move all sub-models to the specified device
        self.prior_model.to(device)
        self.decoder_model.to(device)
        self.clip_model.to(device)
        self.low_res_upsampler.to(device)

        if self.high_res_upsampler is not None:
            self.high_res_upsampler.to(device)

        return super().to(device)


"""
from prior_model import UnCLIPTransformerPrior
from utils import NoisePredictor, TextEncoder
from clip_model import CLIPEncoder
from project_prior import Projection
import torch
from prior_diff import VarianceSchedulerUnCLIP, ForwardUnCLIP, ReverseUnCLIP
from decoder_model import UnClipDecoder
from upsampler import UpsamplerUnCLIP

device = torch.device("cuda")


h_model = VarianceSchedulerUnCLIP(
    num_steps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=True,
    beta_method="cosine"
).to(device)

c_model = CLIPEncoder(model_name="openai/clip-vit-base-patch32").to(device)
tp = Projection(
    input_dim=512,
    output_dim=320,
    hidden_dim=480,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
).to(device)
ip = Projection(
    input_dim=512,
    output_dim=320,
    hidden_dim=480,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
).to(device)

d_model = ForwardUnCLIP(h_model).to(device)
r_model = ReverseUnCLIP(h_model).to(device)

prior_model = UnCLIPTransformerPrior(
    forward_diffusion=d_model,
    reverse_diffusion=r_model,
    text_projection=tp,
    image_projection=ip,
    embedding_dim=320,
    num_layers=12,
    num_attention_heads=8,
    feedforward_dim=512,
    max_sequence_length=2,
    dropout_rate=0.3
).to(device)


dn_model = NoisePredictor(
        in_channels=3,
        down_channels=[16, 32],
        mid_channels=[32, 32],
        up_channels=[32, 16],
        down_sampling=[True, True],
        time_embed_dim=512,
        y_embed_dim=512,
        num_down_blocks=2,
        num_mid_blocks=2,
        num_up_blocks=2,
        down_sampling_factor=2
).to(device)

dt_proj = Projection(
    input_dim=512,
    output_dim=320,
    hidden_dim=468,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
).to(device)
di_proj = Projection(
    input_dim=512,
    output_dim=320,
    hidden_dim=468,
    num_layers=2,
    dropout=0.1,
    use_layer_norm=True
).to(device)

dh_model = VarianceSchedulerUnCLIP(
    num_steps=500,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=False,
    beta_method="linear"
).to(device)
dfor_ = ForwardUnCLIP(h_model).to(device)
drev_ = ReverseUnCLIP(h_model).to(device)

dcond = TextEncoder(
    use_pretrained_model=True,
    model_name="bert-base-uncased",
    vocabulary_size=30522,
    num_layers=2,
    input_dimension=512,
    output_dimension=512,
    num_heads=2,
    context_length=77
).to(device)

decoder_model = UnClipDecoder(
    embedding_dim=512,
    noise_predictor=dn_model,
    forward_diffusion=dfor_,
    reverse_diffusion=drev_,
    conditional_model=dcond,
    tokenizer=None,
    device="cuda",
    output_range=(-1.0, 1.0),
    normalize=True,
    classifier_free=0.1,
    drop_caption=0.5,
    max_length=77
).to(device)


hyp = VarianceSchedulerUnCLIP(
    num_steps=1000,
    beta_start=1e-4,
    beta_end=0.02,
    trainable_beta=False,
    beta_method="cosine"
).to(device)




up_for = ForwardUnCLIP(hyp).to(device)
up_rev = ReverseUnCLIP(hyp).to(device)

upsampler_model_first = UpsamplerUnCLIP(
    forward_diffusion=up_for,
    reverse_diffusion=up_rev,
    in_channels= 3,
    out_channels= 3,
    model_channels= 32,
    num_res_blocks = 2,
    channel_mult = (1, 2, 4, 8),
    dropout = 0.1,
    time_embed_dim  = 756,
    low_res_size = 64,
    high_res_size = 256
).to(device)

upsampler_model_second = UpsamplerUnCLIP(
    forward_diffusion=up_for,
    reverse_diffusion=up_rev,
    in_channels= 3,
    out_channels= 3,
    model_channels= 32,
    num_res_blocks = 2,
    channel_mult = (1, 2, 4, 8),
    dropout = 0.1,
    time_embed_dim  = 756,
    low_res_size = 256,
    high_res_size = 1024
).to(device)



sampler = SampleUnCLIP(
    prior_model=prior_model,
    decoder_model=decoder_model,
    clip_model=c_model,
    first_upsampler_model=upsampler_model_first,
    second_upsampler_model=upsampler_model_second,
    device=None,
    prior_guidance_scale=4.0,
    decoder_guidance_scale=8.0,
    batch_size=1,
    normalize=True,
    reduce_dim=True,
    embedding_dim=512,
    image_size=(3, 64, 64),
    use_second_upsampler=True,
    output_range=(-1.0, 1.0)
).to(device)

f = sampler(
    prompts = ["this is a test prompt"],
    normalize_output = True,
    save_images = True,
    save_path = "unclip_generated"
)
"""

