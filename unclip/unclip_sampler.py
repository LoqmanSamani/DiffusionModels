import torch
import torch.nn as nn
import torchvision
from typing import Optional, Union, List, Tuple
from tqdm.auto import tqdm
import os



class SampleUnCLIP(nn.Module):
    """Generates images using the UnCLIP model pipeline.

    Combines a prior model, decoder model, CLIP model, and upsampler models to generate
    images from text prompts or noise. Performs diffusion-based sampling with classifier-free
    guidance in both prior and decoder stages, followed by upsampling to higher resolutions.

    Parameters
    ----------
    `prior_net` : nn.Module
        The UnCLIP prior model for generating image embeddings from text.
    `decoder_net` : nn.Module
        The UnCLIP decoder model for generating low-resolution images from embeddings.
    `clip_net` : nn.Module
        CLIP model for encoding text prompts into embeddings.
    `low_res_upsampler` : nn.Module
        First upsampler model for scaling images from 64x64 to 256x256.
    `high_res_upsampler` : nn.Module, optional
        Second upsampler model for scaling images from 256x256 to 1024x1024, default None.
    `device` : str, optional
        Device for computation (default: CUDA).
    `offload_device`: str
        Device for offloading (default: CPU)
    `clip_embed_dim` : int, optional
        Dimensionality of CLIP embeddings (default: 512).
    `prior_guidance_scale` : float, optional
        Classifier-free guidance scale for the prior model (default: 4.0).
    `decoder_guidance_scale` : float, optional
        Classifier-free guidance scale for the decoder model (default: 8.0).
    `batch_size` : int, optional
        Number of images to generate per batch (default: 1).
    `norm_clip_embed` : bool, optional
        Whether to normalize CLIP embeddings (default: True).
    `prior_dim_reduction` : bool, optional
        Whether to apply dimensionality reduction in the prior model (default: True).
    `init_img_size` : Tuple[int, int, int], optional
        Size of the initial generated images (default: (3, 64, 64) for RGB 64x64).
    `use_high_res_upsampler` : bool, optional
        Whether to use the second upsampler for 1024x1024 output (default: True).
    `norm_range` : Tuple[float, float], optional
        Range for clamping output images (default: (-1.0, 1.0)).
    `use_model_offloading`: bool
        Whether model offloading is used (default: True)
    """
    def __init__(
            self,
            prior_net: nn.Module,
            decoder_net: nn.Module,
            clip_net: nn.Module,
            low_res_upsampler: nn.Module,
            high_res_upsampler: Optional[nn.Module] = None,
            device: str = 'cuda',
            offload_device: str = 'cpu',
            clip_embed_dim: int = 512,
            prior_guidance_scale: float = 4.0,
            decoder_guidance_scale: float = 8.0,
            batch_size: int = 1,
            norm_clip_embed: bool = True,
            prior_dim_reduction: bool = True,
            init_img_size: Tuple[int, int, int] = (3, 64, 64),
            use_high_res_upsampler: bool = True,
            norm_range: Tuple[float, float] = (-1.0, 1.0),
            use_model_offloading: bool = True,
            *args
    ) -> None:
        super().__init__()

        self.device = torch.device(device) if isinstance(device, str) else device
        self.offload_device = torch.device(offload_device)
        self.use_model_offloading = use_model_offloading

        # keep models on CPU initially if offloading is enabled
        init_device = self.offload_device if use_model_offloading else self.device

        self.prior_net = prior_net.to(init_device).eval()
        self.decoder_net = decoder_net.to(init_device).eval()
        self.clip_net = clip_net.to(init_device).eval()
        self.low_res_upsampler = low_res_upsampler.to(init_device).eval()
        self.high_res_upsampler = high_res_upsampler.to(init_device).eval() if high_res_upsampler else None

        self.prior_guidance_scale = prior_guidance_scale
        self.decoder_guidance_scale = decoder_guidance_scale
        self.batch_size = batch_size
        self.norm_clip_embed = norm_clip_embed
        self.prior_dim_reduction = prior_dim_reduction
        self.clip_embed_dim = clip_embed_dim
        self.init_img_size = init_img_size
        self.use_high_res_upsampler = use_high_res_upsampler
        self.norm_range = norm_range
        self.imgs_256 = None
        self.imgs_1024 = None

    def _move_model_to_device(self, model: nn.Module, target_device: torch.device):
        """Helper to move model to device if offloading is enabled."""
        if self.use_model_offloading:
            model.to(target_device)
            if target_device == self.device:
                torch.cuda.empty_cache()

    @torch.no_grad()
    def forward(
            self,
            prompts: Optional[Union[str, List]] = None,
            norm_output: bool = True,
            save_imgs: bool = True,
            save_path: str = "unclip_samples"
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
        `norm_output` : bool, optional
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
        # ====== PRIOR STAGE: generate image embeddings from text ======
        self._move_model_to_device(self.clip_net, self.device)
        self._move_model_to_device(self.prior_net, self.device)
        # encode text prompt using CLIP
        txt_embed = self.clip_net(data=prompts, data_type="text", normalize=self.norm_clip_embed)
        # free CLIP immediately after use
        self._move_model_to_device(self.clip_net, self.offload_device)
        # initialize noise for prior sampling
        embed_noise = torch.randn((self.batch_size, self.clip_embed_dim), device=self.device)
        curr_embed = embed_noise
        if self.prior_dim_reduction:
            txt_embed_reduced = self.prior_net.clip_text_proj(txt_embed)
            curr_embed_reduced = self.prior_net.clip_img_proj(curr_embed)
        else:
            txt_embed_reduced = txt_embed
            curr_embed_reduced = curr_embed
        # prior diffusion sampling with batched CFG
        timesteps = self.decoder_net.fwd_unclip.vs.inference_timesteps.flip(0)
        for t in tqdm(range(len(timesteps) - 1), desc="Prior diffusion", leave=True):
            t_ = timesteps[t].item()
            t_pre = timesteps[t + 1].item()
            time = torch.full((self.batch_size,), t_, device=self.device, dtype=torch.long)
            prev_time = torch.full((self.batch_size,), t_pre, device=self.device, dtype=torch.long)
            guided_embed = self._prior_guided_pred(
                txt_embed_reduced, curr_embed_reduced, time
            )
            curr_embed_reduced, _ = self.prior_net.rwd_unclip(
                curr_embed_reduced, time, prev_time, guided_embed
            )
        # convert back to full embedding dimension
        if self.prior_dim_reduction:
            f_img_embed = self.prior_net.clip_img_proj.inverse_transform(curr_embed_reduced)
        else:
            f_img_embed = curr_embed_reduced
        # free prior model and intermediate tensors
        self._move_model_to_device(self.prior_net, self.offload_device)
        del embed_noise, curr_embed, curr_embed_reduced, txt_embed
        if self.prior_dim_reduction:
            del txt_embed_reduced
        torch.cuda.empty_cache()

        # ====== DECODER STAGE: generate 64x64 images from embeddings ======
        self._move_model_to_device(self.decoder_net, self.device)
        decoder_noise = torch.randn(
            (self.batch_size, *self.init_img_size), device=self.device
        )
        proj_embed = self.decoder_net.clip_decoder_proj(f_img_embed)
        glide_txt_embed = self.decoder_net._encode_text_with_glide(prompts)
        context = self.decoder_net._conc_embed(glide_txt_embed, proj_embed)
        curr_imgs = decoder_noise
        # decoder diffusion with batched CFG
        timesteps = self.decoder_net.fwd_unclip.vs.inference_timesteps.flip(0)
        for t in tqdm(range(len(timesteps) - 1), desc="Decoder 64x64", leave=True):
            t_ = timesteps[t].item()
            t_pre = timesteps[t + 1].item()
            time = torch.full((self.batch_size,), t_, device=self.device, dtype=torch.long)
            prev_time = torch.full((self.batch_size,), t_pre, device=self.device, dtype=torch.long)
            guided_pred = self._decoder_guided_pred(curr_imgs, time, context)
            curr_imgs, _ = self.decoder_net.rwd_unclip(
                curr_imgs, time, prev_time, guided_pred
            )
        samps_64x64 = curr_imgs
        # free decoder
        self._move_model_to_device(self.decoder_net, self.offload_device)
        del decoder_noise, curr_imgs, context, glide_txt_embed, proj_embed, f_img_embed
        torch.cuda.empty_cache()

        # ====== FIRST UPSAMPLER: 64x64 -> 256x256 ======
        self._move_model_to_device(self.low_res_upsampler, self.device)
        up_256_noise = torch.randn(
            (self.batch_size, self.init_img_size[0], 256, 256), device=self.device
        )
        curr_256_imgs = up_256_noise
        timesteps = self.low_res_upsampler.rwd_unclip.vs.inference_timesteps.flip(0)
        for t in tqdm(range(len(timesteps) - 1), desc="Upsampler 256x256", leave=True):
            t_ = timesteps[t].item()
            t_pre = timesteps[t + 1].item()
            time = torch.full((self.batch_size,), t_, device=self.device, dtype=torch.long)
            prev_time = torch.full((self.batch_size,), t_pre, device=self.device, dtype=torch.long)
            pred = self.low_res_upsampler(curr_256_imgs, time, samps_64x64)
            curr_256_imgs, _ = self.low_res_upsampler.rwd_unclip(
                curr_256_imgs, time, prev_time, pred
            )
        self.imgs_256 = curr_256_imgs
        # free low-res upsampler
        self._move_model_to_device(self.low_res_upsampler, self.offload_device)
        del up_256_noise, curr_256_imgs, samps_64x64
        torch.cuda.empty_cache()

        # ====== SECOND UPSAMPLER: 256x256 -> 1024x1024 ======
        if self.use_high_res_upsampler and self.high_res_upsampler:
            self._move_model_to_device(self.high_res_upsampler, self.device)
            up_1024_noise = torch.randn(
                (self.batch_size, self.init_img_size[0], 1024, 1024), device=self.device
            )
            curr_1024_imgs = up_1024_noise
            timesteps = self.high_res_upsampler.rwd_unclip.vs.inference_timesteps.flip(0)
            for t in tqdm(range(len(timesteps) - 1), desc="Upsampler 1024x1024", leave=True):
                t_ = timesteps[t].item()
                t_pre = timesteps[t + 1].item()
                time = torch.full((self.batch_size,), t_, device=self.device, dtype=torch.long)
                prev_time = torch.full((self.batch_size,), t_pre, device=self.device, dtype=torch.long)
                pred = self.high_res_upsampler(curr_1024_imgs, time, self.imgs_256)
                curr_1024_imgs, _ = self.high_res_upsampler.rwd_unclip(
                    curr_1024_imgs, time, prev_time, pred
                )
            self.imgs_1024 = curr_1024_imgs
            # free high-res upsampler
            self._move_model_to_device(self.high_res_upsampler, self.offload_device)
            del up_1024_noise, curr_1024_imgs
            torch.cuda.empty_cache()

        # ====== POST-PROCESSING ======
        if norm_output:
            f_256 = (self.imgs_256 - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])
            f_1024 = None
            if self.imgs_1024 is not None:
                f_1024 = (self.imgs_1024 - self.norm_range[0]) / (self.norm_range[1] - self.norm_range[0])
        else:
            f_256 = self.imgs_256
            f_1024 = self.imgs_1024

        if save_imgs:
            self._save_images(f_256, f_1024, save_path)
        return f_1024 if f_1024 is not None else f_256

    def _prior_guided_pred(
            self,
            txt_embed: torch.Tensor,
            curr_embed: torch.Tensor,
            t: torch.Tensor
    ) -> torch.Tensor:
        """Batched CFG for prior"""
        batch_size = txt_embed.shape[0]
        txt_embed_batched = torch.cat([txt_embed, torch.zeros_like(txt_embed)], dim=0)
        curr_embed_batched = torch.cat([curr_embed, curr_embed], dim=0)
        t_batched = torch.cat([t, t], dim=0)
        pred_batched = self.prior_net(txt_embed_batched, curr_embed_batched, t_batched)
        pred_cond, pred_uncond = pred_batched.chunk(2, dim=0)
        return pred_uncond + self.prior_guidance_scale * (pred_cond - pred_uncond)

    def _decoder_guided_pred(
            self,
            curr_imgs: torch.Tensor,
            t: torch.Tensor,
            context: torch.Tensor
    ) -> torch.Tensor:
        """Batched CFG for decoder"""
        curr_imgs_batched = torch.cat([curr_imgs, curr_imgs], dim=0)
        t_batched = torch.cat([t, t], dim=0)
        context_batched = torch.cat([context, torch.zeros_like(context)], dim=0)
        pred_batched = self.decoder_net.diff_net(curr_imgs_batched, t_batched, context_batched, None)
        pred_cond, pred_uncond = pred_batched.chunk(2, dim=0)
        return pred_uncond + self.decoder_guidance_scale * (pred_cond - pred_uncond)

    def _save_images(self, f_256, f_1024, save_path):
        """Helper method for saving images."""
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, "imgs_256"), exist_ok=True)
        if f_1024 is not None:
            os.makedirs(os.path.join(save_path, "imgs_1024"), exist_ok=True)
        for i in range(self.batch_size):
            img_path_256 = os.path.join(save_path, "imgs_256", f"img_{i + 1}.png")
            torchvision.utils.save_image(f_256[i], img_path_256)
            if f_1024 is not None:
                img_path_1024 = os.path.join(save_path, "imgs_1024", f"img_{i + 1}.png")
                torchvision.utils.save_image(f_1024[i], img_path_1024)