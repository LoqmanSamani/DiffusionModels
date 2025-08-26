.. torchdiff documentation master file, created by
   sphinx-quickstart on Sun Apr 20 15:42:47 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TorchDiff's Documentation
======================================

**TorchDiff** is a PyTorch-based library for building and experimenting with diffusion models, inspired by leading research in generative AI. It provides modular and flexible implementations of state-of-the-art diffusion-based generative models, including Denoising Diffusion Probabilistic Models (DDPM), Denoising Diffusion Implicit Models (DDIM), Score-Based Generative Models through Stochastic Differential Equations (SDE), Latent Diffusion Models (LDM), and UnCLIP (Hierarchical Text-Conditional Image Generation with CLIP Latents). The library supports both conditional (e.g., text-to-image) and unconditional generation, with key components such as forward and reverse diffusion processes, variance schedulers, U-Net-like noise predictors with attention and time embeddings, transformer-based text encoders (e.g., BERT), and a comprehensive evaluation suite featuring image quality metrics (MSE, PSNR, SSIM, FID, LPIPS). Designed for researchers and practitioners, TorchDiff offers a robust, extensible foundation for training, sampling, and customizing advanced generative pipelines.


.. toctree::
   :maxdepth: 1
   :caption: Contents:

   ddpm
   ddim
   sde
   ldm
   unclip
   utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

