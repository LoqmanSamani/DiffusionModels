.. torchdiff documentation master file, created by
   sphinx-quickstart on Sun Apr 20 15:42:47 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to TorchDiff's Documentation
======================================

**TorchDiff** is a PyTorch-based library designed for building and experimenting with diffusion models, offering modular and flexible implementations of state-of-the-art generative models, including Denoising Diffusion Probabilistic Models (DDPM), Denoising Diffusion Implicit Models (DDIM), Latent Diffusion Models (LDM), and Score-Based Generative Modeling through Stochastic Differential Equations (SDE). It supports both conditional (e.g., text-prompt-based) and unconditional image generation, with components like noise predictors, text encoders, and image quality metrics (MSE, PSNR, SSIM, FID, LPIPS) to streamline model training, sampling, and evaluation. Ideal for researchers and practitioners, TorchDiff provides a robust foundation for developing and customizing diffusion-based generative pipelines.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   ddpm
   ddim
   sde
   ldm
   utils

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

