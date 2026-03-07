# TorchDiff

<div align="center">
  <img src="https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/imgs/logo_.png?raw=true" alt="TorchDiff Logo" width="300"/>
</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/license-MIT-red?style=plastic)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-white?style=plastic&logo=pytorch&logoColor=red)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/version-2.7.0-blue?style=plastic)](https://pypi.org/project/torchdiff/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=plastic&logo=python&logoColor=white)](https://www.python.org/)
[![Downloads](https://pepy.tech/badge/torchdiff)](https://pepy.tech/project/torchdiff)
[![Stars](https://img.shields.io/github/stars/LoqmanSamani/TorchDiff?style=plastic&color=yellow)](https://github.com/LoqmanSamani/TorchDiff)
[![Forks](https://img.shields.io/github/forks/LoqmanSamani/TorchDiff?style=plastic&color=orange)](https://github.com/LoqmanSamani/TorchDiff)
[![Issues](https://img.shields.io/github/issues/LoqmanSamani/TorchDiff?style=plastic&color=red)](https://github.com/LoqmanSamani/TorchDiff/issues)

</div>




---

## Overview

**TorchDiff** is a PyTorch library for diffusion models, implementing foundational architectures from recent research. The library provides modular components for building, training, and sampling from diffusion-based generative models.

Version 2.6.0 includes five major model families grounded in the diffusion modeling literature. **DDPM** (Ho et al., 2020) and **DDIM** (Song et al., 2021a) establish the core discrete-time framework. **SDE-based diffusion** (Song et al., 2021b) extends this to continuous stochastic processes with variance-exploding and variance-preserving formulations. **LDM** (Rombach et al., 2022) moves diffusion into learned latent spaces via variational autoencoders. **UnCLIP** (Ramesh et al., 2022) combines CLIP embeddings with hierarchical generation for text-to-image synthesis.

<div align="center">
  <img src="https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/imgs/mount.png?raw=true" alt="Diffusion Model Process" width="1000"/>
  <br>
  <em>Image generated using Sora</em>
  <br><br>
</div>

Each model breaks down into reusable components. Forward diffusion modules gradually add noise following model-specific schedules. Reverse diffusion modules learn to denoise through parameterized score functions or direct prediction. Schedulers control variance schedules across timesteps. Training pipelines handle optimization and loss computation. Sampling routines implement inference algorithms ranging from ancestral sampling to deterministic ODEs.

The library includes two main architectural components. **DiffusionNetwork** provides a U-Net variant with temporal embeddings, cross-attention mechanisms, and residual blocks adapted from stable diffusion architectures. **TextEncoder** wraps transformer models like BERT for conditional generation tasks.

We also provide evaluation utilities including standard metrics (MSE, PSNR, SSIM) and perceptual measures (FID, LPIPS) commonly used in generative modeling research.

---



### Installation

Install the stable release from PyPI.

```bash
pip install torchdiff
```

For development or to access the latest features, install from source.

```bash
git clone https://github.com/LoqmanSamani/TorchDiff.git
cd TorchDiff
pip install -r requirements.txt
pip install .
```

The library requires Python 3.10 or newer. GPU acceleration requires PyTorch with appropriate CUDA support for your hardware.

---

## Quick Start

Here we demonstrate basic DDPM training and sampling on CIFAR-10. The example shows the typical workflow of initializing schedulers, diffusion processes, and networks before training.

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torchdiff.ddpm import (SchedulerDDPM, ForwardDDPM, 
                            ReverseDDPM, TrainDDPM, SampleDDPM)
from torchdiff.utils import DiffusionNetwork, mse_loss

# Prepare CIFAR-10 dataset
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform
)
train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True
)
device = 'cuda'

# Initialize diffusion network
diff_net = DiffusionNetwork(
    in_channels=3,
    down_channels=[32, 64, 128],
    mid_channels=[128, 128],
    up_channels=[128, 64, 32],
    down_sampling=[True, True],
    time_embed_dim=128,
    y_embed_dim=128,
    num_down_blocks=2,
    num_mid_blocks=2,
    num_up_blocks=2,
    dropout_rate=0.1,
    cont_time=False
)
print(f"Model parameters: {sum(p.numel() for p in diff_net.parameters()):,}")

# Configure diffusion process
scheduler = SchedulerDDPM(time_steps=400)
forward_process = ForwardDDPM(scheduler, 'noise')
reverse_process = ReverseDDPM(scheduler, 'noise')

# Setup training
optimizer = torch.optim.Adam(diff_net.parameters(), lr=1e-5)
trainer = TrainDDPM(
    diff_net=diff_net,
    fwd_ddpm=forward_process,
    rwd_ddpm=reverse_process,
    train_loader=train_loader,
    optim=optimizer,
    loss_fn=mse_loss,
    max_epochs=10,
    device=device,
    grad_acc=2
)
trainer()

# Generate samples
sampler = SampleDDPM(
    rwd_ddpm=reverse_process,
    diff_net=diff_net,
    img_size=(32, 32),
    batch_size=10,
    in_channels=3,
    device=device
)
images = sampler()
```

Additional examples covering conditional generation, latent diffusion, and SDE variants are available in the [examples/](https://github.com/LoqmanSamani/TorchDiff/tree/systembiology/examples) directory.

---

## Model Implementations

### Denoising Diffusion Probabilistic Models (DDPM)

DDPM (Ho et al., 2020) frames generation as learning to reverse a Markov chain that gradually corrupts data with Gaussian noise. The forward process follows a fixed variance schedule while the reverse process is learned through a neural network that predicts either noise or the denoised sample at each timestep.

The implementation supports both unconditional generation and conditional variants where generation is guided by auxiliary information like class labels or text embeddings.

**Paper:** [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)

**Example:** [DDPM Notebook](https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/examples/ddpm/ddpm.ipynb)

---

### Denoising Diffusion Implicit Models (DDIM)

DDIM (Song et al., 2021a) reformulates the generative process as a non-Markovian procedure that allows deterministic sampling. This enables faster generation by skipping timesteps while maintaining sample quality. The key insight is that many forward processes can correspond to the same reverse process marginals.

Like DDPM, both conditional and unconditional generation modes are supported.

**Paper:** [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)

**Example:** [DDIM Notebook](https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/examples/ddim/ddim.ipynb)

---

### Score-Based Generative Models via SDE

The SDE framework (Song et al., 2021b) generalizes diffusion models as continuous-time stochastic processes. Rather than discrete timesteps, the model learns score functions for a continuous diffusion process governed by stochastic differential equations.

We implement variance-exploding (VE), variance-preserving (VP), and sub-VP formulations. The reverse process can be simulated using either stochastic differential equations or their deterministic probability flow ODE counterparts. This unifies score matching with denoising diffusion and enables more flexible sampling strategies.

**Paper:** [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)

**Example:** [SDE Notebooks](https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/examples/sde/)

---

### Latent Diffusion Models (LDM)

LDM (Rombach et al., 2022) addresses the computational cost of pixel-space diffusion by operating in the latent space of a pretrained autoencoder. A VAE first compresses images into lower-dimensional representations where diffusion training occurs. This reduces memory requirements and speeds up both training and sampling while maintaining generation quality.

Any of the diffusion backends (DDPM, DDIM, SDE) can operate in this latent space. The architecture enables high-resolution synthesis that would be impractical in pixel space.

**Paper:** [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)

**Example:** [LDM Notebook](https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/examples/ldm/ldm.ipynb)

---

### UnCLIP (Hierarchical Text-Conditional Generation)

UnCLIP (Ramesh et al., 2022) is the architecture underlying DALL·E 2. The model performs text-to-image generation in two stages. First, a prior model maps text embeddings to CLIP image embeddings. Then a decoder performs diffusion in pixel space conditioned on these CLIP embeddings.

This hierarchical approach leverages CLIP's multimodal embedding space where text and images share semantic structure. The architecture requires training multiple components including the prior network, the diffusion decoder, and often super-resolution upsampling modules.

Given the complexity, UnCLIP training requires more extensive setup than other models in this library.

**Paper:** [Hierarchical Text-Conditional Image Generation with CLIP Latents](https://arxiv.org/abs/2204.06125)

**Example:** [UnCLIP Notebook](https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/examples/unclip/unclip.ipynb)

---

## Modular Design

TorchDiff breaks each model into reusable components:

| Component                   | Description                                                             |
| --------------------------- | ----------------------------------------------------------------------- |
| **Forward Diffusion** | Adds noise to data following model-specific schedules                   |
| **Reverse Diffusion** | Removes noise to recover data via learned denoising                     |
| **Scheduler**         | Controls variance/noise schedules across timesteps                      |
| **Training**          | Complete training pipelines with mixed precision, gradient accumulation |
| **Sampling**          | Efficient inference and image generation routines                       |

Additional utilities:

- **DiffusionNetwork**: U-Net architecture with attention and time embeddings
- **TextEncoder**: Transformer-based encoder for conditional generation
- **Metrics**: Evaluation suite (MSE, PSNR, SSIM, FID, LPIPS)
- **Loss Functions**: MSE, SNR-capped, VE sigma-weighted score matching

---

## Resources

Documentation and additional materials are available online.

- **Project Website:** [loqmansamani.github.io/torchdiff](https://loqmansamani.github.io/torchdiff/)
- **API Documentation:** [torchdiff.readthedocs.io](https://torchdiff.readthedocs.io/en/latest/index.html)

---

## Development Roadmap

We are actively developing TorchDiff with several improvements planned for future releases.

**Model Extensions**
New diffusion variants and training algorithms from recent literature will be added as they become established. We are particularly interested in methods that improve sample efficiency or generation quality.

**Performance Optimization**
Sampling speed and memory efficiency remain active areas of research. We plan to integrate faster sampling methods and more efficient architectures as they emerge.

**Experimental Utilities**
Additional tools for hyperparameter tuning, ablation studies, and model comparison will make experimentation more straightforward.

---

## Contributing

Contributions are welcome. If you find bugs or have feature requests, please open an [issue](../../issues). Pull requests with improvements or new implementations are appreciated.

Community feedback helps guide development priorities and ensures the library remains useful for research.

---

## License

TorchDiff is released under the [MIT License](https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/LICENSE).

---

## Citation

If you use TorchDiff in your research, please cite the relevant papers and this repository.

### Diffusion Model Papers

```bibtex
@article{ho2020denoising,
  title={Denoising Diffusion Probabilistic Models},
  author={Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
  journal={Advances in Neural Information Processing Systems},
  year={2020}
}

@article{song2021denoising,
  title={Denoising Diffusion Implicit Models},
  author={Song, Jiaming and Meng, Chenlin and Ermon, Stefano},
  journal={International Conference on Learning Representations},
  year={2021}
}

@article{song2021score,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Song, Yang and Sohl-Dickstein, Jascha and Kingma, Diederik P and Kumar, Abhishek and Ermon, Stefano and Poole, Ben},
  journal={International Conference on Learning Representations},
  year={2021}
}

@article{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Björn},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2022}
}

@article{ramesh2022hierarchical,
  title={Hierarchical Text-Conditional Image Generation with CLIP Latents},
  author={Ramesh, Aditya and Pavlov, Mikhail and Goh, Gabriel and Gray, Scott and Voss, Chelsea and Radford, Alec and Chen, Mark and Sutskever, Ilya},
  journal={arXiv preprint arXiv:2204.06125},
  year={2022}
}
```

### TorchDiff Repository

```bibtex
@misc{torchdiff2025,
  author = {Samani, Loghman},
  title = {TorchDiff: A Modular Diffusion Modeling Library in PyTorch},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/LoqmanSamani/TorchDiff}}
}
```
