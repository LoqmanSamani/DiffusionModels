# TorchDiff

<div align="center">
  <img src="https://github.com/LoqmanSamani/TorchDiff/blob/main/imgs/logo_.png?raw=true" alt="TorchDiff Logo" width="300"/>
</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/license-MIT-red?style=plastic)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-white?style=plastic&logo=pytorch&logoColor=red)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/version-2.1.0-blue?style=plastic)](https://pypi.org/project/torchdiff/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue?style=plastic&logo=python&logoColor=white)](https://www.python.org/)
[![Downloads](https://pepy.tech/badge/torchdiff)](https://pepy.tech/project/torchdiff)

</div>

---

## 🔎 Overview  

**TorchDiff** is a PyTorch-based library for building and experimenting with diffusion models, inspired by leading research papers.  

The **TorchDiff 2.1.0** release includes implementations of five major diffusion model families:  
- **DDPM** (Denoising Diffusion Probabilistic Models)  
- **DDIM** (Denoising Diffusion Implicit Models)  
- **SDE-based Diffusion**  
- **LDM** (Latent Diffusion Models)  
- **UnCLIP** (the model powering OpenAI's *DALL·E 2*)  

These models support both **conditional** (e.g., text-to-image) and **unconditional** generation.

---

## ⚡ Installation  

```bash
pip install torchdiff
```

Requires **Python 3.8+**. For GPU acceleration, ensure PyTorch is installed with the correct CUDA version.

---

## ⚡ Quick Start  

Here's a minimal working example to train and sample with **DDPM**:

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from torchdiff.ddpm import (SchedulerDDPM, ForwardDDPM, 
                            ReverseDDPM, TrainDDPM, SampleDDPM)
from torchdiff.utils import DiffusionNetwork, mse_loss

# dataset: CIFAR10
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
device = 'cuda' # gpu is used for training and sampling

# model components
diff_net = DiffusionNetwork(
    in_channels = 3,
    down_channels = [32, 64, 128],
    mid_channels = [128, 128],
    up_channels = [128, 64, 32],
    down_sampling = [True, True],
    time_embed_dim = 128,
    y_embed_dim = 128,
    num_down_blocks = 2,
    num_mid_blocks = 2,
    num_up_blocks = 2,
    dropout_rate = 0.1,
    cont_time = False # time is not continuous, if SDE models it should be true
)
print(sum(p.numel() for p in diff_net.parameters()))

vs = SchedulerDDPM(time_steps = 400)
fwd = ForwardDDPM(vs, 'noise') # network is trained to predict noise
rwd = ReverseDDPM(vs, 'noise')

# optimizer
optim = torch.optim.Adam(diff_net.parameters(), lr=1e-5)

# training algorithm
trainer = TrainDDPM(
    diff_net = diff_net,
    fwd_ddpm = fwd,
    rwd_ddpm = rwd,
    train_loader = train_loader,
    optim = optim,
    loss_fn = mse_loss,
    max_epochs = 10,
    device = device,
    grad_acc = 2
)
#trainer()

# Sampling
sampler = SampleDDPM(
    rwd_ddpm = rwd,
    diff_net = diff_net,
    img_size = (32, 32),
    batch_size = 10,
    in_channels = 3,
    device = device
)
images = sampler()
```

---

## 🧩 Implemented Models  

### 1. **DDPM** - Denoising Diffusion Probabilistic Models  
Learn to reverse a gradual noise-adding process for high-quality image generation.

### 2. **DDIM** - Denoising Diffusion Implicit Models  
Accelerated sampling with reduced denoising steps while maintaining quality.

### 3. **SDE-based Diffusion**  
Generalized diffusion via stochastic processes with VE, VP, sub-VP, and ODE variants.

### 4. **LDM** - Latent Diffusion Models  
Efficient high-resolution synthesis in compressed latent space using VAE.

### 5. **UnCLIP** - Hierarchical Text-Conditional Generation  
DALL·E 2 architecture leveraging CLIP latents for text-to-image generation.

---

## 🏗️ Modular Design

TorchDiff breaks each model into reusable components:
- **Forward Diffusion**: Adds noise to data
- **Reverse Diffusion**: Removes noise to recover data  
- **Scheduler**: Controls noise schedules
- **Training**: Complete training pipelines
- **Sampling**: Efficient inference and generation

Additional utilities:
- **Diffusion Network**: U-Net-like model with attention and time embeddings
- **Text Encoder**: Transformer-based for conditional generation
- **Metrics**: Evaluation suite (MSE, PSNR, SSIM, FID, LPIPS)

---

## 📚 Documentation & Examples

- **GitHub Repository**: [https://github.com/LoqmanSamani/TorchDiff](https://github.com/LoqmanSamani/TorchDiff)
- **Documentation**: [https://torchdiff.readthedocs.io](https://torchdiff.readthedocs.io)  
- **Project Website**: [https://loqmansamani.github.io/torchdiff/](https://loqmansamani.github.io/torchdiff/)

---

## 🔐 License  

Released under the [MIT License](https://github.com/LoqmanSamani/TorchDiff/blob/main/LICENSE).

---

## 🤝 Contributing  

Contributions welcome! Open issues or submit PRs on [GitHub](https://github.com/LoqmanSamani/TorchDiff).

---

## 📖 Citation  

If you use TorchDiff in your research, please cite:

```bibtex
@misc{torchdiff2025,
  author = {Samani, Loghman},
  title = {TorchDiff: A Modular Diffusion Modeling Library in PyTorch},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/LoqmanSamani/TorchDiff}},
}
```