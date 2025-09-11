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

The **TorchDiff 2.0.0** release includes implementations of five major diffusion model families:  
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

from torchdiff.ddpm import VarianceSchedulerDDPM, ForwardDDPM, ReverseDDPM, TrainDDPM, SampleDDPM
from torchdiff.utils import NoisePredictor

# Dataset setup
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Model components
noise_pred = NoisePredictor(in_channels=3)
vs = VarianceSchedulerDDPM(num_steps=1000)
fwd, rev = ForwardDDPM(vs), ReverseDDPM(vs)

# Training
trainer = TrainDDPM(
    noise_predictor=noise_pred, forward_diffusion=fwd, reverse_diffusion=rev,
    conditional_model=None, optimizer=torch.optim.Adam(noise_pred.parameters(), lr=1e-4),
    objective=nn.MSELoss(), data_loader=train_loader, max_epochs=1, device="cpu"
)
trainer()

# Sampling
sampler = SampleDDPM(reverse_diffusion=rev, noise_predictor=noise_pred,
                     image_shape=(32, 32), batch_size=4, in_channels=3, device="cpu")
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
- **Variance Scheduler**: Controls noise schedules
- **Training**: Complete training pipelines
- **Sampling**: Efficient inference and generation

Additional utilities:
- **Noise Predictor**: U-Net-like model with attention and time embeddings
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