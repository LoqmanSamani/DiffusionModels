# TorchDiff


<div align="center">
  <img src="imgs/logo_.png" alt="TorchDiff Logo" width="300"/>
</div>

<div align="center">

[![License: MIT](https://img.shields.io/badge/license-MIT-red?style=plastic)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-white?style=plastic&logo=pytorch&logoColor=red)](https://pytorch.org/)
[![Version](https://img.shields.io/badge/version-2.2.0-blue?style=plastic)](https://pypi.org/project/torchdiff/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue?style=plastic&logo=python&logoColor=white)](https://www.python.org/)
[![Downloads](https://pepy.tech/badge/torchdiff)](https://pepy.tech/project/torchdiff)
[![Stars](https://img.shields.io/github/stars/LoqmanSamani/TorchDiff?style=plastic&color=yellow)](https://github.com/LoqmanSamani/TorchDiff)
[![Forks](https://img.shields.io/github/forks/LoqmanSamani/TorchDiff?style=plastic&color=orange)](https://github.com/LoqmanSamani/TorchDiff)
[![Issues](https://img.shields.io/github/issues/LoqmanSamani/TorchDiff?style=plastic&color=red)](https://github.com/LoqmanSamani/TorchDiff/issues)

</div>

---

## 🔎 Overview  

**TorchDiff** is a PyTorch-based library for building and experimenting with diffusion models, inspired by leading research papers.  

The **TorchDiff 2.0.0** release includes implementations of five major diffusion model families:  
- **DDPM** (Denoising Diffusion Probabilistic Models)  
- **DDIM** (Denoising Diffusion Implicit Models)  
- **SDE-based Diffusion**  
- **LDM** (Latent Diffusion Models)  
- **UnCLIP** (the model powering OpenAI’s *DALL·E 2*)  

These models support both **conditional** (e.g., text-to-image) and **unconditional** generation.  

<div align="center">
  <img src="imgs/mount.png" alt="Diffusion Model Process" width="1000"/>
  <br>
  <em>Image generated using Sora</em>
  <br><br>
</div>

TorchDiff is designed with **modularity** in mind. Each model is broken down into reusable components:  
- **Forward Diffusion**: Adds noise (e.g., `ForwardDDPM`).  
- **Reverse Diffusion**: Removes noise to recover data (e.g., `ReverseDDPM`).  
- **Scheduler**: Controls noise schedules (e.g., `SchedulerDDPM`).  
- **Training**: Full training pipelines (e.g., `TrainDDPM`).  
- **Sampling**: Efficient inference and generation (e.g., `SampleDDPM`).  

Additional utilities:  
- **Diffusion Network**: A U-Net-like model with attention and time embeddings used as main model.  
- **Text Encoder**: Transformer-based (e.g., BERT) for conditional generation.  
- **Metrics**: Evaluation suite including MSE, PSNR, SSIM, FID, and LPIPS.  

---

## ⚡ Quick Start  

Here’s a minimal working example to train and sample with **DDPM** on dummy data:

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
trainer()

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

For detailed examples, check the [examples/](https://github.com/LoqmanSamani/TorchDiff/tree/systembiology/examples) directory.  

---

## 📚 Resources  
- 🌐 [Project Website](https://loqmansamani.github.io/torchdiff/)  
- 📖 [API Reference](https://torchdiff.readthedocs.io/en/latest/index.html)  

---

## ⚡ Installation  

Install from **PyPI (recommended):**
```bash
pip install torchdiff
```

Or install from source for development:  
```bash
# Clone repository
git clone https://github.com/LoqmanSamani/TorchDiff.git
cd TorchDiff

# Install dependencies
pip install -r requirements.txt

# Install package
pip install .
```

> Requires **Python 3.10+**. For GPU acceleration, ensure PyTorch is installed with the correct CUDA version.  

---

## 🧩 Implemented Models  

### 1. Denoising Diffusion Probabilistic Models (DDPM)  
**Paper**: [Ho et al., 2020](https://arxiv.org/abs/2006.11239)  

DDPMs learn to reverse a gradual noise-adding process to generate high-quality images. TorchDiff provides a modular implementation for both unconditional and conditional (text-guided) generation.  

📓 [DDPM Example Notebook](https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/examples/ddpm.ipynb)  

---

### 2. Denoising Diffusion Implicit Models (DDIM)  
**Paper**: [Song et al., 2021](https://arxiv.org/abs/2010.02502)  

DDIM accelerates sampling by reducing the number of denoising steps while maintaining image quality. TorchDiff supports both conditional and unconditional DDIM generation.  

📓 [DDIM Example Notebook](https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/examples/ddim.ipynb)  

---

### 3. Score-Based Generative Models via Stochastic Differential Equations (SDE)  
**Paper**: [Song et al., 2021](https://arxiv.org/abs/2011.13456)  

SDE-based models generalize diffusion via stochastic processes, supporting multiple formulations: **VE, VP, sub-VP**, and deterministic **ODE** variants. TorchDiff includes full training and sampling pipelines for both conditional and unconditional use cases.  

📓 [SDE Example Notebook](https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/examples/sde.ipynb)  

---

### 4. Latent Diffusion Models (LDM)  
**Paper**: [Rombach et al., 2022](https://arxiv.org/abs/2112.10752)  

LDMs operate in a compressed latent space using a VAE, enabling **efficient high-resolution image synthesis** with reduced computational cost. TorchDiff supports using DDPM, DDIM, or SDE as the diffusion backbone in latent space.  

📓 [LDM Example Notebook](https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/examples/ldm.ipynb)  

---

### 5. UnCLIP (Hierarchical Text-Conditional Image Generation with CLIP Latents)  
**Paper**: [Ramesh et al., 2022](https://arxiv.org/abs/2204.06125)  

UnCLIP, the architecture behind *DALL·E 2*, leverages **CLIP latents** to enable hierarchical text-to-image generation. It first maps text into CLIP’s multimodal embedding space, then performs diffusion-based generation in that space, followed by refinement in pixel space.  

Training UnCLIP is significantly more complex than other diffusion families, and thus a minimal example is not shown here.  

📓 [UnCLIP Example Notebook](https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/examples/unclip.ipynb)  

---

## 🔐 License  
Released under the [MIT License](https://github.com/LoqmanSamani/TorchDiff/blob/systembiology/LICENSE).  

---

## 🚧 Roadmap / Future Work  
TorchDiff is under active development. Planned features include:  
- 🧠 New diffusion variants and improved training algorithms.  
- ⚡ Faster and more memory-efficient sampling.  
- 🎯 Additional utilities to simplify experimentation.  

---

## 🤝 Contributing  
Contributions are welcome!  

- Open an [Issue](../../issues) to report bugs or request features.  
- Submit a PR with improvements or new features.  

Your feedback helps make TorchDiff better for the community.  


---

## 📖 Citation  

If you use **TorchDiff** in your research or project, please cite the original papers and this repository.  

### Core Diffusion Papers  

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
  journal={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{song2021score,
  title={Score-Based Generative Modeling through Stochastic Differential Equations},
  author={Song, Yang and Sohl-Dickstein, Jascha and Kingma, Diederik P and Kumar, Abhishek and Ermon, Stefano and Poole, Ben},
  journal={International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{rombach2022high,
  title={High-Resolution Image Synthesis with Latent Diffusion Models},
  author={Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Björn},
  journal={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
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
  howpublished = {\url{https://github.com/LoqmanSamani/TorchDiff}},
}
```

