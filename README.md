# Diffusion Models

![License: MIT](https://img.shields.io/badge/license-MIT-red?style=plastic)
![PyTorch](https://img.shields.io/badge/PyTorch-white?style=plastic&logo=pytorch&logoColor=red)

![Diffusion Model](imgs/img.png)  
*Source: [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)*  

## Overview

This project is an **educational deep dive into diffusion models**, focusing on both theoretical understanding and hands-on implementation. The goal is to **implement various diffusion models from scratch using PyTorch**, following the original research papers.

The project is designed for **learning and experimentation**, providing a well-documented structure for researchers, students, and engineers interested in generative modeling.

---

## Implemented & Planned Models

### ✅ Implemented Models

1. **[Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)** – [Code](./ddpm)  
   - The foundational diffusion model, introducing probabilistic image generation through iterative denoising.  

2. **[Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502)** – [Code](./ddim)  
   - An optimized extension of DDPM that speeds up sampling using a deterministic reverse process, reducing the number of diffusion steps while maintaining high-quality results.

---

### 🔄 In Progress

3. **[Score-Based Generative Models (SDE)](https://arxiv.org/abs/2011.13456)**  
   - Uses stochastic differential equations (SDEs) instead of deterministic noise prediction. Offers a fundamentally different approach by learning gradients of the data distribution.

---

### 🔜 Planned Models

4. **[Latent Diffusion Models (LDM)](https://arxiv.org/abs/2112.10752)**  
   - Runs diffusion in a compressed latent space instead of pixel space, significantly improving efficiency while maintaining high-resolution synthesis.

5. **[Cascade Diffusion Models (CDM)](https://arxiv.org/abs/2111.13431)**  
   - Uses a multi-stage approach to progressively refine images, generating high-quality outputs with improved resolution.

6. **[Conditional Diffusion Models](https://arxiv.org/abs/2205.11485)**  
   - Adds conditional information (e.g., class labels) to guide the image generation process, allowing controlled outputs like text-to-image synthesis.

7. **[Flow-Based Diffusion Models](https://arxiv.org/abs/2205.11499)**  
   - Combines normalizing flows with diffusion processes to enhance likelihood estimation and model flexibility.

---

## 📂 Project Structure

Each model follows a structured format:

- **`network.py`** – Defines the neural network (e.g., UNet).
- **`forward_diffusion.py`** – Implements the forward (noising) process.
- **`reverse_diffusion.py`** – Implements the reverse (denoising) process.
- **`train.py`** – Training script (not fully trained due to high computational costs).
- **`generate.py`** – Inference script for generating images.
- **`config.py`** – Stores hyperparameters and model configurations.
- **`tests/`** – Unit tests to ensure correctness and debugging.

🚨 *Note: Due to high computational requirements, models are not trained. However, all implementations are tested and well-documented for easy use.*

---

## 🚀 How to Use

### Clone the Repository
```commandline
git clone https://github.com/your-username/diffusion-models.git
cd diffusion-models
```
### Install Dependencies
```commandline
pip install -r requirements.txt
```

### Train a Model (Modify `config.py` Before Running)
```commandline
python ddpm/train.py
```

### Run an Existing Model (e.g., DDIM)
```commandline
python ddim/generate.py
```

## 🎯 Why This Project?

- ✅ **Understand Diffusion Models** from first principles.  
- ✅ **Hands-on implementation** of research papers.  
- ✅ **Explore different architectures** (stochastic, deterministic, latent-space, etc.).  
- ✅ **Fully documented and tested** for easy learning and experimentation.  


## 🤝 Contribute & Learn Together!  

💡 **Want to contribute?** Open an issue or submit a PR!  
💬 **Have questions?** Start a discussion.  
⭐ **If you find this useful, give it a star!**  




















