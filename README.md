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

1. **[Denoising Diffusion Probabilistic Models (DDPM)](https://arxiv.org/abs/2006.11239)** – [Source](./ddpm)  
   - The original diffusion model that generates images by gradually removing noise over multiple steps. It produces high-quality results but is relatively slow.  
   - The model uses a U-Net architecture ([code](https://github.com/LoqmanSamani/DiffusionModels/blob/systembiology/ddpm/network.py), [structure](https://github.com/LoqmanSamani/DiffusionModels/blob/systembiology/ddpm/unet_structure.md)) architecture with time embeddings and self-attention layers to predict noise.  

2. **[Denoising Diffusion Implicit Models (DDIM)](https://arxiv.org/abs/2010.02502)** – [Source](./ddim)  
   - A faster version of DDPM that requires fewer denoising steps while maintaining high image quality. Instead of following a completely random process, it takes a more direct path to generate images.  
   - DDIM uses the same neural network as DDPM for noise prediction. The key difference is during inference: the number of steps from a fully noisy image to the final output is much lower than in training, controlled by the parameter **τ (Tau)**.  

3. **[Score-Based Generative Modeling through Stochastic Differential Equations (SDE)](https://arxiv.org/abs/2011.13456)** - [Source](./sde)  
   - A more flexible version of diffusion models that uses Stochastic Differential Equations (SDEs) to control how noise is added and removed. This lets us tweak the process for better results and use different ways to solve it, like stochastic sampling or ODE-based methods for more control.  
   - There are three types of SDE models:
     - **Variance Exploding (VE) SDE**: Starts with a tiny bit of noise and keeps adding more, making it great for high-resolution images.
     - **Variance Preserving (VP) SDE**: Works like DDPM, keeping noise levels steady while gradually refining the image.
     - **Sub-Variance Preserving (Sub-VP) SDE**: A mix between VE and VP that balances speed and quality, often needing fewer steps to get a good image.  
   - All three models use the same neural network, a modified U-Net (like DDPM) with extra attention layers and time embeddings to handle the noise removal process.
---

### 🔄 In Progress

4. **[Latent Diffusion Models (LDM)](https://arxiv.org/abs/2112.10752)**  
   - Runs diffusion in a compressed latent space instead of pixel space, significantly improving efficiency while maintaining high-resolution synthesis.

---

### 🔜 Planned Models


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
clone https://github.com/LoqmanSamani/diffusion-models.git
cd diffusion-models
```
### Install Dependencies
```commandline
pip install -r requirements.txt
```

### Training a Model

Before running any training or generation script, ensure that you properly configure the `config.py` file. This file contains all the essential hyperparameters, dataset paths, model architecture details, and other configuration settings.

Once the configuration file is ready, you can start training the model (`train.py`).

### Running a Pre-Trained Model (e.g., DDIM)

To generate images using a pre-trained model, first, make sure the model  is saved at the correct path as defined in your `config.py` file.

Then, you can run the model to generate new samples (`generate.py`).


## 🎯 Why This Project?

- ✅ **Understand Diffusion Models** from first principles.  
- ✅ **Hands-on implementation** of research papers.  
- ✅ **Explore different architectures** (stochastic, deterministic, latent-space, etc.).  
- ✅ **Fully documented and tested** for easy learning and experimentation.  


## 🤝 Contribute & Learn Together!  

💡 **Want to contribute?** Open an issue or submit a PR!  
💬 **Have questions?** Start a discussion.  
⭐ **If you find this useful, give it a star!**  




















