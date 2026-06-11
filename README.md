# GANs from Scratch

A PyTorch implementation of four foundational Generative Adversarial Network (GAN) architectures, all trained on the MNIST handwritten digits dataset.

---

## Table of Contents

- [Overview](#overview)
- [The Mathematics of GANs](#the-mathematics-of-gans)
- [Implementations](#implementations)
  - [1. Simple GAN](#1-simple-gan)
  - [2. DCGAN](#2-dcgan)
  - [3. Conditional GAN (cGAN)](#3-conditional-gan-cgan)
  - [4. Wasserstein GAN (WGAN / WGAN-GP)](#4-wasserstein-gan-wgan--wgan-gp)
- [Project Structure](#project-structure)
- [Setup & Usage](#setup--usage)
- [Dependencies](#dependencies)

---

## Overview

This repository implements four GAN variants from scratch using PyTorch:

| Model | Key Idea | Loss |
|---|---|---|
| Simple GAN | Fully connected generator & discriminator | Binary Cross-Entropy |
| DCGAN | Convolutional architecture for stable training | Binary Cross-Entropy |
| Conditional GAN | Class-conditioned generation | Binary Cross-Entropy |
| WGAN / WGAN-GP | Wasserstein distance; weight clipping or gradient penalty | Wasserstein loss |

All models are trained on **MNIST** (28×28 grayscale images of digits 0–9).

---

## The Mathematics of GANs

### Core GAN Framework (Goodfellow et al., 2014)

A GAN consists of two networks trained in opposition:

- **Generator** $G(z; \theta_g)$: maps a noise vector $z \sim p_z$ to a synthetic sample.
- **Discriminator** $D(x; \theta_d)$: outputs the probability that a sample $x$ is real.

The training objective is a minimax game:

$$\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}} [\log D(x)] + \mathbb{E}_{z \sim p_z} [\log (1 - D(G(z)))]$$

**Discriminator** maximises $V$ — it wants $D(x) \to 1$ for real data and $D(G(z)) \to 0$ for fakes.  
**Generator** minimises $V$ (equivalently maximises $\log D(G(z))$ in practice) — it wants to fool the discriminator.

At the global optimum, the generator distribution $p_g$ equals the data distribution $p_{\text{data}}$, and the discriminator outputs $D(x) = \frac{1}{2}$ everywhere.

The optimal discriminator for a fixed generator is:

$$D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_g(x)}$$

Substituting $D^*$ into $V$ yields the Jensen–Shannon Divergence:

$$C(G) = -\log 4 + 2 \cdot \text{JSD}(p_{\text{data}} \| p_g)$$

The minimum $C(G) = -\log 4$ is achieved when $p_g = p_{\text{data}}$.

---

## Implementations

### 1. Simple GAN

The vanilla GAN using only fully-connected (linear) layers.

#### Architecture

| Component | Layers | Activations |
|---|---|---|
| Generator | $64 \to 256 \to 784$ | LeakyReLU(0.01), Tanh |
| Discriminator | $784 \to 128 \to 1$ | LeakyReLU(0.01), Sigmoid |

#### Loss

Binary Cross-Entropy is used for both networks:

$$\mathcal{L}_D = -\mathbb{E}[\log D(x)] - \mathbb{E}[\log(1 - D(G(z)))]$$

$$\mathcal{L}_G = -\mathbb{E}[\log D(G(z))]$$

#### Hyperparameters

| Parameter | Value |
|---|---|
| Noise dimension $z$ | 64 |
| Learning rate | 3×10⁻⁴ |
| Batch size | 32 |
| Epochs | 100 |
| Optimizer | Adam |

---

### 2. DCGAN

Radford et al. (2015) introduced architectural guidelines for stable GAN training using deep convolutional networks.

#### Key Design Principles

- Replace pooling layers with **strided convolutions** (discriminator) and **transposed convolutions** (generator).
- Use **Batch Normalization** in both networks (except the generator output and discriminator input).
- Use **LeakyReLU** in the discriminator; **ReLU** in generator hidden layers, **Tanh** on output.

#### Architecture

**Generator** — projects noise $z \in \mathbb{R}^{100}$ to a $64 \times 64$ image:

$$z \; (100 \times 1 \times 1) \xrightarrow{\text{ConvT}} 512 \xrightarrow{\text{ConvT}} 256 \xrightarrow{\text{ConvT}} 128 \xrightarrow{\text{ConvT}} 64 \xrightarrow{\text{ConvT}} C_{\text{img}} \; (64 \times 64)$$

**Discriminator** — maps a $64 \times 64$ image to a scalar:

$$C_{\text{img}} \; (64 \times 64) \xrightarrow{\text{Conv}} 64 \xrightarrow{\text{Conv}} 128 \xrightarrow{\text{Conv}} 256 \xrightarrow{\text{Conv}} 512 \xrightarrow{\text{Conv}} 1$$

#### Weight Initialisation

All weights initialised from $\mathcal{N}(0, 0.02)$; batch norm scale from $\mathcal{N}(1, 0.02)$.

#### Hyperparameters

| Parameter | Value |
|---|---|
| Noise dimension $z$ | 100 |
| Image size | 64×64 |
| Learning rate | 2×10⁻⁴ |
| Batch size | 128 |
| Epochs | 50 |
| Optimizer | Adam ($\beta_1=0.5$, $\beta_2=0.999$) |

---

### 3. Conditional GAN (cGAN)

Mirza & Osindero (2014) extended the GAN framework to allow **conditional generation** by feeding class labels into both networks.

#### Mathematical Formulation

The objective becomes conditioned on side information $y$ (the digit label):

$$\min_G \max_D \; V(D, G) = \mathbb{E}_{x \sim p_{\text{data}}(x|y)} [\log D(x \mid y)] + \mathbb{E}_{z \sim p_z} [\log (1 - D(G(z \mid y)))]$$

Here $y$ is embedded via a learnable embedding matrix $E \in \mathbb{R}^{N_c \times d_e}$ (where $N_c = 10$ classes, $d_e = 100$).

#### Architecture

**Generator** — concatenates embedded label with noise before convolution:

$$\text{input} = [z \; \| \; E(y)] \in \mathbb{R}^{200} \xrightarrow{\text{ConvT blocks}} \text{image} \; (64 \times 64)$$

**Discriminator** — concatenates a spatially-tiled label embedding as an extra channel:

$$\text{input} = [x \; \| \; \text{tile}(E(y))] \xrightarrow{\text{Conv blocks}} 1$$

#### Hyperparameters

| Parameter | Value |
|---|---|
| Noise dimension $z$ | 100 |
| Embedding dimension | 100 |
| Num classes | 10 |
| Image size | 64×64 |
| Batch size | 64 |

---

### 4. Wasserstein GAN (WGAN / WGAN-GP)

Standard GANs suffer from **mode collapse** and **vanishing gradients** because the Jensen–Shannon Divergence saturates when $p_g$ and $p_{\text{data}}$ have non-overlapping supports. Arjovsky et al. (2017) proposed using the **Earth Mover's (Wasserstein-1) distance** instead.

#### Wasserstein Distance

The Wasserstein-1 distance between two distributions is:

$$W(p_{\text{data}}, p_g) = \inf_{\gamma \in \Pi(p_{\text{data}}, p_g)} \mathbb{E}_{(x, y) \sim \gamma} [\|x - y\|]$$

By the Kantorovich–Rubinstein duality this becomes:

$$W(p_{\text{data}}, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_{\text{data}}} [f(x)] - \mathbb{E}_{z \sim p_z} [f(G(z))]$$

where the supremum is over all **1-Lipschitz** functions $f$.

#### WGAN Loss

The critic $f_w$ (not a classifier — no sigmoid) is trained to approximate the supremum:

$$\mathcal{L}_{\text{critic}} = -\mathbb{E}_{x \sim p_{\text{data}}} [f_w(x)] + \mathbb{E}_{z \sim p_z} [f_w(G(z))]$$

$$\mathcal{L}_G = -\mathbb{E}_{z \sim p_z} [f_w(G(z))]$$

The Lipschitz constraint is enforced by **weight clipping**: $w \leftarrow \text{clip}(w, -c, c)$ after each critic update.

#### Hyperparameters (Original WGAN)

| Parameter | Value |
|---|---|
| Clip value $c$ | 0.01 |
| Critic iterations per generator step | 5 |
| Learning rate | 5×10⁻⁵ |
| Optimizer | RMSprop |

---

#### WGAN-GP (Gulrajani et al., 2017)

Weight clipping can cause **exploding or vanishing gradients** and limits model capacity. WGAN-GP replaces clipping with a **gradient penalty** that directly enforces the 1-Lipschitz constraint:

$$\mathcal{L}_{\text{critic}} = \underbrace{-\mathbb{E}[f_w(x)] + \mathbb{E}[f_w(G(z))]}_{\text{Wasserstein loss}} + \lambda \underbrace{\mathbb{E}_{\hat{x}} \left[(\|\nabla_{\hat{x}} f_w(\hat{x})\|_2 - 1)^2\right]}_{\text{gradient penalty}}$$

where $\hat{x} = \epsilon x + (1 - \epsilon) G(z)$ is a random interpolation between real and fake samples ($\epsilon \sim \mathcal{U}[0,1]$), and $\lambda = 10$.

Because the gradient penalty requires per-sample statistics, **Batch Normalization is replaced with Instance Normalization** in WGAN-GP.

#### Hyperparameters (WGAN-GP)

| Parameter | Value |
|---|---|
| Gradient penalty $\lambda$ | 10 |
| Critic iterations per generator step | 5 |
| Learning rate | 1×10⁻⁴ |
| Optimizer | Adam ($\beta_1=0.0$, $\beta_2=0.9$) |

---

## Project Structure
GANs-from-scratch/
├── Simple GAN/
│ └── simple_GAN.py # Standalone training script
├── DCGAN/
│ ├── model.py # Generator & Discriminator
│ └── train.py # Training loop
├── Conditional GAN/
│ ├── model.py # Conditional Generator & Discriminator
│ └── train.py # Training loop with class conditioning
├── Wasserten GAN/
│ ├── model.py # WGAN architectures
│ └── train.py # Supports --formulation original|improved
└── requirements.txt


Each implementation saves:
- `outputs/` — fake and real image grids per epoch
- `checkpoints/latest_checkpoint.pth` — resume-able state
- `checkpoints/best_checkpoint.pth` — best generator checkpoint
- `losses.json` — per-epoch generator and discriminator losses

---

## Setup & Usage

```bash
# Clone the repository
git clone https://github.com/knightwalker96/gans-from-scratch.git
cd gans-from-scratch

# Install dependencies
pip install torch torchvision tqdm

# Train Simple GAN
cd "Simple GAN"
python simple_GAN.py

# Train DCGAN
cd ../DCGAN
python train.py

# Train Conditional GAN
cd "../Conditional GAN"
python train.py

# Train WGAN (original, weight clipping)
cd "../Wasserten GAN"
python train.py --formulation original

# Train WGAN-GP (improved, gradient penalty)
python train.py --formulation improved
```

---

Training automatically resumes from the latest checkpoint if one exists.

## Dependencies

- Python 3.x
- [PyTorch](https://pytorch.org/)
- torchvision
- tqdm

---

## References

1. Goodfellow, I. et al. (2014). *Generative Adversarial Nets*. NeurIPS.
2. Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks*. ICLR 2016.
3. Mirza, M., & Osindero, S. (2014). *Conditional Generative Adversarial Nets*. arXiv:1411.1784.
4. Arjovsky, M., Chintala, S., & Bottou, L. (2017). *Wasserstein GAN*. ICML 2017.
5. Gulrajani, I. et al. (2017). *Improved Training of Wasserstein GANs*. NeurIPS.
