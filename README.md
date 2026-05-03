# DCGAN vs WGAN-GP: Tackling Mode Collapse in Generative Adversarial Networks

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Kaggle-20BEFF?logo=kaggle&logoColor=white)
![GPU](https://img.shields.io/badge/GPU-NVIDIA_T4_x2-76B900?logo=nvidia&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## 📌 Overview

This project implements and compares **two landmark Generative Adversarial Network (GAN) architectures** side-by-side:

| Model | Paper | Key Idea |
|-------|-------|----------|
| **DCGAN** | [Radford et al., 2015](https://arxiv.org/abs/1511.06434) | Deep Convolutional GAN with BatchNorm + LeakyReLU |
| **WGAN-GP** | [Gulrajani et al., 2017](https://arxiv.org/abs/1704.00028) | Wasserstein loss + Gradient Penalty to fix mode collapse |

The notebook trains both models on **Anime Faces** dataset and produces a quantitative and visual comparison of image quality, training stability, and loss behaviour.

---

## 🗂️ Project Structure

```
DCGAN-and-WGAN/
├── DCGANandWGAN.ipynb      ← Main Kaggle notebook (all sections)
└── README.md
```

---

## 🧠 Architecture

### DCGAN Generator
```
Noise (100,1,1)
  → ConvTranspose2d ×4  (4→8→16→32→64 spatial)
  → BatchNorm + ReLU (each hidden layer)
  → Tanh activation (output ∈ [-1, 1])
  → Output: (3, 64, 64)
```

### DCGAN Discriminator
```
Image (3, 64, 64)
  → Conv2d ×4  (64→32→16→8→4 spatial)
  → BatchNorm + LeakyReLU(0.2) (each hidden layer)
  → Conv2d → Flatten → Sigmoid
  → Output: real / fake probability
```

### WGAN-GP Critic
```
Same architecture as Discriminator BUT:
  ✗ No Sigmoid at output  →  raw real-valued score
  ✗ No BatchNorm          →  InstanceNorm (GP-compatible)
  ✓ Gradient Penalty term enforces 1-Lipschitz constraint
```

> **WGAN-GP Generator** reuses the exact DCGAN Generator (only the training objective changes).

---

## ⚙️ Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `IMAGE_SIZE` | 64 | Input/output image resolution |
| `BATCH_SIZE` | 64 | Training batch size |
| `Z_DIM` | 100 | Latent noise vector dimension |
| `FEATURES_G` | 64 | Base feature maps in Generator |
| `FEATURES_D` | 64 | Base feature maps in Discriminator/Critic |
| `LR` | 0.0002 | Adam learning rate |
| `BETA1` | 0.5 | Adam β₁ (momentum) |
| `BETA2` | 0.999 | Adam β₂ (RMS) |
| `DCGAN_EPOCHS` | 50 | Training epochs for DCGAN |
| `WGAN_EPOCHS` | 50 | Training epochs for WGAN-GP |
| `LAMBDA_GP` | 10 | Gradient penalty weight |
| `CRITIC_STEPS` | 5 | Critic updates per generator step |
| `SAVE_EVERY` | 5 | Checkpoint save frequency (epochs) |

---

## 📐 Loss Functions

### DCGAN — Binary Cross Entropy (BCE)
```
L_D = - E[log D(x)]  -  E[log(1 - D(G(z)))]
L_G = - E[log D(G(z))]
```
- Uses `BCEWithLogitsLoss` (numerically stable, autocast-safe)
- Label smoothing: real labels → **0.9**, fake labels → **0.0**

### WGAN-GP — Wasserstein Distance + Gradient Penalty
```
L_C = E[C(G(z))]  -  E[C(x)]  +  λ · GP
L_G = - E[C(G(z))]

GP = λ · E[(‖∇C(x̂)‖₂ - 1)²]
     where  x̂ = ε·x + (1-ε)·G(z),  ε ~ Uniform[0,1]
```
- Critic outputs **raw scores** (no Sigmoid)
- Gradient Penalty enforces the **1-Lipschitz constraint**
- No weight clipping (unlike original WGAN)

---

## 📦 Dataset

| Dataset | Source | Size |
|---------|--------|------|
| **Anime Faces** | [Kaggle – soumikrakshit/anime-faces](https://www.kaggle.com/datasets/soumikrakshit/anime-faces) | ~21k images, 64×64 |

Images are normalized to `[-1, 1]` using `mean=[0.5,0.5,0.5]`, `std=[0.5,0.5,0.5]`.

---

## 🚀 How to Run

### On Kaggle (Recommended)
1. Go to [Kaggle Notebooks](https://www.kaggle.com/code)
2. Upload `DCGANandWGAN.ipynb`
3. Add the **Anime Faces** dataset from the right panel
4. Set **Accelerator → GPU T4 x2**
5. Run all cells

### Locally
```bash
# Install dependencies
pip install torch torchvision tqdm matplotlib clean-fid gradio torchmetrics

# Launch notebook
jupyter notebook DCGANandWGAN.ipynb
```
> ⚠️ Update `ANIME_PATH` in Cell 2.1 to your local dataset path.

---

## 📊 Results & Comparison

### Side-by-Side Output (same noise input)
```
Row 1: DCGAN    samples  →  [16 generated anime faces]
Row 2: WGAN-GP  samples  →  [16 generated anime faces]
```

### Key Differences Observed

| Aspect | DCGAN | WGAN-GP |
|--------|-------|---------|
| **Training Stability** | Prone to mode collapse | Stable; Wasserstein distance is meaningful |
| **Loss Interpretation** | BCE loss can diverge | Critic loss ≈ Wasserstein distance (monotonic) |
| **Output Diversity** | Can collapse to similar images | Better mode coverage |
| **Gradient Signal** | Vanishes when D is too strong | Consistent via gradient penalty |
| **Training Speed** | 1 G + 1 D step per batch | 1 G + 5 C steps per batch |

---

## 🗺️ Notebook Sections

| Section | Description |
|---------|-------------|
| **1. Environment Setup** | Dependencies, seeds, GPU configuration, hyperparameters |
| **2. Data Preparation** | Dataset loading, transforms, DataLoader, sample visualization |
| **3. Model Architecture** | `DCGANGenerator`, `DCGANDiscriminator`, `WGANCritic`, weight init |
| **4. Loss Functions & Optimizers** | BCE, Wasserstein loss, Gradient Penalty, Adam optimizers |
| **5. Helper Utilities** | Checkpoint save/load, image display, GPU memory monitor |
| **6. DCGAN Training** | Discriminator step, Generator step, full training loop, loss curves |
| **7. WGAN-GP Training** | Critic step, Generator step, full training loop, loss curves |
| **8. Evaluation & Comparison** | Final image grids, combined loss curves, FID score, model saving |

---

## 🔬 Technical Highlights

- ✅ **Mixed Precision Training** — `torch.amp.GradScaler` for both models
- ✅ **Multi-GPU Support** — `nn.DataParallel` for T4 x2
- ✅ **Checkpoint System** — save/resume every N epochs
- ✅ **Fixed Noise Tracking** — same `FIXED_NOISE` used every epoch to monitor progress visually
- ✅ **Label Smoothing** — real labels at `0.9` for more stable DCGAN training
- ✅ **Gradient Penalty** — full implementation from scratch (interpolated images → autograd)

---

## 📚 References

1. **DCGAN**: Radford, A., Metz, L., & Chintala, S. (2015). *Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks.* [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)

2. **WGAN**: Arjovsky, M., Chintala, S., & Bottou, L. (2017). *Wasserstein GAN.* [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)

3. **WGAN-GP**: Gulrajani, I., Ahmed, F., Arjovsky, M., Dumoulin, V., & Courville, A. (2017). *Improved Training of Wasserstein GANs.* [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)

---

## 👤 Author

**Shaban Aftab**  
BS Computer Science — FAST NUCES  
Generative AI Assignment — Semester VIII

---

## 📄 License

This project is licensed under the **MIT License** — see [LICENSE](LICENSE) for details.
