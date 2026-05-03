import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import io
import time
import os
from collections import OrderedDict

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DCGAN vs WGAN-GP | Anime Face Generator",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Dark gradient background ── */
.stApp {
    background: linear-gradient(135deg, #0d0d1a 0%, #1a1a2e 50%, #16213e 100%);
    color: #e0e0f0;
}

/* ── Hero banner ── */
.hero {
    background: linear-gradient(90deg, #7b2ff7, #f107a3);
    border-radius: 16px;
    padding: 2.5rem 2rem;
    margin-bottom: 1.5rem;
    text-align: center;
    box-shadow: 0 8px 32px rgba(123,47,247,0.4);
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    color: white;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    font-size: 1.05rem;
    color: rgba(255,255,255,0.85);
    margin: 0.6rem 0 0;
}

/* ── Cards ── */
.card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(8px);
}
.card h3 {
    margin: 0 0 0.4rem;
    font-size: 1rem;
    color: #a78bfa;
    font-weight: 600;
}
.card p, .card li {
    font-size: 0.88rem;
    color: #c4c4e0;
    margin: 0.2rem 0;
}

/* ── Metric boxes ── */
.metric-row {
    display: flex;
    gap: 1rem;
    margin-bottom: 1rem;
}
.metric-box {
    flex: 1;
    background: rgba(123,47,247,0.15);
    border: 1px solid rgba(123,47,247,0.3);
    border-radius: 10px;
    padding: 0.9rem;
    text-align: center;
}
.metric-box .val {
    font-size: 1.6rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-box .lbl {
    font-size: 0.78rem;
    color: #9090b0;
    margin-top: 0.2rem;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: rgba(255,255,255,0.04);
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 8px;
    color: #9090b0;
    font-weight: 500;
    padding: 8px 20px;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(90deg, #7b2ff7, #f107a3) !important;
    color: white !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: rgba(13,13,26,0.9);
    border-right: 1px solid rgba(255,255,255,0.08);
}
section[data-testid="stSidebar"] * {
    color: #d0d0f0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(90deg, #7b2ff7, #f107a3);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    width: 100%;
    transition: opacity 0.2s, transform 0.1s;
}
.stButton > button:hover {
    opacity: 0.9;
    transform: translateY(-1px);
}

/* ── Image grid label ── */
.model-label {
    text-align: center;
    font-size: 0.9rem;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 20px;
    margin-bottom: 8px;
    display: inline-block;
}
.dcgan-label { background: rgba(123,47,247,0.2); color: #a78bfa; border: 1px solid #7b2ff7; }
.wgan-label  { background: rgba(241,7,163,0.2);  color: #f472b6; border: 1px solid #f107a3; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.1) !important; }

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(255,255,255,0.08) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    color: white !important;
    border-radius: 8px !important;
    width: 100%;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURE  (exact copy from notebook)
# ─────────────────────────────────────────────────────────────────
Z_DIM       = 100
NUM_CHANNELS = 3
FEATURES_G  = 64
IMAGE_SIZE  = 64


class DCGANGenerator(nn.Module):
    """Input: noise (B, 100, 1, 1)  →  Output: RGB image (B, 3, 64, 64)"""

    def __init__(self, z_dim, channels_img, features_g):
        super().__init__()
        fg = features_g
        self.net = nn.Sequential(
            self._block(z_dim,  fg*16, 4, 1, 0),   # → 4×4
            self._block(fg*16,  fg*8,  4, 2, 1),   # → 8×8
            self._block(fg*8,   fg*4,  4, 2, 1),   # → 16×16
            self._block(fg*4,   fg*2,  4, 2, 1),   # → 32×32
            nn.ConvTranspose2d(fg*2, channels_img, 4, 2, 1),  # → 64×64
            nn.Tanh(),
        )

    def _block(self, in_c, out_c, k, s, p):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, k, s, p, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, z):
        return self.net(z)


# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
DEVICE = torch.device("cpu")   # Streamlit Cloud → CPU only


@st.cache_resource(show_spinner=False)
def load_model(weight_path: str) -> DCGANGenerator:
    """Load generator weights, strip DataParallel 'module.' prefix if present."""
    model = DCGANGenerator(Z_DIM, NUM_CHANNELS, FEATURES_G).to(DEVICE)
    model.eval()

    if not os.path.exists(weight_path):
        return None

    raw = torch.load(weight_path, map_location=DEVICE)

    # Handle DataParallel checkpoint (keys start with 'module.')
    new_sd = OrderedDict()
    for k, v in raw.items():
        new_sd[k[7:] if k.startswith("module.") else k] = v

    model.load_state_dict(new_sd)
    return model


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert single image tensor [-1,1] → PIL Image."""
    arr = t.detach().cpu().float()
    arr = (arr * 0.5 + 0.5).clamp(0, 1)
    arr = (arr.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return Image.fromarray(arr)


def generate_images(model: DCGANGenerator, n: int, seed: int) -> list:
    """Generate n PIL images from the model."""
    torch.manual_seed(seed)
    noise = torch.randn(n, Z_DIM, 1, 1, device=DEVICE)
    with torch.no_grad():
        fakes = model(noise)
    return [tensor_to_pil(fakes[i]) for i in range(n)]


def pil_grid(images: list, cols: int = 4) -> Image.Image:
    """Stitch PIL images into a single grid image."""
    rows = (len(images) + cols - 1) // cols
    W, H = images[0].size
    grid = Image.new("RGB", (cols * W, rows * H), color=(20, 20, 35))
    for idx, img in enumerate(images):
        c, r = idx % cols, idx // cols
        grid.paste(img, (c * W, r * H))
    return grid


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Controls")
    st.markdown("---")

    model_choice = st.radio(
        "🤖 Model",
        ["DCGAN", "WGAN-GP", "Compare Both"],
        index=2,
    )

    st.markdown("---")
    n_images = st.select_slider(
        "🖼️ Number of images",
        options=[4, 8, 16],
        value=8,
    )

    use_fixed_seed = st.checkbox("🔒 Fix random seed", value=False)
    seed = st.number_input("Seed", min_value=0, max_value=99999,
                           value=42, step=1, disabled=not use_fixed_seed)
    if not use_fixed_seed:
        seed = np.random.randint(0, 99999)

    st.markdown("---")
    generate_btn = st.button("✨ Generate Images", use_container_width=True)

    st.markdown("---")

    # ── Model info ──
    st.markdown("### 📐 Architecture")
    st.markdown("""
<div class="card">
<h3>Both models share:</h3>
<ul>
<li>Noise dim: <b>100</b></li>
<li>Output: <b>64 × 64 RGB</b></li>
<li>Base features: <b>64</b></li>
<li>5 ConvTranspose layers</li>
<li>BatchNorm + ReLU</li>
<li>Tanh output</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("### ⚖️ Key Difference")
    st.markdown("""
<div class="card">
<h3>Training objective:</h3>
<ul>
<li>DCGAN → <b>BCE loss</b></li>
<li>WGAN-GP → <b>Wasserstein + Gradient Penalty</b></li>
</ul>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# MAIN — HERO
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🎨 DCGAN vs WGAN-GP</h1>
    <p>Anime Face Generation · Tackling Mode Collapse · 64 × 64 · Trained on Anime Faces Dataset</p>
</div>
""", unsafe_allow_html=True)

# ── Metrics row ──
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="metric-box"><div class="val">64×64</div><div class="lbl">Output Resolution</div></div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="metric-box"><div class="val">100</div><div class="lbl">Latent Dimensions</div></div>', unsafe_allow_html=True)
with col3:
    st.markdown('<div class="metric-box"><div class="val">2 GAN</div><div class="lbl">Architectures</div></div>', unsafe_allow_html=True)
with col4:
    st.markdown('<div class="metric-box"><div class="val">50</div><div class="lbl">Training Epochs</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────────
DCGAN_WEIGHTS = "dcgan_generator.pth"
WGAN_WEIGHTS  = "wgan_generator.pth"

with st.spinner("Loading model weights…"):
    dcgan_model = load_model(DCGAN_WEIGHTS)
    wgan_model  = load_model(WGAN_WEIGHTS)

models_ok = {"DCGAN": dcgan_model is not None, "WGAN-GP": wgan_model is not None}

if not any(models_ok.values()):
    st.error(
        "⚠️  No weight files found. Place `dcgan_generator.pth` and/or "
        "`wgan_generator.pth` in the same folder as `app.py`, then restart."
    )
    st.info(
        "**To get the weights:** train the notebook on Kaggle, then upload "
        "the `.pth` files to this repo or your Hugging Face model repository."
    )
    st.stop()

for name, ok in models_ok.items():
    if not ok:
        st.warning(f"⚠️  {name} weights not found — skipping that model.")


# ─────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────
tab_gen, tab_compare, tab_about = st.tabs(["🎨 Generate", "⚖️ Compare", "📖 About"])


# ══════════════════════════════════════════════════════════════════
# TAB 1 — GENERATE
# ══════════════════════════════════════════════════════════════════
with tab_gen:
    if generate_btn or "gen_images" not in st.session_state:

        if model_choice == "DCGAN" and dcgan_model:
            with st.spinner("Generating with DCGAN…"):
                imgs = generate_images(dcgan_model, n_images, seed)
            st.session_state.gen_images = imgs
            st.session_state.gen_label  = "DCGAN"
            st.session_state.gen_seed   = seed

        elif model_choice == "WGAN-GP" and wgan_model:
            with st.spinner("Generating with WGAN-GP…"):
                imgs = generate_images(wgan_model, n_images, seed)
            st.session_state.gen_images = imgs
            st.session_state.gen_label  = "WGAN-GP"
            st.session_state.gen_seed   = seed

        elif model_choice == "Compare Both":
            # handled in compare tab — default to DCGAN here if available
            if dcgan_model:
                with st.spinner("Generating with DCGAN…"):
                    imgs = generate_images(dcgan_model, n_images, seed)
                st.session_state.gen_images = imgs
                st.session_state.gen_label  = "DCGAN (switch to Compare tab for both)"
                st.session_state.gen_seed   = seed

    if "gen_images" in st.session_state:
        imgs   = st.session_state.gen_images
        label  = st.session_state.gen_label
        g_seed = st.session_state.gen_seed

        label_cls = "dcgan-label" if "DCGAN" in label else "wgan-label"
        st.markdown(
            f'<span class="model-label {label_cls}">{label}</span> '
            f'<span style="font-size:0.8rem;color:#777"> seed={g_seed}</span>',
            unsafe_allow_html=True,
        )

        cols_per_row = 4
        for row_start in range(0, len(imgs), cols_per_row):
            row_imgs = imgs[row_start: row_start + cols_per_row]
            cols = st.columns(len(row_imgs))
            for col, img in zip(cols, row_imgs):
                col.image(img, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        grid_img = pil_grid(imgs, cols=min(4, len(imgs)))
        st.download_button(
            label="⬇️ Download Grid PNG",
            data=pil_to_bytes(grid_img),
            file_name=f"{label.split()[0].lower()}_generated_seed{g_seed}.png",
            mime="image/png",
            use_container_width=True,
        )


# ══════════════════════════════════════════════════════════════════
# TAB 2 — COMPARE BOTH
# ══════════════════════════════════════════════════════════════════
with tab_compare:
    if not (dcgan_model and wgan_model):
        st.warning("Both model weights are needed for comparison.")
    else:
        st.markdown("### Side-by-Side Comparison — Same Noise Input")
        st.caption("Both models receive the **identical random noise** so differences are purely due to architecture & training objective.")

        if generate_btn or "cmp_dcgan" not in st.session_state:
            with st.spinner("Generating comparison images…"):
                cmp_imgs_dcgan = generate_images(dcgan_model, n_images, seed)
                cmp_imgs_wgan  = generate_images(wgan_model,  n_images, seed)
            st.session_state.cmp_dcgan = cmp_imgs_dcgan
            st.session_state.cmp_wgan  = cmp_imgs_wgan
            st.session_state.cmp_seed  = seed

        cmp_d = st.session_state.cmp_dcgan
        cmp_w = st.session_state.cmp_wgan
        cmp_s = st.session_state.cmp_seed

        # Header row
        left_h, right_h = st.columns(2)
        left_h.markdown('<div style="text-align:center"><span class="model-label dcgan-label">🟣 DCGAN</span></div>', unsafe_allow_html=True)
        right_h.markdown('<div style="text-align:center"><span class="model-label wgan-label">🩷 WGAN-GP</span></div>', unsafe_allow_html=True)

        st.markdown(f'<p style="text-align:center;font-size:0.78rem;color:#777">seed = {cmp_s}</p>', unsafe_allow_html=True)

        cols_per_row = 4
        for row_start in range(0, min(len(cmp_d), len(cmp_w)), cols_per_row):
            d_row = cmp_d[row_start: row_start + cols_per_row]
            w_row = cmp_w[row_start: row_start + cols_per_row]

            left_cols  = st.columns(len(d_row))
            right_cols = st.columns(len(w_row))

            left_block, right_block = st.columns(2)
            with left_block:
                sub_cols = st.columns(len(d_row))
                for sc, img in zip(sub_cols, d_row):
                    sc.image(img, use_container_width=True)
            with right_block:
                sub_cols = st.columns(len(w_row))
                for sc, img in zip(sub_cols, w_row):
                    sc.image(img, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Download both grids
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button(
                "⬇️ Download DCGAN Grid",
                data=pil_to_bytes(pil_grid(cmp_d)),
                file_name=f"dcgan_seed{cmp_s}.png",
                mime="image/png",
                use_container_width=True,
            )
        with dl2:
            st.download_button(
                "⬇️ Download WGAN-GP Grid",
                data=pil_to_bytes(pil_grid(cmp_w)),
                file_name=f"wgan_seed{cmp_s}.png",
                mime="image/png",
                use_container_width=True,
            )

        # Comparison table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📊 Architecture Comparison")
        st.markdown("""
| Feature | DCGAN | WGAN-GP |
|---------|-------|---------|
| **Loss function** | Binary Cross-Entropy | Wasserstein Distance |
| **Regularization** | Label smoothing (0.9) | Gradient Penalty (λ=10) |
| **Discriminator output** | Sigmoid → probability | Raw score (no Sigmoid) |
| **Critic updates / G step** | 1 : 1 | 5 : 1 |
| **Normalization** | BatchNorm | InstanceNorm (critic) |
| **Mode collapse risk** | Higher | Lower |
| **Training stability** | Moderate | High |
| **Generator architecture** | Shared ✓ | Shared ✓ |
""")


# ══════════════════════════════════════════════════════════════════
# TAB 3 — ABOUT
# ══════════════════════════════════════════════════════════════════
with tab_about:
    c1, c2 = st.columns([1.2, 1])

    with c1:
        st.markdown("## 📖 Project Overview")
        st.markdown("""
This app demonstrates two landmark GAN architectures trained on the
**Anime Faces dataset** (21,551 images, 64×64 pixels):

**DCGAN** *(Radford et al., 2015)* introduces convolutional layers,
BatchNorm, and LeakyReLU activations to stabilise GAN training.

**WGAN-GP** *(Gulrajani et al., 2017)* replaces the BCE discriminator
with a Wasserstein critic and adds a gradient penalty to enforce the
Lipschitz constraint — greatly reducing mode collapse.
""")

        st.markdown("### 🏗️ Generator Architecture")
        st.code("""
Noise (100, 1, 1)
  → ConvTranspose2d → BN → ReLU  [→ 4×4]
  → ConvTranspose2d → BN → ReLU  [→ 8×8]
  → ConvTranspose2d → BN → ReLU  [→ 16×16]
  → ConvTranspose2d → BN → ReLU  [→ 32×32]
  → ConvTranspose2d → Tanh       [→ 64×64 RGB]
""", language="text")

        st.markdown("### ⚙️ Hyperparameters")
        st.markdown("""
| Parameter | Value |
|-----------|-------|
| Image size | 64 × 64 |
| Batch size | 64 |
| Noise dim (Z) | 100 |
| Learning rate | 0.0002 |
| Adam β₁ / β₂ | 0.5 / 0.999 |
| Epochs | 50 each |
| WGAN-GP λ | 10 |
| Critic steps | 5 per G step |
""")

    with c2:
        st.markdown("## 📚 References")
        st.markdown("""
1. **DCGAN** — Radford et al. (2015)
   *Unsupervised Representation Learning with DCGANs*
   [arXiv:1511.06434](https://arxiv.org/abs/1511.06434)

2. **WGAN** — Arjovsky et al. (2017)
   *Wasserstein GAN*
   [arXiv:1701.07875](https://arxiv.org/abs/1701.07875)

3. **WGAN-GP** — Gulrajani et al. (2017)
   *Improved Training of Wasserstein GANs*
   [arXiv:1704.00028](https://arxiv.org/abs/1704.00028)
""")

        st.markdown("## 🗄️ Dataset")
        st.markdown("""
**Anime Faces** — Soumik Rakshit
- 21,551 anime face images
- Resized to 64 × 64
- Normalized to [-1, 1]
- [Kaggle Dataset](https://www.kaggle.com/datasets/soumikrakshit/anime-faces)
""")

        st.markdown("## 👤 Author")
        st.markdown("""
**Shaban Aftab**
BS Computer Science — FAST NUCES
Generative AI Assignment — Semester VIII

[![GitHub](https://img.shields.io/badge/GitHub-Shaban--Aftab-181717?logo=github)](https://github.com/Shaban-Aftab/DCGAN-and-WGAN)
""")


# ─────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align:center;font-size:0.8rem;color:#555">'
    "DCGAN vs WGAN-GP · Anime Face Generation · Built with Streamlit · "
    "Model trained on Kaggle T4 x2 GPU"
    "</p>",
    unsafe_allow_html=True,
)
