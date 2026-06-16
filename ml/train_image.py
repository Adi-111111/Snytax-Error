#!/usr/bin/env python3
"""
Software baseline for the FPGA neural image renderer.

Architecture mirrors the hardware MLP exactly:
  16 inputs -> 64 (ReLU) -> 32 (ReLU) -> 3 (RGB, sigmoid)

Feature layout (matches hardware input_builder + regfile):
  [0]     x normalized to [0,1]
  [1]     y normalized to [0,1]
  [2..7]  Fourier encoding of x  — sin/cos at 3 frequencies
  [8..13] Fourier encoding of y  — sin/cos at 3 frequencies
  [14]    style scalar  (0.0 = image A,  1.0 = image B)
  [15]    unused

Usage:
  python train_image.py                       # synthetic gradient test
  python train_image.py photo.jpg             # single image fit
  python train_image.py imgA.jpg imgB.jpg     # style blend between two images

Dependencies: torch  numpy  pillow  matplotlib
"""

import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Architecture — must stay in sync with mlp_pipeline.sv
# ---------------------------------------------------------------------------
N_FEATURES = 16
H1, H2, H3 = 64, 32, 3
FOURIER_FREQS = [1.0, 2.0, 4.0]   # 3 freqs × sin+cos × x+y = 12 slots


class ImageMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, H1), nn.ReLU(),
            nn.Linear(H1,         H2), nn.ReLU(),
            nn.Linear(H2,         H3), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------
def build_features(xs, ys, style=0.0):
    """
    xs, ys : 1-D float32 tensors, values in [0, 1]
    Returns (N, 16) tensor matching hardware feature layout.
    """
    N = len(xs)
    parts = [xs.unsqueeze(1), ys.unsqueeze(1)]
    for f in FOURIER_FREQS:
        parts.append(torch.sin(2 * torch.pi * f * xs).unsqueeze(1))
        parts.append(torch.cos(2 * torch.pi * f * xs).unsqueeze(1))
    for f in FOURIER_FREQS:
        parts.append(torch.sin(2 * torch.pi * f * ys).unsqueeze(1))
        parts.append(torch.cos(2 * torch.pi * f * ys).unsqueeze(1))
    parts.append(torch.full((N, 1), float(style)))
    parts.append(torch.zeros(N, 1))
    return torch.cat(parts, dim=1)   # (N, 16)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def load_image(path, max_size=256):
    from PIL import Image
    img = Image.open(path).convert('RGB')
    img.thumbnail((max_size, max_size))
    return np.array(img, dtype=np.float32) / 255.0


def image_to_dataset(img, style=0.0):
    H, W, _ = img.shape
    yi, xi = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    xs = torch.tensor(xi.flatten() / max(W - 1, 1), dtype=torch.float32)
    ys = torch.tensor(yi.flatten() / max(H - 1, 1), dtype=torch.float32)
    feats   = build_features(xs, ys, style=style)
    targets = torch.tensor(img.reshape(-1, 3), dtype=torch.float32)
    return feats, targets, H, W


# ---------------------------------------------------------------------------
# INT8 quantization — simulates hardware weight precision
# ---------------------------------------------------------------------------
def quantize_weights(model):
    """Simulate Q4.4 weight quantization: round to nearest 1/16."""
    q = copy.deepcopy(model)
    with torch.no_grad():
        for layer in q.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data = (layer.weight.data * 16).round().clamp(-128, 127) / 16.0
    return q


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train(model, features, targets, epochs=3000, lr=1e-3, batch=4096):
    opt  = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.MSELoss()
    N = len(features)

    for epoch in range(1, epochs + 1):
        idx  = torch.randperm(N)[:batch]
        pred = model(features[idx])
        loss = loss_fn(pred, targets[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if epoch % 500 == 0:
            print(f"  epoch {epoch:5d} / {epochs}   loss = {loss.item():.5f}")


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
@torch.no_grad()
def render(model, H, W, style=0.0):
    yi, xi = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    xs = torch.tensor(xi.flatten() / max(W - 1, 1), dtype=torch.float32)
    ys = torch.tensor(yi.flatten() / max(H - 1, 1), dtype=torch.float32)
    feats = build_features(xs, ys, style=style)
    return model(feats).numpy().reshape(H, W, 3)


def psnr(pred, ref):
    mse = np.mean((pred - ref) ** 2)
    return float('inf') if mse == 0 else -10 * np.log10(mse + 1e-10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = sys.argv[1:]

    if len(args) == 0:
        print("No image supplied — using synthetic gradient.")
        H, W = 128, 128
        yi, xi = np.meshgrid(np.linspace(0, 1, H), np.linspace(0, 1, W), indexing='ij')
        img_a = np.stack([
            xi,
            yi,
            np.abs(np.sin(xi * np.pi * 3) * np.cos(yi * np.pi * 3)),
        ], axis=-1).astype(np.float32)
        img_b = None

    elif len(args) == 1:
        print(f"Single image: {args[0]}")
        img_a = load_image(args[0])
        img_b = None

    else:
        print(f"Style blend: {args[0]}  <->  {args[1]}")
        img_a = load_image(args[0])
        img_b = load_image(args[1])
        # match spatial size
        Ha, Wa = img_a.shape[:2]
        from PIL import Image as PILImage
        tmp = PILImage.fromarray((img_b * 255).astype(np.uint8)).resize((Wa, Ha))
        img_b = np.array(tmp, dtype=np.float32) / 255.0

    H, W = img_a.shape[:2]

    # ---- dataset --------------------------------------------------------------
    feat_a, tgt_a, H, W = image_to_dataset(img_a, style=0.0)
    if img_b is not None:
        feat_b, tgt_b, _, _ = image_to_dataset(img_b, style=1.0)
        features = torch.cat([feat_a, feat_b])
        targets  = torch.cat([tgt_a,  tgt_b])
    else:
        features, targets = feat_a, tgt_a

    # ---- model info -----------------------------------------------------------
    model  = ImageMLP()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nNetwork: {N_FEATURES} -> {H1} -> {H2} -> {H3}")
    print(f"Params:  {n_params}  "
          f"({n_params * 4 // 1024} KB float32  /  {n_params // 1024} KB int8)\n")

    # ---- train ----------------------------------------------------------------
    train(model, features, targets)
    model_q = quantize_weights(model)

    # ---- plot -----------------------------------------------------------------
    if img_b is None:
        rec    = render(model,   H, W, style=0.0)
        rec_q  = render(model_q, H, W, style=0.0)

        p_f32  = psnr(rec,   img_a)
        p_int8 = psnr(rec_q, img_a)
        print(f"\nPSNR float32 : {p_f32:.1f} dB")
        print(f"PSNR int8    : {p_int8:.1f} dB")

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        axes[0].imshow(np.clip(img_a, 0, 1));  axes[0].set_title('Original');                 axes[0].axis('off')
        axes[1].imshow(np.clip(rec,   0, 1));  axes[1].set_title(f'Float32  {p_f32:.1f} dB'); axes[1].axis('off')
        axes[2].imshow(np.clip(rec_q, 0, 1));  axes[2].set_title(f'INT8     {p_int8:.1f} dB');axes[2].axis('off')
        fig.suptitle(f'{H1}->{H2}->{H3}  |  {n_params} params  |  {n_params//1024} KB int8 weight file')

    else:
        styles = [0.0, 0.33, 0.67, 1.0]
        labels = ['A (0.0)', '0.33', '0.67', 'B (1.0)']
        fig, axes = plt.subplots(2, len(styles) + 1, figsize=(14, 6))

        axes[0, 0].imshow(np.clip(img_a, 0, 1)); axes[0, 0].set_title('Source A'); axes[0, 0].axis('off')
        axes[1, 0].imshow(np.clip(img_b, 0, 1)); axes[1, 0].set_title('Source B'); axes[1, 0].axis('off')
        axes[0, 0].set_ylabel('float32', fontsize=9)
        axes[1, 0].set_ylabel('int8',    fontsize=9)

        for col, (s, lbl) in enumerate(zip(styles, labels)):
            rec   = render(model,   H, W, style=s)
            rec_q = render(model_q, H, W, style=s)
            axes[0, col + 1].imshow(np.clip(rec,   0, 1)); axes[0, col + 1].set_title(f'style {lbl}'); axes[0, col + 1].axis('off')
            axes[1, col + 1].imshow(np.clip(rec_q, 0, 1)); axes[1, col + 1].set_title(f'int8');         axes[1, col + 1].axis('off')

        fig.suptitle('Style interpolation — top: float32  bottom: int8 quantized')

    plt.tight_layout()
    out = 'neural_image_baseline.png'
    plt.savefig(out, dpi=150)
    print(f"\nSaved: {out}")
    plt.show()


if __name__ == '__main__':
    main()
