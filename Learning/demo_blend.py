#!/usr/bin/env python3
"""
Demo: train the FPGA-matched MLP to blend a butterfly wing with a peacock feather.

Usage:
  python demo_blend.py                          # procedural fallback
  python demo_blend.py butterfly.jpg peacock.jpg  # real photos

Real photos are center-cropped to square and resized to 256×256 before training.
"""

import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

# ---------------------------------------------------------------------------
# Architecture — mirrors mlp_pipeline.sv exactly
# ---------------------------------------------------------------------------
N_FEATURES = 16
FOURIER_FREQS = [1.0, 2.0, 4.0]   # 3 freqs × sin+cos × x+y = 12 slots (features 2-13)
                                    # feature 14 = style,  feature 15 = unused

class ImageMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 64), nn.ReLU(),
            nn.Linear(64,         32), nn.ReLU(),
            nn.Linear(32,          3), nn.Sigmoid(),
        )
    def forward(self, x):
        return self.net(x)


def build_features(xs, ys, style):
    parts = [xs.unsqueeze(1), ys.unsqueeze(1)]
    for f in FOURIER_FREQS:
        parts += [torch.sin(2*torch.pi*f*xs).unsqueeze(1),
                  torch.cos(2*torch.pi*f*xs).unsqueeze(1)]
    for f in FOURIER_FREQS:
        parts += [torch.sin(2*torch.pi*f*ys).unsqueeze(1),
                  torch.cos(2*torch.pi*f*ys).unsqueeze(1)]
    parts.append(torch.full((len(xs), 1), float(style)))
    parts.append(torch.zeros(len(xs), 1))
    return torch.cat(parts, dim=1)


def image_to_dataset(img, style):
    H, W, _ = img.shape
    yi, xi = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    xs = torch.tensor(xi.flatten() / (W-1), dtype=torch.float32)
    ys = torch.tensor(yi.flatten() / (H-1), dtype=torch.float32)
    return build_features(xs, ys, style), torch.tensor(img.reshape(-1, 3), dtype=torch.float32)


# ---------------------------------------------------------------------------
# Real-image loader — center-crops to square, resizes to HxW, full bleed
# ---------------------------------------------------------------------------
def load_square(path, size=256, box=None):
    """
    Load and resize to size×size.
    box: (left, top, right, bottom) in 0-1 relative coords — crops before resize.
         None → center-crop to square.
    """
    from PIL import Image
    img  = Image.open(path).convert('RGB')
    W, H = img.size
    if box is not None:
        l, t, r, b = box
        img = img.crop((int(l*W), int(t*H), int(r*W), int(b*H)))
    else:
        side = min(W, H)
        left = (W - side) // 2
        top  = (H - side) // 2
        img  = img.crop((left, top, left + side, top + side))
    img = img.resize((size, size), Image.LANCZOS)
    return np.array(img, dtype=np.float32) / 255.0


# ---------------------------------------------------------------------------
# Procedural images — fallback when no photos are provided
# ---------------------------------------------------------------------------
def make_morpho_butterfly(H=256, W=256):
    """Macro crop of Morpho butterfly wing scales — full bleed, no outline mask."""
    yi, xi = np.mgrid[0:H, 0:W]
    x = xi / (W-1)
    y = yi / (H-1)

    # Two overlapping iridescence waves simulate thin-film interference
    irid  = 0.55 + 0.45 * np.sin((x * 8 + y * 5) * np.pi)
    irid2 = 0.50 + 0.50 * np.cos((x * 3 - y * 6) * np.pi * 1.4)

    # Fine scale/vein grid — tighter than the wing outline version
    vein = np.clip(np.cos(x * np.pi * 9)**12 + np.cos(y * np.pi * 7)**12, 0, 1) * 0.75

    blue = np.clip(irid * irid2 - vein * 0.35, 0, 1)

    # Warm highlights at top-right (mimics directional light on scales)
    highlight = np.clip((x + (1-y) - 0.9) * 1.5, 0, 1) * 0.25

    R = np.clip(blue * 0.12 + highlight * 0.6, 0, 1)
    G = np.clip(blue * 0.28 + highlight * 0.4, 0, 1)
    B = np.clip(blue * 0.98 - vein * 0.1 + highlight * 0.1, 0, 1)

    return np.stack([R, G, B], axis=-1).astype(np.float32)


def make_peacock_feather(H=256, W=256):
    """Peacock feather eye zoomed so rings bleed to the image edges."""
    cy, cx = H/2, W/2
    yi, xi = np.mgrid[0:H, 0:W]
    # scale so r=1 sits at the image edge (not inside it)
    y = (yi - cy) / (H * 0.40)
    x = (xi - cx) / (W * 0.40)
    r = np.sqrt(x**2 + y**2)
    angle = np.arctan2(y, x)

    # radiating barb texture (visible across the whole frame)
    barb = 0.15 * (0.5 + 0.5*np.sin(angle * 28)) * np.clip(r * 0.6, 0, 1)

    # base: golden-brown barbs that fill beyond the rings
    R = np.full((H, W), 0.50); G = np.full((H, W), 0.32); B = np.full((H, W), 0.08)

    def ring(r_in, r_out, dr, dr_g, dg_g, db_g):
        t = np.clip((r_out - r) / dr, 0, 1) * (r > r_in)
        return t * dr_g, t * dg_g, t * db_g

    dR,dG,dB = ring(0.68, 1.20, 0.40,  -0.20,  0.28,  0.20)   # outer iridescent green
    R+=dR; G+=dG; B+=dB
    dR,dG,dB = ring(0.48, 0.75, 0.28,  -0.22,  0.22,  0.18)   # mid green
    R+=dR; G+=dG; B+=dB
    dR,dG,dB = ring(0.28, 0.54, 0.26,   0.50,  0.38, -0.28)   # gold ring
    R+=dR; G+=dG; B+=dB
    dR,dG,dB = ring(0.13, 0.33, 0.20,  -0.35,  0.25,  0.45)   # teal ring
    R+=dR; G+=dG; B+=dB
    dR,dG,dB = ring(0.00, 0.16, 0.16,  -0.38, -0.20,  0.55)   # dark blue centre
    R+=dR; G+=dG; B+=dB

    R += barb; G += barb * 0.55
    shimmer = 0.07 * np.sin(r * 16 - angle * 2)
    G += shimmer; B += shimmer * 0.6

    return np.clip(np.stack([R, G, B], axis=-1), 0, 1).astype(np.float32)


# ---------------------------------------------------------------------------
# INT8 simulation
# ---------------------------------------------------------------------------
def quantize(model):
    """Simulate Q4.4 weight quantization: round to nearest 1/16."""
    q = copy.deepcopy(model)
    with torch.no_grad():
        for layer in q.net:
            if isinstance(layer, nn.Linear):
                layer.weight.data = (layer.weight.data * 16).round().clamp(-128, 127) / 16.0
    return q


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train(model, feats, tgts, epochs=4000, lr=1e-3, batch=8192):
    opt   = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    N     = len(feats)
    for ep in range(1, epochs+1):
        idx  = torch.randperm(N)[:batch]
        loss = nn.functional.mse_loss(model(feats[idx]), tgts[idx])
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 500 == 0:
            print(f"  {ep:5d}/{epochs}  loss={loss.item():.5f}")


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
@torch.no_grad()
def render(model, H, W, style):
    yi, xi = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    xs = torch.tensor(xi.flatten()/(W-1), dtype=torch.float32)
    ys = torch.tensor(yi.flatten()/(H-1), dtype=torch.float32)
    return model(build_features(xs, ys, style)).numpy().reshape(H, W, 3)


def psnr(a, b):
    mse = np.mean((a-b)**2)
    return float('inf') if mse==0 else -10*np.log10(mse+1e-10)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    H, W = 256, 256
    args  = sys.argv[1:]

    if len(args) >= 2:
        print(f"Loading real photos:\n  A: {args[0]}\n  B: {args[1]}")
        # peacok.jpg: butterfly wing fills upper-right — crop to wing, skip hand/sky
        # butterfly.jpg: peacock eye is centred — default square crop
        img_a = load_square(args[0], H, box=(0.25, 0.05, 0.95, 0.75))
        img_b = load_square(args[1], H)
        label_a, label_b = args[0], args[1]
    else:
        print("No photos supplied — using procedural fallback.")
        img_a = make_morpho_butterfly(H, W)
        img_b = make_peacock_feather(H, W)
        label_a, label_b = 'Morpho butterfly', 'Peacock feather'

    # Save source images for reference
    plt.imsave('butterfly_wing.png', img_a)
    plt.imsave('peacock_feather.png', img_b)
    print("Saved source images: butterfly_wing.png  peacock_feather.png")

    feat_a, tgt_a = image_to_dataset(img_a, style=0.0)
    feat_b, tgt_b = image_to_dataset(img_b, style=1.0)
    feats = torch.cat([feat_a, feat_b])
    tgts  = torch.cat([tgt_a,  tgt_b])

    n_params = sum(p.numel() for p in ImageMLP().parameters())
    print(f"\nNetwork: {N_FEATURES}->64->32->3  |  {n_params} params  |  {n_params//1024} KB int8\n")

    model = ImageMLP()
    print("Training on both images simultaneously (style=0 → butterfly, style=1 → peacock)...")
    train(model, feats, tgts)
    model_q = quantize(model)

    # PSNR
    r_a = render(model, H, W, 0.0)
    r_b = render(model, H, W, 1.0)
    print(f"\nPSNR butterfly : {psnr(r_a, img_a):.1f} dB")
    print(f"PSNR peacock   : {psnr(r_b, img_b):.1f} dB")

    # ---------------------------------------------------------------------------
    # Plot: source images + interpolation strip (float32 and int8)
    # ---------------------------------------------------------------------------
    styles = [0.0, 0.25, 0.5, 0.75, 1.0]
    style_labels = ['butterfly\n(style=0.0)', '0.25', '0.5', '0.75', 'peacock\n(style=1.0)']

    fig = plt.figure(figsize=(16, 8))
    gs  = fig.add_gridspec(3, 5, hspace=0.08, wspace=0.04)

    # Row 0: source images (span middle 3 cols to leave label space)
    ax_src_a = fig.add_subplot(gs[0, 0])
    ax_src_b = fig.add_subplot(gs[0, 4])
    ax_src_a.imshow(img_a); ax_src_a.set_title(f'Source A\n{label_a}', fontsize=8); ax_src_a.axis('off')
    ax_src_b.imshow(img_b); ax_src_b.set_title(f'Source B\n{label_b}', fontsize=8); ax_src_b.axis('off')
    for col in [1,2,3]:
        fig.add_subplot(gs[0, col]).set_visible(False)

    # Rows 1-2: float32 and int8 blends
    row_labels = ['float32', 'int8']
    for row, mdl in enumerate([model, model_q]):
        for col, (s, lbl) in enumerate(zip(styles, style_labels)):
            ax = fig.add_subplot(gs[row+1, col])
            img = render(mdl, H, W, s)
            ax.imshow(np.clip(img, 0, 1))
            if row == 0:
                ax.set_title(lbl, fontsize=8)
            if col == 0:
                ax.set_ylabel(row_labels[row], fontsize=9)
            ax.axis('off')

    fig.suptitle(
        f'FPGA neural blend: Morpho butterfly  ←→  Peacock feather\n'
        f'{n_params} params  |  {n_params//1024} KB int8 weight file  |  '
        f'top=float32  bottom=int8',
        fontsize=11
    )

    out = 'blend_result.png'
    plt.savefig(out, dpi=160, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close()

    # ---------------------------------------------------------------------------
    # Full-resolution 640×480 render — matches FPGA HDMI output exactly
    # ---------------------------------------------------------------------------
    FW, FH = 640, 480
    print(f"\nRendering 640×480 frames (int8 model, matches FPGA output)...")
    full_styles = [0.0, 0.25, 0.5, 0.75, 1.0]

    fig2, axes2 = plt.subplots(1, len(full_styles), figsize=(20, 6))
    for ax, s in zip(axes2, full_styles):
        frame = render(model_q, FH, FW, s)
        ax.imshow(np.clip(frame, 0, 1))
        ax.set_title(f'style = {s}', fontsize=10)
        ax.axis('off')
        plt.imsave(f'full_res_style_{s:.2f}.png', np.clip(frame, 0, 1))

    fig2.suptitle(f'640×480 INT8 render — what the FPGA will output', fontsize=12)
    plt.tight_layout()
    full_out = 'blend_fullres.png'
    plt.savefig(full_out, dpi=120, bbox_inches='tight')
    print(f"Saved: {full_out}  (+ individual full_res_style_*.png frames)")
    plt.show()
