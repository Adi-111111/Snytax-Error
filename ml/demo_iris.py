#!/usr/bin/env python3
"""
Software baseline for the FPGA Iris decision boundary visualiser.

Mirrors hardware exactly:
  16 inputs -> 64 (ReLU) -> 32 (ReLU) -> 3 (softmax -> RGB)
  axis_x_select / axis_y_select assign any feature slot to the screen axes
  feature_regs holds held values for the remaining features (AXI-Lite registers)

Output colour:  R = P(setosa)   G = P(versicolor)   B = P(virginica)
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler

N_FEATURES   = 16
FEATURE_NAMES = ['sepal length', 'sepal width', 'petal length', 'petal width']
CLASS_NAMES   = ['setosa (R)', 'versicolor (G)', 'virginica (B)']
CLASS_COLORS  = ['#ff4444', '#44ff44', '#4488ff']

# ---------------------------------------------------------------------------
# Architecture — mirrors mlp_pipeline.sv exactly
# ---------------------------------------------------------------------------
class IrisMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(N_FEATURES, 64), nn.ReLU(),
            nn.Linear(64,         32), nn.ReLU(),
            nn.Linear(32,          3),
        )

    def forward(self, x):
        return self.net(x)                          # raw logits — for training

    def predict_color(self, x):
        return torch.softmax(self.net(x), dim=-1)   # RGB in [0,1] — for rendering


# ---------------------------------------------------------------------------
# input_builder — mirrors input_builder.sv exactly
# ---------------------------------------------------------------------------
def build_features(xs, ys, axis_x_select, axis_y_select, feature_regs):
    """
    xs, ys         : normalised screen coords [0,1],  shape (N,)
    axis_x_select  : feature index assigned to the screen x axis
    axis_y_select  : feature index assigned to the screen y axis
    feature_regs   : (N_FEATURES,) held values for non-axis features
    """
    N     = len(xs)
    feats = np.tile(feature_regs.astype(np.float32), (N, 1))
    feats[:, axis_x_select] = xs
    feats[:, axis_y_select] = ys
    return torch.tensor(feats, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Hardware-accurate calibration — mirrors mlp_pipeline.sv integer arithmetic
# exactly, then computes the z_offset/z_scale needed for regfile[72]/[73].
#
# Hardware pipeline:
#   rgb = clamp( (l3_acc - z_offset) * z_scale >> 16,  0, 255 )
#
# l3_acc is a 48-bit fixed-point accumulator from Q1.15 inputs × int8 weights.
# z_offset and z_scale must be calibrated per trained network — the reset
# defaults (0 and 65536) will produce garbage output for any new weight set.
# ---------------------------------------------------------------------------
def hw_calibrate(model_q, feature_regs_float, axis_x=0, axis_y=1, grid=80):
    """
    Sweep a grid of axis inputs, run the exact fixed-point forward pass,
    find the l3_acc range, and return (z_offset, z_scale) for the hardware.
    """
    BIAS = np.int64(32767)   # BIAS_CONST_L1 / BIAS_CONST_L23 in hardware

    # Extract int8 weight matrices from the quantized model.
    # Bias is appended as the last column — hardware injects BIAS_CONST for it.
    def extract(layer):
        w  = layer.weight.data.numpy()
        b  = layer.bias.data.numpy()
        w8 = np.round(w * 16).clip(-128, 127).astype(np.int64)  # Q4.4
        b8 = np.round(b * 16).clip(-128, 127).astype(np.int64)  # Q4.4
        return np.hstack([w8, b8.reshape(-1, 1)])

    linears = [l for l in model_q.net if isinstance(l, nn.Linear)]
    W1 = extract(linears[0])   # (64, 17)
    W2 = extract(linears[1])   # (32, 65)
    W3 = extract(linears[2])   # ( 3, 33)

    # Build calibration input grid in Q1.15 — sweep full axis range
    axis_q15 = np.linspace(0, 32767, grid, dtype=np.int64)
    gx, gy   = np.meshgrid(axis_q15, axis_q15, indexing='ij')
    N        = grid * grid

    held = np.round(
        np.clip(feature_regs_float, 0, 0.9999) * 32768
    ).astype(np.int64)

    x_in = np.tile(held, (N, 1))
    x_in[:, axis_x] = gx.flatten()
    x_in[:, axis_y] = gy.flatten()

    # 32-bit wrap — mirrors l1_buffer / l2_buffer[31:0] truncation in hardware
    def trunc32(v):
        v = v & 0xFFFFFFFF
        return np.where(v >= 0x80000000, v - 0x100000000, v)

    # Layer 1  (N_FEATURES=16 inputs + bias at col 16)
    l1_acc  = x_in @ W1[:, :N_FEATURES].T + BIAS * W1[:, N_FEATURES]
    l1_buf  = trunc32(np.maximum(0, l1_acc))

    # Layer 2  (H1=64 inputs + bias at col 64)
    H1 = W1.shape[0]
    l2_acc  = l1_buf @ W2[:, :H1].T + BIAS * W2[:, H1]
    l2_buf  = trunc32(np.maximum(0, l2_acc))

    # Layer 3  (H2=32 inputs + bias at col 32)
    H2 = W2.shape[0]
    l3_acc  = l2_buf @ W3[:, :H2].T + BIAS * W3[:, H2]

    g_max  = int(np.abs(l3_acc).max())
    # z_shift = number of bits to drop so max maps to ~255
    # g_max >> z_shift = 255  →  z_shift = ceil(log2(g_max / 255))
    z_shift = max(0, int(np.ceil(np.log2(g_max / 255)))) if g_max > 255 else 0

    return z_shift, l3_acc


# ---------------------------------------------------------------------------
# INT8 quantization — simulates hardware weight precision
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
# Training
# ---------------------------------------------------------------------------
def train(model, X, y, epochs=3000, lr=1e-3):
    X_t     = torch.tensor(X, dtype=torch.float32)
    y_t     = torch.tensor(y, dtype=torch.long)
    opt     = optim.Adam(model.parameters(), lr=lr)
    sched   = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(1, epochs + 1):
        logits = model(X_t)
        loss   = loss_fn(logits, y_t)
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 500 == 0:
            acc = (logits.argmax(1) == y_t).float().mean()
            print(f"  {ep:5d}/{epochs}  loss={loss.item():.4f}  acc={acc:.3f}")


# ---------------------------------------------------------------------------
# Render decision boundary
# ---------------------------------------------------------------------------
@torch.no_grad()
def render(model, axis_x, axis_y, feature_regs, H=480, W=640):
    yi, xi = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    xs = xi.flatten() / (W - 1)
    ys = yi.flatten() / (H - 1)
    feats = build_features(xs, ys, axis_x, axis_y, feature_regs)
    return model.predict_color(feats).numpy().reshape(H, W, 3)


def accuracy(model, X, y):
    with torch.no_grad():
        pred = model(torch.tensor(X, dtype=torch.float32)).argmax(1).numpy()
    return (pred == y).mean()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    # Load + normalise to [0,1]
    iris    = load_iris()
    scaler  = MinMaxScaler()
    X_norm  = scaler.fit_transform(iris.data.astype(np.float32))
    y       = iris.target

    # Pad to 16 features — iris uses slots 0-3, rest stay 0
    X16 = np.zeros((len(X_norm), N_FEATURES), dtype=np.float32)
    X16[:, :4] = X_norm

    # feature_regs: hold non-axis features at their dataset mean
    feature_regs = np.zeros(N_FEATURES, dtype=np.float32)
    feature_regs[:4] = X_norm.mean(axis=0)

    # ---- train ---------------------------------------------------------------
    model = IrisMLP()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Network: {N_FEATURES}->64->32->3  |  {n_params} params  |  {n_params//1024} KB int8")
    print(f"Dataset: Iris  {len(X_norm)} samples  3 classes\n")
    train(model, X16, y)
    model_q = quantize(model)

    acc_f32  = accuracy(model,   X16, y)
    acc_int8 = accuracy(model_q, X16, y)
    print(f"\nAccuracy  float32={acc_f32:.1%}   int8={acc_int8:.1%}")

    # ---- hardware calibration ------------------------------------------------
    print("\nRunning hardware-accurate calibration (Q1.15 × int8 forward pass)...")
    z_shift, l3_acc = hw_calibrate(
        model_q, feature_regs, axis_x=0, axis_y=1
    )
    print(f"  l3_acc range : [{l3_acc.min():.3e},  {l3_acc.max():.3e}]")
    print(f"  z_shift      : {z_shift}  (drops {z_shift} bits, maps max → {int(l3_acc.max()) >> z_shift})")
    print(f"\n  AXI-Lite writes needed (after adding regfile[74] to hardware):")
    print(f"    axi_write(9'h128, 32'h{z_shift:08X});  // z_shift → regfile[74]")
    print(f"    axi_write(9'h120, 32'h00000000);        // z_offset = 0")
    print(f"    axi_write(9'h124, 32'h00000001);        // z_scale  = 1")

    # ---- four axis-pair views ------------------------------------------------
    axis_pairs = [
        (2, 3, 'petal length  vs  petal width'),    # clearest boundary
        (0, 1, 'sepal length  vs  sepal width'),    # messier — shows contrast
        (0, 2, 'sepal length  vs  petal length'),
        (1, 3, 'sepal width   vs  petal width'),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    for col, (ax_x, ax_y, title) in enumerate(axis_pairs):
        regs = feature_regs.copy()
        for row, mdl in enumerate([model, model_q]):
            boundary = render(mdl, ax_x, ax_y, regs)
            axes[row, col].imshow(boundary, origin='upper',
                                  extent=[0, 1, 1, 0], aspect='auto')
            for cls in range(3):
                mask = y == cls
                axes[row, col].scatter(X_norm[mask, ax_x], X_norm[mask, ax_y],
                                       c=CLASS_COLORS[cls], s=18,
                                       edgecolors='white', linewidths=0.4, zorder=5)
            if row == 0:
                axes[row, col].set_title(title, fontsize=9)
            if col == 0:
                axes[row, col].set_ylabel('float32' if row == 0 else 'int8', fontsize=9)
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])

    fig.suptitle(
        f'FPGA Iris decision boundary — axis_x_select / axis_y_select demo\n'
        f'{n_params} params  |  {n_params//1024} KB int8  |  '
        f'R=setosa  G=versicolor  B=virginica  |  '
        f'float32 {acc_f32:.1%}  /  int8 {acc_int8:.1%}',
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig('iris_boundary.png', dpi=150, bbox_inches='tight')
    print("Saved: iris_boundary.png")
    plt.close()

    # ---- 640x480 full-res render — sepal axes, petal features held -----------
    print("\nRendering 640×480 (sepal length vs sepal width, int8)...")
    full = render(model_q, 0, 1, feature_regs, H=480, W=640)

    fig2, ax2 = plt.subplots(figsize=(10, 7.5))
    ax2.imshow(full, origin='upper', extent=[0, 1, 1, 0], aspect='auto')
    for cls in range(3):
        mask = y == cls
        ax2.scatter(X_norm[mask, 0], X_norm[mask, 1],
                    c=CLASS_COLORS[cls], s=25,
                    edgecolors='white', linewidths=0.5, zorder=5,
                    label=CLASS_NAMES[cls])
    ax2.set_xlabel('sepal length (normalised)'); ax2.set_ylabel('sepal width (normalised)')
    ax2.set_title(
        f'640×480 INT8 — what the FPGA outputs\n'
        f'axis_x_select=0 (sepal length)   axis_y_select=1 (sepal width)\n'
        f'petal length held={feature_regs[2]:.2f}   petal width held={feature_regs[3]:.2f}'
    )
    ax2.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('iris_fullres.png', dpi=120)
    print("Saved: iris_fullres.png")
    plt.close()

    # ---- sweep: petal length 0→1, sepal axes fixed ---------------------------
    # Sepal boundary is sensitive to held petal values — produces dramatic movement
    print("\nRendering held-feature sweep (petal length 0→1, sepal axes fixed)...")
    sweep_vals = np.linspace(0, 1, 5)
    fig3, axes3 = plt.subplots(1, 5, figsize=(20, 4))

    for ax3, sv in zip(axes3, sweep_vals):
        regs = feature_regs.copy()
        regs[2] = sv                               # sweep petal length
        boundary = render(model_q, 0, 1, regs, H=240, W=320)
        ax3.imshow(boundary, origin='upper', extent=[0, 1, 1, 0], aspect='auto')
        for cls in range(3):
            mask = y == cls
            ax3.scatter(X_norm[mask, 0], X_norm[mask, 1],
                        c=CLASS_COLORS[cls], s=8,
                        edgecolors='white', linewidths=0.3, zorder=5)
        ax3.set_title(f'petal_len = {sv:.2f}', fontsize=9)
        ax3.set_xlabel('sepal length', fontsize=7)
        if ax3 is axes3[0]:
            ax3.set_ylabel('sepal width', fontsize=7)
        ax3.set_xticks([]); ax3.set_yticks([])

    fig3.suptitle(
        'Held feature sweep — petal length 0→1  (sepal axes fixed)\n'
        'Each frame = one AXI-Lite register write on the FPGA',
        fontsize=10
    )
    plt.tight_layout()
    plt.savefig('iris_sweep.png', dpi=150, bbox_inches='tight')
    print("Saved: iris_sweep.png")
    plt.show()
