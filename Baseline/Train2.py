import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import colorsys
from ucimlrepo import fetch_ucirepo

H1     = 64
H2     = 32
EPOCHS = 3000
LR     = 0.001

def make_colours(n):
    colours = {}
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb(i / n, 1.0, 1.0)
        colours[i] = [int(r * 255), int(g * 255), int(b * 255)]
    return colours

print("Fetching dry bean dataset...")
data         = fetch_ucirepo(id=602)
X_raw        = data.data.features.values.astype(np.float32)
y_names      = data.data.targets.values.ravel()
classes      = sorted(list(set(y_names)))
class_to_id  = {c: i for i, c in enumerate(classes)}
y_raw        = np.array([class_to_id[y] for y in y_names], dtype=int)
feature_cols = list(data.data.features.columns)

x_min = X_raw.min(axis=0)
x_max = X_raw.max(axis=0)
X     = (X_raw - x_min) / (x_max - x_min + 1e-8)

n_features = X.shape[1]
n_classes  = len(classes)

CLASS_COLOURS = make_colours(n_classes)

print(f"Dataset: {len(X)} samples  {n_features} features  {n_classes} classes")
print(f"Architecture: {n_features} -> {H1} -> {H2} -> {n_classes} (train) -> 3 (display)")
for i, name in enumerate(classes):
    print(f"  {name}: {CLASS_COLOURS[i]}")

colour_tensor = torch.tensor(
    [[c / 255.0 for c in CLASS_COLOURS[i]] for i in range(n_classes)],
    dtype=torch.float32
)
labels  = torch.tensor(y_raw, dtype=torch.long)
inputs  = torch.tensor(X, dtype=torch.float32)

# Network outputs n_classes logits for cross entropy training
# After training we add a fixed colour projection layer
backbone = nn.Sequential(
    nn.Linear(n_features, H1), nn.ReLU(),
    nn.Linear(H1, H2),         nn.ReLU(),
    nn.Linear(H2, n_classes),
)

optimiser = torch.optim.Adam(backbone.parameters(), lr=LR)

print("Training...")
for epoch in range(EPOCHS):
    optimiser.zero_grad()
    logits = backbone(inputs)

    ce_loss = F.cross_entropy(logits, labels)

    # softmax probabilities weighted blend of class colours
    probs       = F.softmax(logits, dim=1)
    blended     = probs @ colour_tensor
    target_rgb  = colour_tensor[labels]
    colour_loss = F.mse_loss(blended, target_rgb)

    loss = ce_loss + 2.0 * colour_loss
    loss.backward()
    optimiser.step()

    if (epoch + 1) % 500 == 0:
        with torch.no_grad():
            acc = (logits.argmax(dim=1) == labels).float().mean().item()
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={loss.item():.4f}  acc={acc*100:.1f}%")

# Build the final 3-output model by appending colour_tensor as a fixed linear layer
# output = softmax(backbone(x)) @ colour_tensor
# This is equivalent to a linear layer with weights = colour_tensor.T and no bias
# We bake this into a combined model so the FPGA just runs one forward pass

with torch.no_grad():
    # final layer: (n_classes) -> 3 using colour_tensor as fixed weights
    # colour_tensor shape: (n_classes, 3) — each row is one class colour
    # we want W3_extra of shape (3, n_classes) for nn.Linear
    W_colour = colour_tensor.T.numpy()  # shape (3, n_classes)
    b_colour = np.zeros(3)

# Replace backbone's last layer output with colour projection
# New architecture: n_features -> H1 -> H2 -> 3
# where the last layer = colour projection through softmax
# Since we cannot easily bake softmax into a linear layer exactly,
# we instead train a small linear head on top of the H2 features directly

# Retrain just the final RGB head on frozen backbone features
print("\nFitting RGB output head...")
backbone.eval()
with torch.no_grad():
    h2_features = []
    for i in range(0, len(inputs), 512):
        batch = inputs[i:i+512]
        h = batch
        for layer in list(backbone.children())[:4]:
            h = layer(h)
        h2_features.append(h)
    h2_features = torch.cat(h2_features, dim=0)

rgb_head = nn.Linear(H2, 3, bias=True)
rgb_optim = torch.optim.Adam(rgb_head.parameters(), lr=0.01)
target_rgb = colour_tensor[labels]

for epoch in range(1000):
    rgb_optim.zero_grad()
    out  = rgb_head(h2_features)
    loss = F.mse_loss(out, target_rgb)
    loss.backward()
    rgb_optim.step()
    if (epoch + 1) % 200 == 0:
        print(f"  RGB head epoch {epoch+1}/1000  loss={loss.item():.6f}")

# Combine into one Sequential model: n_features -> H1 -> H2 -> 3
combined = nn.Sequential(
    backbone[0], backbone[1],
    backbone[2], backbone[3],
    rgb_head,
)

with torch.no_grad():
    raw = combined(inputs).numpy()

z_offsets, z_scales = [], []
for k in range(3):
    ch_min  = float(raw[:, k].min())
    ch_max  = float(raw[:, k].max())
    margin  = 0.05 * (ch_max - ch_min)
    ch_min -= margin
    ch_max += margin
    z_offsets.append(ch_min)
    z_scales.append(255.0 / (ch_max - ch_min))

print("\nPer-channel scaling:")
for k in range(3):
    print(f"  channel {k}: offset={z_offsets[k]:.4f}  scale={z_scales[k]:.4f}")

def to_q44_hex(arr):
    out = []
    for v in np.array(arr).flatten():
        q = max(-128, min(127, int(round(v * 16))))
        out.append(f"{q & 0xFF:02X}")
    return out

def to_q1616_hex(v):
    q = int(round(v * 65536))
    q = max(-(1 << 31), min((1 << 31) - 1, q))
    return f"{q & 0xFFFFFFFF:08X}"

def to_q816_hex(v):
    q = int(round(v * 65536))
    q = max(0, min((1 << 32) - 1, q))
    return f"{q & 0xFFFFFFFF:08X}"

layers = [
    ("W1", combined[0].weight.detach().numpy()),
    ("b1", combined[0].bias.detach().numpy()),
    ("W2", combined[2].weight.detach().numpy()),
    ("b2", combined[2].bias.detach().numpy()),
    ("W3", combined[4].weight.detach().numpy()),
    ("b3", combined[4].bias.detach().numpy()),
]

with open("weights.hex", "w") as f:
    for k in range(3):
        f.write(f"// z_offset_{k} Q16.16\n")
        f.write(to_q1616_hex(z_offsets[k]) + "\n\n")
        f.write(f"// z_scale_{k} Q8.16\n")
        f.write(to_q816_hex(z_scales[k]) + "\n\n")
    for name, values in layers:
        f.write(f"// {name} shape={np.array(values).shape}\n")
        f.write(" ".join(to_q44_hex(values)) + "\n\n")

meta = {
    "dataset":       "dry_bean",
    "n_features":    n_features,
    "n_classes":     n_classes,
    "h1_size":       H1,
    "h2_size":       H2,
    "feature_cols":  feature_cols,
    "class_names":   classes,
    "class_colours": CLASS_COLOURS,
    "x_min":         x_min.tolist(),
    "x_max":         x_max.tolist(),
    "z_offsets":     z_offsets,
    "z_scales":      z_scales,
}

with open("weights_meta.json", "w") as f:
    json.dump(meta, f, indent=2)

print("\nWeight clipping check:")
for name, values in layers:
    arr     = np.array(values)
    clipped = int(np.sum(np.abs(arr) > 7.9375))
    print(f"  {name}: min={arr.min():.3f}  max={arr.max():.3f}  clipped={clipped}")

print("\nSaved weights.hex and weights_meta.json")