import torch
import torch.nn as nn
import numpy as np
import json
from ucimlrepo import fetch_ucirepo

H1     = 64
H2     = 32
EPOCHS = 1500
LR     = 0.001

CLASS_COLOURS = {
    0: [200, 0,   0  ],
    1: [0,   200, 0  ],
    2: [0,   0,   200],
    3: [200, 200, 0  ],
    4: [200, 0,   200],
    5: [0,   200, 200],
    6: [200, 120, 0  ],
}

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

colours = np.array([CLASS_COLOURS[i] for i in range(n_classes)], dtype=np.float32) / 255.0
targets = torch.tensor(colours[y_raw], dtype=torch.float32)
inputs  = torch.tensor(X, dtype=torch.float32)

model = nn.Sequential(
    nn.Linear(n_features, H1), nn.ReLU(),
    nn.Linear(H1, H2),         nn.ReLU(),
    nn.Linear(H2, 3),
)

optimiser = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn   = nn.MSELoss()

for epoch in range(EPOCHS):
    optimiser.zero_grad()
    loss = loss_fn(model(inputs), targets)
    loss.backward()
    optimiser.step()
    if (epoch + 1) % 250 == 0:
        print(f"  Epoch {epoch+1}/{EPOCHS}  loss={loss.item():.6f}")

with torch.no_grad():
    raw = model(inputs).numpy()

# per-channel normalisation
z_offsets = []
z_scales  = []
for k in range(3):
    ch_min = float(raw[:, k].min())
    ch_max = float(raw[:, k].max())
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
    ("W1", model[0].weight.detach().numpy()),
    ("b1", model[0].bias.detach().numpy()),
    ("W2", model[2].weight.detach().numpy()),
    ("b2", model[2].bias.detach().numpy()),
    ("W3", model[4].weight.detach().numpy()),
    ("b3", model[4].bias.detach().numpy()),
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