import colorsys
import json
import os
import struct
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random

# config
SEED = 1234
DETERMINISTIC = True
H1_SIZE = 32
H2_SIZE = 16
N_EPOCHS = 2000
LEARNING_RATE = 0.003
WEIGHT_DECAY = 1e-3

COLOUR_LOSS_WEIGHT = 5.0

def set_reproducible_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if DETERMINISTIC:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def class_colour(class_idx, n_classes):
    hue = class_idx / max(1, n_classes)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return [int(round(r * 255)), int(round(g * 255)), int(round(b * 255))]

class DecisionSurfaceMLP(nn.Module):

    def __init__(self, n_features, h1, h2, n_classes):
        super().__init__()
        self.fc1 = nn.Linear(n_features, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3_rgb = nn.Linear(h2, 3)
        self.fc3_class = nn.Linear(h2, n_classes)

    def forward(self, x):
        a1 = torch.relu(self.fc1(x))
        a2 = torch.relu(self.fc2(a1))
        rgb = self.fc3_rgb(a2)
        logits = self.fc3_class(a2)
        return rgb, logits


def load_dataset(csv_path):
    df = pd.read_csv(csv_path)
    if df.shape[1] < 2:
        raise ValueError("dataset needs at least one feature column and a label column")

    label_col = df.columns[-1]
    feature_cols = list(df.columns[:-1])

    X = df[feature_cols].to_numpy(dtype=np.float64)
    labels_raw = df[label_col].astype(str).to_numpy()
    class_names = sorted(set(labels_raw.tolist()))
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    y = np.array([class_to_idx[c] for c in labels_raw], dtype=np.int64)

    feature_min = X.min(axis=0)
    feature_max = X.max(axis=0)
    span = feature_max - feature_min
    span[span == 0] = 1.0
    X_norm = np.clip((X - feature_min) / span, 0.0, 1.0)

    return {
        "X": X_norm,
        "y": y,
        "feature_names": feature_cols,
        "feature_min": feature_min.tolist(),
        "feature_max": feature_max.tolist(),
        "class_names": class_names,
        "n_features": len(feature_cols),
        "n_classes": len(class_names),
    }



def train(csv_path, output_dir, h1=H1_SIZE, h2=H2_SIZE, n_epochs=N_EPOCHS, progress_cb=None):
    set_reproducible_seed()
    os.makedirs(output_dir, exist_ok=True)

    data = load_dataset(csv_path)
    n = data["n_features"]
    n_classes = data["n_classes"]

    X_np = data["X"].astype(np.float32)
    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.long)

    class_colours_list = [class_colour(i, n_classes) for i in range(n_classes)]
    targets_rgb_unit = torch.tensor(class_colours_list, dtype=torch.float32) / 255.0
    y_rgb = targets_rgb_unit[y]

    model = DecisionSurfaceMLP(n, h1, h2, n_classes)
    optimiser = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    for epoch in range(n_epochs):
        optimiser.zero_grad()
        rgb_pred, logits = model(X)

        loss_ce = ce_loss(logits, y)
        loss_colour = mse_loss(rgb_pred, y_rgb)
        loss = loss_ce + COLOUR_LOSS_WEIGHT * loss_colour

        loss.backward()
        optimiser.step()

        if progress_cb is not None and (epoch % 10 == 0 or epoch == n_epochs - 1):
            progress_cb(epoch, n_epochs, float(loss.item()))

    weights = extract_quantised_weights(model, X_np, n, h1, h2)
    write_weights_hex(weights, os.path.join(output_dir, "weights.hex"))

    class_colours = {data["class_names"][i]: class_colours_list[i] for i in range(n_classes)}

    metadata = {
        "n_features": n,
        "h1": h1,
        "h2": h2,
        "feature_names": data["feature_names"],
        "feature_min": data["feature_min"],
        "feature_max": data["feature_max"],
        "class_names": data["class_names"],
        "class_colours": class_colours,
        "z_shift": weights["z_shift"],
        "z_offset": weights["z_offset"],
        "z_scale": weights["z_scale"],
        "fixed_z_min": weights["fixed_z_min"],
        "fixed_z_max": weights["fixed_z_max"],
        "fixed_z_span": weights["fixed_z_span"],
        "q44_saturation_count": weights["saturation_count"],
        "q44_saturation_fraction": weights["saturation_fraction"],
    }
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata



def quantise_q4_4_signed_array(a):
    q = np.rint(np.asarray(a, dtype=np.float64) * 16.0).astype(np.int32)
    return np.clip(q, -128, 127).astype(np.int32)


def signed_q4_4_to_bytes(q):
    q = np.asarray(q, dtype=np.int32)
    return (q & 0xFF).astype(np.uint8)


def rounded_shift_right(x, shift):
    x = np.asarray(x, dtype=np.int64)
    if shift <= 0:
        return x
    add = 1 << (shift - 1)
    pos = x >= 0
    out = np.empty_like(x, dtype=np.int64)
    out[pos] = (x[pos] + add) >> shift
    out[~pos] = -(((-x[~pos]) + add) >> shift)
    return out


def fixed_forward_raw_z3(X_norm, q):
    Xq = np.rint(np.clip(X_norm, 0.0, 1.0) * 32767.0).astype(np.int64)

    W1 = q["W1"].astype(np.int64)
    b1 = q["b1"].astype(np.int64)
    W2 = q["W2"].astype(np.int64)
    b2 = q["b2"].astype(np.int64)
    W3 = q["W3"].astype(np.int64)
    b3 = q["b3"].astype(np.int64)

    a1 = rounded_shift_right(Xq @ W1.T, 15) + b1
    a1 = np.maximum(a1, 0)

    a2 = rounded_shift_right(a1 @ W2.T, 4) + b2
    a2 = np.maximum(a2, 0)

    z3 = rounded_shift_right(a2 @ W3.T, 4) + b3
    return z3.astype(np.int64)


def extract_quantised_weights(model, X_norm, n, h1, h2):
    W1f = model.fc1.weight.detach().cpu().numpy()
    b1f = model.fc1.bias.detach().cpu().numpy()
    W2f = model.fc2.weight.detach().cpu().numpy()
    b2f = model.fc2.bias.detach().cpu().numpy()
    W3f = model.fc3_rgb.weight.detach().cpu().numpy()
    b3f = model.fc3_rgb.bias.detach().cpu().numpy()

    q = {
        "W1": quantise_q4_4_signed_array(W1f),
        "b1": quantise_q4_4_signed_array(b1f),
        "W2": quantise_q4_4_signed_array(W2f),
        "b2": quantise_q4_4_signed_array(b2f),
        "W3": quantise_q4_4_signed_array(W3f),
        "b3": quantise_q4_4_signed_array(b3f),
    }

    all_float = np.concatenate([x.reshape(-1) for x in (W1f, b1f, W2f, b2f, W3f, b3f)])
    saturation_count = int(np.sum(np.rint(all_float * 16.0) != np.clip(np.rint(all_float * 16.0), -128, 127)))
    saturation_fraction = float(saturation_count / max(1, all_float.size))
    if saturation_count:
        max_abs = float(np.max(np.abs(all_float)))
        print(
            f"warning: {saturation_count}/{all_float.size} Q4.4 values saturated "
            f"(max abs float weight/bias {max_abs:.3f})",
            file=sys.stderr,
        )

    raw = bytearray()
    raw.extend(signed_q4_4_to_bytes(q["W1"].reshape(-1)).tobytes())
    raw.extend(signed_q4_4_to_bytes(q["b1"]).tobytes())
    raw.extend(signed_q4_4_to_bytes(q["W2"].reshape(-1)).tobytes())
    raw.extend(signed_q4_4_to_bytes(q["b2"]).tobytes())
    raw.extend(signed_q4_4_to_bytes(q["W3"].reshape(-1)).tobytes())
    raw.extend(signed_q4_4_to_bytes(q["b3"]).tobytes())

    z3 = fixed_forward_raw_z3(X_norm, q)
    z_min = int(z3.min())
    z_max = int(z3.max())
    span = max(1, z_max - z_min)
    margin = max(1, int(round(0.05 * span)))

    z_offset = int(z_min - margin)
    target_span = int(span + 2 * margin)
    z_scale = max(1, int(round(255.0 / max(1, target_span))))
    if target_span > 255:
        z_scale = 1
        print(
            f"warning: quantised z span {target_span} > 255; using z_scale=1, "
            "some clipping may remain. Consider lower COLOUR_LOSS_WEIGHT or more regularisation.",
            file=sys.stderr,
        )

    z_shift = 0

    return {
        "raw": bytes(raw),
        "z_offset": z_offset,
        "z_scale": z_scale,
        "z_shift": z_shift,
        "fixed_z_min": z_min,
        "fixed_z_max": z_max,
        "fixed_z_span": span,
        "saturation_count": saturation_count,
        "saturation_fraction": saturation_fraction,
    }


def write_weights_hex(weights, path):
    header = struct.pack(">i", int(weights["z_offset"])) + struct.pack(">I", int(weights["z_scale"]))
    with open(path, "wb") as f:
        f.write(header)
        f.write(weights["raw"])


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("usage: python3 network.py <dataset.csv> <output_dir>")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2]

    def report(epoch, n_epochs, loss):
        print(f"epoch {epoch}/{n_epochs}  loss={loss:.4f}")

    meta = train(csv_path, output_dir, progress_cb=report)
    print("done.")
    print(json.dumps(meta, indent=2))