import os, sys, json, time, io, struct, threading, subprocess
import numpy as np
from PIL import Image, ImageDraw
from flask import Flask, request, jsonify, Response, send_from_directory

try:
    from pynq import Overlay
    PYNQ_AVAILABLE = True
except ImportError:
    PYNQ_AVAILABLE = False
    print("pynq not found — stub mode")

# paths
OVERLAY_PATH = "maths_accel.bit"
OUTPUT_DIR   = "output"
DATASET_PATH = "output/dataset.csv"
TRAIN_SCRIPT = os.path.join(os.path.dirname(__file__), "network.py")

# AXI-Lite offsets (regfile index * 4)
REG_FEATURE_BASE = 0x000
REG_AXIS_X       = 0x100
REG_AXIS_Y       = 0x104
REG_N_FEATURES   = 0x108
REG_H1_SIZE      = 0x10C
REG_H2_SIZE      = 0x110
REG_WEIGHT_ADDR  = 0x114
REG_WEIGHT_DATA  = 0x118
REG_WEIGHT_WE    = 0x11C
REG_Z_OFFSET     = 0x120
REG_Z_SCALE      = 0x124

L1, L2, L3 = 0, 1, 2

state = {
    "phase":       "idle",
    "error_msg":   "",
    "train_start": 0.0,
    "metadata":    None,
    "weights":     None,   # parsed numpy weights for /predict
    "axis_x":      0,
    "axis_y":      1,
    "features":    [0.5] * 32,
}
state_lock  = threading.Lock()
frame_lock  = threading.Lock()
latest_frame = None

pg   = None
vdma = None

if PYNQ_AVAILABLE:
    try:
        ol   = Overlay(OVERLAY_PATH)
        pg   = ol.pixel_gen_0
        vdma = ol.axi_vdma_0
        vdma.readchannel.start()
        vdma.writechannel.start()
        print("overlay loaded")
    except Exception as e:
        print(f"overlay failed: {e}")
        PYNQ_AVAILABLE = False


def write_reg(offset, value):
    if pg is None:
        return
    pg.write(offset, int(value) & 0xFFFF_FFFF)


def float_to_q15(v):
    # features normalised [0,1] → Q1.15 unsigned
    return int(round(max(0.0, min(1.0, float(v))) * 32767))


def load_weights(weights_path, meta):
    with open(weights_path) as f:
        lines = [l.strip() for l in f if l.strip()]
    data = bytes(int(l, 16) for l in lines)

    n, h1, h2 = meta["n_features"], meta["h1_size"], meta["h2_size"]

    z_offset = struct.unpack(">i", data[0:4])[0]
    z_scale  = struct.unpack(">I", data[4:8])[0]

    write_reg(REG_Z_OFFSET,   z_offset)
    write_reg(REG_Z_SCALE,    z_scale)
    write_reg(REG_N_FEATURES, n)
    write_reg(REG_H1_SIZE,    h1)
    write_reg(REG_H2_SIZE,    h2)
    write_reg(REG_WEIGHT_WE,  0)

    idx = 8

    def send(layer, neuron, col):
        nonlocal idx
        write_reg(REG_WEIGHT_ADDR, (layer << 14) | (neuron << 8) | col)
        write_reg(REG_WEIGHT_DATA, data[idx])
        write_reg(REG_WEIGHT_WE,   1)
        write_reg(REG_WEIGHT_WE,   0)
        idx += 1

    for j in range(h1):
        for i in range(n + 1):
            send(L1, j, i)
    for j in range(h2):
        for i in range(h1 + 1):
            send(L2, j, i)
    for j in range(3):
        for i in range(h2 + 1):
            send(L3, j, i)

    print(f"loaded {idx - 8} weights")


def push_feature(idx, norm_val):
    write_reg(REG_FEATURE_BASE + idx * 4, float_to_q15(norm_val))


def push_all_features():
    with state_lock:
        meta  = state["metadata"]
        feats = list(state["features"])
        ax_x  = state["axis_x"]
        ax_y  = state["axis_y"]
    if meta is None:
        return
    for i in range(meta["n_features"]):
        if i != ax_x and i != ax_y:
            push_feature(i, feats[i])


def _q44(b):
    v = b if b < 128 else b - 256
    return v / 16.0


def parse_weights(hex_path, meta):
    with open(hex_path) as f:
        data = bytes(int(l.strip(), 16) for l in f if l.strip())

    n, h1, h2 = meta["n_features"], meta["h1_size"], meta["h2_size"]
    idx = 8

    def read_layer(rows, cols):
        nonlocal idx
        W = np.zeros((rows, cols))
        b = np.zeros(rows)
        for j in range(rows):
            for i in range(cols):
                W[j, i] = _q44(data[idx]); idx += 1
            b[j] = _q44(data[idx]); idx += 1
        return W, b

    W1, b1 = read_layer(h1, n)
    W2, b2 = read_layer(h2, h1)
    W3, b3 = read_layer(3, h2)

    return {
        "W1": W1, "b1": b1,
        "W2": W2, "b2": b2,
        "W3": W3, "b3": b3,
        "z_offset": struct.unpack(">i", data[0:4])[0] / 65536.0,
        "z_scale":  struct.unpack(">I", data[4:8])[0] / 65536.0,
    }


def _draw_axes(img, meta, ax_x, ax_y):
    W, H  = img.size
    names = meta["feature_names"]
    fmin  = meta["feature_min"]
    fmax  = meta["feature_max"]

    x_name = names[ax_x] if ax_x < len(names) else f"f{ax_x}"
    y_name = names[ax_y] if ax_y < len(names) else f"f{ax_y}"
    x_lo, x_hi = fmin[ax_x], fmax[ax_x]
    y_lo, y_hi = fmin[ax_y], fmax[ax_y]

    bar = Image.new("RGBA", img.size, (0, 0, 0, 0))
    bd  = ImageDraw.Draw(bar)
    bd.rectangle([(0, H-34), (W, H)],       fill=(0, 0, 0, 170))
    bd.rectangle([(0, 0),    (38, H-34)],   fill=(0, 0, 0, 130))
    img  = Image.alpha_composite(img.convert("RGBA"), bar).convert("RGB")
    draw = ImageDraw.Draw(img)

    WHITE, DIM = (230, 230, 230), (160, 160, 160)
    n_ticks = 5

    for t in range(n_ticks + 1):
        px  = int(t * (W-1) / n_ticks)
        val = x_lo + t * (x_hi - x_lo) / n_ticks
        lbl = f"{val:.0f}" if abs(val) >= 10 else f"{val:.2f}"
        draw.line([(px, H-34), (px, H-26)], fill=DIM, width=1)
        draw.text((px - len(lbl)*3, H-24), lbl, fill=DIM)
    draw.text((W//2 - len(x_name)*3, H-13), x_name, fill=WHITE)

    for t in range(n_ticks + 1):
        py  = int(t * (H-35) / n_ticks)
        val = y_hi - t * (y_hi - y_lo) / n_ticks
        lbl = f"{val:.0f}" if abs(val) >= 10 else f"{val:.2f}"
        draw.line([(25, py), (35, py)], fill=DIM, width=1)
        draw.text((1, py-5), lbl, fill=DIM)

    limg = Image.new("RGBA", (len(y_name)*7+4, 14), (0,0,0,0))
    ImageDraw.Draw(limg).text((2, 1), y_name, fill=WHITE)
    limg = limg.rotate(90, expand=True)
    img.paste(limg, (1, max(0, (H-35)//2 - limg.height//2)), limg)

    return img


def compositor_thread():
    global latest_frame
    while True:
        try:
            with state_lock:
                phase = state["phase"]
                meta  = state["metadata"]
                ax_x  = state["axis_x"]
                ax_y  = state["axis_y"]

            if vdma is None or phase != "ready" or meta is None:
                time.sleep(0.1)
                continue

            raw = np.array(vdma.readchannel.readframe(), dtype=np.uint8)
            img = _draw_axes(Image.fromarray(raw), meta, ax_x, ax_y)

            with frame_lock:
                latest_frame = np.array(img)

            out      = vdma.writechannel.newframe()
            out[:]   = np.array(img)
            vdma.writechannel.writeframe(out)

        except Exception as e:
            print(f"compositor: {e}")
            time.sleep(0.5)

        time.sleep(0.18)


def _train_worker():
    weights_path = os.path.join(OUTPUT_DIR, "weights.hex")
    meta_path    = os.path.join(OUTPUT_DIR, "metadata.json")

    try:
        result = subprocess.run(
            [sys.executable, TRAIN_SCRIPT,
             "--csv",      DATASET_PATH,
             "--weights",  weights_path,
             "--metadata", meta_path],
            capture_output=True, text=True, timeout=600
        )

        if result.returncode != 0:
            err = (result.stderr or result.stdout or "unknown error")[-600:]
            with state_lock:
                state["phase"]     = "error"
                state["error_msg"] = err
            print(f"training failed: {err[:100]}")
            return

        with open(meta_path) as f:
            meta = json.load(f)

        load_weights(weights_path, meta)
        parsed = parse_weights(weights_path, meta)

        defaults = meta.get("feature_mean", [0.5] * meta["n_features"])
        fmin, fmax = meta["feature_min"], meta["feature_max"]
        norm_defaults = []
        for i, v in enumerate(defaults[:meta["n_features"]]):
            span = fmax[i] - fmin[i]
            norm_defaults.append((v - fmin[i]) / span if span > 0 else 0.5)

        with state_lock:
            state["phase"]    = "ready"
            state["metadata"] = meta
            state["weights"]  = parsed
            state["axis_x"]   = 0
            state["axis_y"]   = 1
            state["features"] = norm_defaults + [0.5] * max(0, 32 - len(norm_defaults))

        write_reg(REG_AXIS_X, 0)
        write_reg(REG_AXIS_Y, 1)
        push_all_features()
        print("training done")

    except subprocess.TimeoutExpired:
        with state_lock:
            state["phase"]     = "error"
            state["error_msg"] = "timed out"
    except Exception as e:
        with state_lock:
            state["phase"]     = "error"
            state["error_msg"] = str(e)
        print(f"training error: {e}")


app = Flask(__name__, static_folder="static")
os.makedirs(OUTPUT_DIR, exist_ok=True)


@app.route("/")
def index():
    path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    with open(path) as f:
        return Response(f.read(), mimetype="text/html")


@app.route("/upload", methods=["POST"])
def upload():
    if "csv" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["csv"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "must be .csv"}), 400

    with state_lock:
        if state["phase"] == "training":
            return jsonify({"error": "already training"}), 409

    f.save(DATASET_PATH)

    with state_lock:
        state["phase"]       = "training"
        state["error_msg"]   = ""
        state["train_start"] = time.time()
        state["metadata"]    = None

    threading.Thread(target=_train_worker, daemon=True).start()
    return jsonify({"ok": True})


@app.route("/status")
def status():
    with state_lock:
        phase = state["phase"]
        meta  = state["metadata"]
        err   = state["error_msg"]
        t0    = state["train_start"]
        ax_x  = state["axis_x"]
        ax_y  = state["axis_y"]
        feats = list(state["features"])

    if phase == "training":
        return jsonify({"state": "training", "elapsed": round(time.time() - t0, 1)})
    if phase == "error":
        return jsonify({"state": "error", "msg": err})
    if phase == "ready" and meta:
        return jsonify({
            "state":         "ready",
            "feature_names": meta["feature_names"],
            "feature_min":   meta["feature_min"],
            "feature_max":   meta["feature_max"],
            "n_features":    meta["n_features"],
            "axis_x":        ax_x,
            "axis_y":        ax_y,
            "features":      feats[:meta["n_features"]],
        })
    return jsonify({"state": "idle"})


@app.route("/axis", methods=["POST"])
def set_axis():
    body = request.json or {}
    axis = body.get("axis")
    idx  = int(body.get("index", 0))

    with state_lock:
        meta = state["metadata"]
        if meta and idx >= meta["n_features"]:
            return jsonify({"error": "out of range"}), 400
        if axis == "x" and idx == state["axis_y"]:
            return jsonify({"error": "already Y axis"}), 400
        if axis == "y" and idx == state["axis_x"]:
            return jsonify({"error": "already X axis"}), 400
        if axis == "x":
            state["axis_x"] = idx
            write_reg(REG_AXIS_X, idx)
        elif axis == "y":
            state["axis_y"] = idx
            write_reg(REG_AXIS_Y, idx)
        else:
            return jsonify({"error": "axis must be x or y"}), 400

    push_all_features()
    return jsonify({"ok": True})


@app.route("/feature", methods=["POST"])
def set_feature():
    body  = request.json or {}
    idx   = int(body.get("index", 0))
    value = max(0.0, min(1.0, float(body.get("value", 0.5))))

    with state_lock:
        meta = state["metadata"]
        if meta and idx >= meta["n_features"]:
            return jsonify({"error": "out of range"}), 400
        state["features"][idx] = value

    push_feature(idx, value)
    return jsonify({"ok": True})


@app.route("/frame.jpg")
def get_frame():
    with frame_lock:
        frame = latest_frame

    if frame is None:
        img = Image.new("RGB", (320, 240), (12, 12, 12))
        ImageDraw.Draw(img).text((100, 115), "no signal", fill=(40, 40, 40))
    else:
        img = Image.fromarray(frame).resize((320, 240), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)
    buf.seek(0)
    return Response(buf.read(), mimetype="image/jpeg",
                    headers={"Cache-Control": "no-store"})


@app.route("/predict", methods=["POST"])
def predict():
    body = request.json or {}
    with state_lock:
        meta    = state["metadata"]
        weights = state["weights"]
        feats   = list(state["features"])
        ax_x    = state["axis_x"]
        ax_y    = state["axis_y"]

    if meta is None or weights is None:
        return jsonify({"error": "not ready"}), 400

    n = meta["n_features"]
    x = np.array(feats[:n], dtype=float)
    if "x_norm" in body: x[ax_x] = float(body["x_norm"])
    if "y_norm" in body: x[ax_y] = float(body["y_norm"])

    W1, b1 = weights["W1"], weights["b1"]
    W2, b2 = weights["W2"], weights["b2"]
    W3, b3 = weights["W3"], weights["b3"]

    z1 = W1 @ x + b1
    a1 = np.maximum(0, z1)
    z2 = W2 @ a1 + b2
    a2 = np.maximum(0, z2)
    z3 = W3 @ a2 + b3

    rgb = np.clip(
        (z3 - weights["z_offset"]) * weights["z_scale"], 0, 255
    ).astype(int).tolist()

    # gradient of output w.r.t. input for feature importance
    r1 = (z1 > 0).astype(float)
    r2 = (z2 > 0).astype(float)
    J  = (W3 @ (W2 * r2[:, None])) @ (W1 * r1[:, None])  # (3, n)
    imp = np.mean(np.abs(J), axis=0)
    imp = (imp / imp.max()).tolist() if imp.max() > 0 else [0.0] * n

    def norm_acts(a):
        m = a.max()
        return (a / m).tolist() if m > 0 else a.tolist()

    return jsonify({
        "rgb": rgb,
        "activations": {
            "input": x.tolist(),
            "l1":    norm_acts(a1),
            "l2":    norm_acts(a2),
            "l3":    norm_acts(np.maximum(0, z3)),
        },
        "importance": imp,
    })


@app.route("/weights_data")
def weights_data():
    with state_lock:
        w    = state["weights"]
        meta = state["metadata"]
    if w is None:
        return jsonify({"error": "not ready"}), 400
    return jsonify({
        "W1": w["W1"].tolist(), "b1": w["b1"].tolist(),
        "W2": w["W2"].tolist(), "b2": w["b2"].tolist(),
        "W3": w["W3"].tolist(), "b3": w["b3"].tolist(),
        "z_offset": w["z_offset"],
        "z_scale":  w["z_scale"],
    })


if __name__ == "__main__":
    threading.Thread(target=compositor_thread, daemon=True).start()
    print("http://0.0.0.0:5001")
    app.run(host="0.0.0.0", port=5001, threaded=True)