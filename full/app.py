import json
import os
import threading
import time

import requests
from flask import Flask, jsonify, request, Response

import ml

# config

PYNQ_URL = "http://192.168.2.99:5002"
OUTPUT_DIR = "output"
DATASET_PATH = os.path.join(OUTPUT_DIR, "dataset.csv")

SCREEN_W = 640
SCREEN_H = 480

REG_FEATURE_BASE = 0x000
REG_AXIS_X = 0x100
REG_AXIS_Y = 0x104
REG_OVERLAY_ENABLE = 0x130
REG_CROSSHAIR_X = 0x134
REG_CROSSHAIR_Y = 0x138
REG_AXIS_X_POS = 0x13C
REG_AXIS_Y_POS = 0x140
REG_THICKNESS = 0x144
REG_CROSS_COLOUR = 0x148
REG_AXIS_COLOUR = 0x14C
REG_CROSS_HALF = 0x150

WHITE = 0x00FFFFFF

app = Flask(__name__)

state = {
 "phase": "idle",
 "error_msg": "",
 "train_start": 0.0,
 "train_progress": {"epoch": 0, "n_epochs": 0, "loss": None},
 "metadata": None,
 "axis_x": 0,
 "axis_y": 1,
 "features": [0.5] * 32,
}
state_lock = threading.Lock()


def pynq_post(path, payload, timeout=2.0):
 try:
 r = requests.post(f"{PYNQ_URL}{path}", json=payload, timeout=timeout)
 if not r.ok:
 print(f"pynq POST {path} failed: {r.status_code} {r.text[:200]}")
 return r
 except requests.RequestException as e:
 print(f"pynq unreachable ({path}): {e}")
 return None


def pynq_get(path, timeout=2.0):
 try:
 r = requests.get(f"{PYNQ_URL}{path}", timeout=timeout)
 if not r.ok:
 print(f"pynq GET {path} failed: {r.status_code} {r.text[:200]}")
 return r
 except requests.RequestException as e:
 print(f"pynq unreachable ({path}): {e}")
 return None


def load_weights_to_pynq(weights_path, meta):
 with open(weights_path, "rb") as f:
 data = f.read()

 return pynq_post("/load_weights", {
 "hex": data.hex(),
 "n": meta["n_features"],
 "h1": meta["h1"],
 "h2": meta["h2"],
 "z_shift": meta.get("z_shift", 0),
 }, timeout=30.0)


def send_axes_to_pynq(axis_x, axis_y):
 pynq_post("/batch_update", {
 "regs": [[REG_AXIS_X, axis_x], [REG_AXIS_Y, axis_y]],
 })


def float_to_q15(v):
 return int(round(max(0.0, min(1.0, float(v))) * 32767))


def push_feature(idx, norm_val):
 pynq_post("/batch_update", {
 "features": [[idx, float_to_q15(norm_val)]],
 })


def push_all_features():
 with state_lock:
 meta = state["metadata"]
 feats = list(state["features"])
 ax_x, ax_y = state["axis_x"], state["axis_y"]

 if meta is None:
 return

 updates = []
 for i in range(meta["n_features"]):
 if i != ax_x and i != ax_y:
 updates.append([i, float_to_q15(feats[i])])

 if updates:
 pynq_post("/batch_update", {"features": updates})


def norm_to_pixel(norm, size):
 return max(0, min(size - 1, int(round(norm * (size - 1)))))


def push_overlay_defaults():
 pynq_post("/batch_update", {
 "regs": [
 [REG_OVERLAY_ENABLE, 1],
 [REG_THICKNESS, 1],
 [REG_CROSS_COLOUR, WHITE],
 [REG_AXIS_X_POS, 0],
 [REG_AXIS_Y_POS, 0],
 [REG_AXIS_COLOUR, WHITE],
 [REG_CROSS_HALF, 6],
 ]
 })


def push_crosshair():
 with state_lock:
 meta = state["metadata"]
 feats = list(state["features"])
 ax_x, ax_y = state["axis_x"], state["axis_y"]

 if meta is None:
 return

 px = norm_to_pixel(feats[ax_x], SCREEN_W)
 py = norm_to_pixel(feats[ax_y], SCREEN_H)
 pynq_post("/batch_update", {
 "regs": [[REG_CROSSHAIR_X, px], [REG_CROSSHAIR_Y, py]]
 })

def _train_worker():
 def progress_cb(epoch, n_epochs, loss):
 with state_lock:
 state["train_progress"] = {"epoch": epoch, "n_epochs": n_epochs, "loss": loss}

 try:
 meta = ml.train(DATASET_PATH, OUTPUT_DIR, progress_cb=progress_cb)

 weights_path = os.path.join(OUTPUT_DIR, "weights.hex")
 r = load_weights_to_pynq(weights_path, meta)
 if r is None or not r.ok:
 detail = r.text[:300] if r is not None else "PYNQ unreachable"
 with state_lock:
 state["phase"] = "error"
 state["error_msg"] = f"trained ok but /load_weights failed: {detail}"
 return

 with state_lock:
 state["phase"] = "ready"
 state["metadata"] = meta
 state["axis_x"] = 0
 state["axis_y"] = 1
 state["features"] = [0.5] * meta["n_features"]

 send_axes_to_pynq(0, 1)
 push_overlay_defaults()
 push_all_features()
 push_crosshair()
 print("training complete, weights loaded onto PYNQ")

 except Exception as e:
 with state_lock:
 state["phase"] = "error"
 state["error_msg"] = str(e)
 print("training failed:", e)


# routes


@app.route("/")
def index():
 base = os.path.dirname(__file__)
 candidates = [os.path.join(base, "index.html")]
 path = next((x for x in candidates if os.path.exists(x)), None)
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

 os.makedirs(OUTPUT_DIR, exist_ok=True)
 f.save(DATASET_PATH)

 with state_lock:
 state["phase"] = "training"
 state["error_msg"] = ""
 state["train_start"] = time.time()
 state["train_progress"] = {"epoch": 0, "n_epochs": 0, "loss": None}
 state["metadata"] = None

 threading.Thread(target=_train_worker, daemon=True).start()
 return jsonify({"ok": True})


@app.route("/load_existing", methods=["POST"])
def load_existing():
 weights_path = os.path.join(OUTPUT_DIR, "weights.hex")
 meta_path = os.path.join(OUTPUT_DIR, "metadata.json")

 if not os.path.exists(weights_path) or not os.path.exists(meta_path):
 return jsonify({"error": f"missing weights.hex/metadata.json in {OUTPUT_DIR}"}), 400

 try:
 with open(meta_path) as f:
 meta = json.load(f)

 r = load_weights_to_pynq(weights_path, meta)
 if r is None or not r.ok:
 detail = r.text[:300] if r is not None else "PYNQ unreachable"
 return jsonify({"error": f"/load_weights failed: {detail}"}), 502

 with state_lock:
 state["phase"] = "ready"
 state["metadata"] = meta
 state["axis_x"] = 0
 state["axis_y"] = 1
 state["features"] = [0.5] * meta["n_features"]

 send_axes_to_pynq(0, 1)
 push_overlay_defaults()
 push_all_features()
 push_crosshair()
 return jsonify({"ok": True})

 except Exception as e:
 return jsonify({"error": str(e)}), 500


@app.route("/white_screen", methods=["POST"])
def white_screen():
 r = pynq_post("/white_screen", {})
 if r is None:
 return jsonify({"error": "PYNQ unreachable"}), 502
 return jsonify(r.json()), r.status_code


@app.route("/pynq_status")
def pynq_status():
 r = pynq_get("/status")
 if r is None:
 return jsonify({"error": "PYNQ unreachable"}), 502
 return jsonify(r.json()), r.status_code


@app.route("/status")
def status():
 with state_lock:
 phase = state["phase"]
 meta = state["metadata"]
 err = state["error_msg"]
 t0 = state["train_start"]
 progress = dict(state["train_progress"])
 ax_x, ax_y = state["axis_x"], state["axis_y"]
 feats = list(state["features"])

 if phase == "training":
 return jsonify({
 "state": "training",
 "elapsed": round(time.time() - t0, 1),
 "progress": progress,
 })

 if phase == "error":
 return jsonify({"state": "error", "msg": err})

 if phase == "ready" and meta:
 return jsonify({
 "state": "ready",
 "feature_names": meta["feature_names"],
 "feature_min": meta["feature_min"],
 "feature_max": meta["feature_max"],
 "n_features": meta["n_features"],
 "class_names": meta.get("class_names", []),
 "class_colours": meta.get("class_colours", {}),
 "axis_x": ax_x,
 "axis_y": ax_y,
 "crosshair_x": norm_to_pixel(feats[ax_x], SCREEN_W),
 "crosshair_y": norm_to_pixel(feats[ax_y], SCREEN_H),
 "features": feats[:meta["n_features"]],
 })

 return jsonify({"state": "idle"})


@app.route("/axis", methods=["POST"])
def set_axis():
 body = request.json or {}
 axis = body.get("axis")
 idx = int(body.get("index", 0))

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
 elif axis == "y":
 state["axis_y"] = idx
 else:
 return jsonify({"error": "axis must be x or y"}), 400

 ax_x, ax_y = state["axis_x"], state["axis_y"]

 send_axes_to_pynq(ax_x, ax_y)
 push_all_features()
 push_crosshair()
 return jsonify({"ok": True})


@app.route("/feature", methods=["POST"])
def set_feature():
 body = request.json or {}
 idx = int(body.get("index", 0))
 value = max(0.0, min(1.0, float(body.get("value", 0.5))))

 with state_lock:
 meta = state["metadata"]
 if meta and idx >= meta["n_features"]:
 return jsonify({"error": "out of range"}), 400
 state["features"][idx] = value
 ax_x, ax_y = state["axis_x"], state["axis_y"]

 push_feature(idx, value)

 if idx == ax_x or idx == ax_y:
 push_crosshair()

 return jsonify({"ok": True})



def _i8(byte_val):
 v = int(byte_val) & 0xFF
 return v - 256 if v >= 128 else v


def _read_exported_weights_for_js():
 weights_path = os.path.join(OUTPUT_DIR, "weights.hex")
 meta_path = os.path.join(OUTPUT_DIR, "metadata.json")
 if not os.path.exists(weights_path) or not os.path.exists(meta_path):
 raise FileNotFoundError(f"missing weights.hex/metadata.json in {OUTPUT_DIR}")

 with open(meta_path) as f:
 meta = json.load(f)
 with open(weights_path, "rb") as f:
 data = f.read()

 if len(data) < 8:
 raise ValueError("weights.hex is shorter than its 8-byte header")

 z_offset = int.from_bytes(data[0:4], "big", signed=True)
 z_scale = int.from_bytes(data[4:8], "big", signed=False)
 raw = data[8:]

 n = int(meta["n_features"])
 h1 = int(meta["h1"])
 h2 = int(meta["h2"])
 expected = h1*n + h1 + h2*h1 + h2 + 3*h2 + 3
 if len(raw) != expected:
 raise ValueError(f"bad weights.hex payload length: got {len(raw)}, expected {expected}")

 vals = [_i8(b) for b in raw]
 k = 0

 def take(count):
 nonlocal k
 out = vals[k:k+count]
 k += count
 return out

 W1_flat = take(h1*n)
 b1 = take(h1)
 W2_flat = take(h2*h1)
 b2 = take(h2)
 W3_flat = take(3*h2)
 b3 = take(3)

 W1 = [W1_flat[i*n:(i+1)*n] for i in range(h1)]
 W2 = [[v / 16.0 for v in W2_flat[i*h1:(i+1)*h1]] for i in range(h2)]
 b2_for_js = [v * 16.0 for v in b2]
 W3 = [[v / 16.0 for v in W3_flat[i*h2:(i+1)*h2]] for i in range(3)]
 b3_for_js = [v * 256.0 for v in b3]

 return {
 "W1": W1,
 "b1": b1,
 "W2": W2,
 "b2": b2_for_js,
 "W3": W3,
 "b3": b3_for_js,
 "z_offset": z_offset,
 "z_scale": z_scale,
 "n_features": n,
 "h1": h1,
 "h2": h2,
 }


@app.route("/weights_data")
def weights_data():
 try:
 return jsonify(_read_exported_weights_for_js())
 except Exception as e:
 return jsonify({"error": str(e)}), 500


@app.route("/probe", methods=["POST"])
def set_probe():
 body = request.json or {}
 x_norm = max(0.0, min(1.0, float(body.get("x", body.get("nx", 0.5)))))
 y_norm = max(0.0, min(1.0, float(body.get("y", body.get("ny", 0.5)))))

 with state_lock:
 meta = state["metadata"]
 ax_x, ax_y = state["axis_x"], state["axis_y"]
 if meta is not None:
 state["features"][ax_x] = x_norm
 state["features"][ax_y] = y_norm

 px = norm_to_pixel(x_norm, SCREEN_W)
 py = norm_to_pixel(y_norm, SCREEN_H)
 r = pynq_post("/batch_update", {"regs": [[REG_CROSSHAIR_X, px], [REG_CROSSHAIR_Y, py]]})
 if r is None:
 return jsonify({"error": "PYNQ unreachable", "x": px, "y": py}), 502
 return jsonify({"ok": bool(r.ok), "x": px, "y": py, "nx": x_norm, "ny": y_norm,
 "pynq": r.json() if r.ok else r.text}), r.status_code


@app.route("/crosshair", methods=["POST"])
def set_crosshair():
 body = request.json or {}

 if "x" in body or "y" in body:
 px = int(body.get("x", SCREEN_W // 2))
 py = int(body.get("y", SCREEN_H // 2))
 else:
 px = norm_to_pixel(float(body.get("nx", 0.5)), SCREEN_W)
 py = norm_to_pixel(float(body.get("ny", 0.5)), SCREEN_H)

 px = max(0, min(SCREEN_W - 1, px))
 py = max(0, min(SCREEN_H - 1, py))
 r = pynq_post("/batch_update", {"regs": [[REG_CROSSHAIR_X, px], [REG_CROSSHAIR_Y, py]]})
 if r is None:
 return jsonify({"error": "PYNQ unreachable"}), 502
 return jsonify({"ok": bool(r.ok), "x": px, "y": py, "pynq": r.json() if r.ok else r.text}), r.status_code


@app.route("/overlay", methods=["POST"])
def set_overlay():
 body = request.json or {}
 regs = []

 if "enable" in body:
 regs.append([REG_OVERLAY_ENABLE, 1 if bool(body["enable"]) else 0])
 if "crosshair" in body:
 if bool(body["crosshair"]):
 regs.extend([[REG_CROSS_COLOUR, WHITE], [REG_CROSS_HALF, 6]])
 else:
 regs.extend([[REG_CROSS_COLOUR, 0], [REG_CROSS_HALF, 0]])
 if "axis_lines" in body:
 if bool(body["axis_lines"]):
 regs.extend([[REG_AXIS_COLOUR, WHITE], [REG_AXIS_X_POS, 0], [REG_AXIS_Y_POS, 0]])
 else:
 regs.append([REG_AXIS_COLOUR, 0])

 if not regs:
 return jsonify({"error": "nothing to update"}), 400

 r = pynq_post("/batch_update", {"regs": regs})
 if r is None:
 return jsonify({"error": "PYNQ unreachable"}), 502
 return jsonify(r.json() if r.ok else {"error": r.text}), r.status_code


@app.route("/reg", methods=["POST"])
def reg_passthrough():
 r = pynq_post("/reg", request.json or {})
 if r is None:
 return jsonify({"error": "PYNQ unreachable"}), 502
 return jsonify(r.json() if r.ok else {"error": r.text}), r.status_code


@app.route("/batch_update", methods=["POST"])
def batch_update_passthrough():
 r = pynq_post("/batch_update", request.json or {})
 if r is None:
 return jsonify({"error": "PYNQ unreachable"}), 502
 return jsonify(r.json() if r.ok else {"error": r.text}), r.status_code


if __name__ == "__main__":
 os.makedirs(OUTPUT_DIR, exist_ok=True)
 print("http://0.0.0.0:5001")
 app.run(host="0.0.0.0", port=5001, threaded=True)