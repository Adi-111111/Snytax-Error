# Run on PYNQ:
#   cd /home/xilinx/jupyter_notebooks/UI7
#   source /etc/profile.d/pynq_venv.sh
#   sudo -E /usr/local/share/pynq-venv/bin/python3 pynq_server2.py

import json
import os
import signal
import struct
import sys
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse

from pynq import Overlay
from pynq.lib.video import VideoMode

#config

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OVERLAY_PATH = os.path.join(SCRIPT_DIR, "base.bit")
HOST = "0.0.0.0"
PORT = 5002

W = 640
H = 480
VIDEO_MODE = VideoMode(W, H, 24)


REG_FEATURE_BASE   = 0x000
REG_AXIS_X         = 0x100
REG_AXIS_Y         = 0x104
REG_N_FEATURES     = 0x108
REG_H1_SIZE        = 0x10C
REG_H2_SIZE        = 0x110
REG_WEIGHT_ADDR    = 0x114
REG_WEIGHT_DATA    = 0x118
REG_WEIGHT_WE      = 0x11C
REG_Z_OFFSET       = 0x120
REG_Z_SCALE        = 0x124
REG_Z_SHIFT        = 0x128
REG_DISPLAY_ENABLE = 0x12C
REG_OVERLAY_ENABLE = 0x130
REG_CROSSHAIR_X    = 0x134
REG_CROSSHAIR_Y    = 0x138
REG_AXIS_X_POS     = 0x13C
REG_AXIS_Y_POS     = 0x140
REG_THICKNESS      = 0x144
REG_CROSS_COLOUR   = 0x148
REG_AXIS_COLOUR    = 0x14C
REG_CROSS_HALF     = 0x150
REG_TEST_PATTERN_ENABLE = 0x18C

DEBUG_TVALID     = 0x154
DEBUG_TREADY     = 0x158
DEBUG_HANDSHAKE  = 0x15C
DEBUG_TLAST      = 0x160
DEBUG_TUSER      = 0x164
DEBUG_FRAME      = 0x168
DEBUG_FIFO_WR    = 0x16C
DEBUG_FIFO_RD    = 0x170
DEBUG_MLP_VALID  = 0x174
DEBUG_NONZERO_TDATA    = 0x178
DEBUG_WHITE_TDATA      = 0x17C
DEBUG_LAST_TDATA       = 0x180
DEBUG_CROSSHAIR_PIXELS = 0x184
DEBUG_AXIS_PIXELS      = 0x188

L1, L2, L3 = 0, 1, 2
WHITE = 0x00FFFFFF

DEFAULT_N_FEATURES = 16
DEFAULT_H1_SIZE = 32
DEFAULT_H2_SIZE = 16

#debug
MM2S_DMACR  = 0x00
MM2S_DMASR  = 0x04
PARK_PTR    = 0x28
S2MM_DMACR  = 0x30
S2MM_DMASR  = 0x34
MM2S_VSIZE  = 0x50
MM2S_HSIZE  = 0x54
MM2S_STRIDE = 0x58
MM2S_SA1    = 0x5C
MM2S_SA2    = 0x60
MM2S_SA3    = 0x64
MM2S_SA4    = 0x68
S2MM_VSIZE  = 0xA0
S2MM_HSIZE  = 0xA4
S2MM_STRIDE = 0xA8
S2MM_SA1    = 0xAC
S2MM_SA2    = 0xB0
S2MM_SA3    = 0xB4
S2MM_SA4    = 0xB8

ol = None
pg = None
vdma = None
hdmi_out = None
video_started = False
video_error = ""

#low-level helpers

def hex32(v):
    return "0x%08X" % (int(v) & 0xFFFFFFFF)


def write_reg(offset, value):
    v = int(value) & 0xFFFFFFFF
    if v >= 0x80000000:
        v -= 0x100000000
    pg.write(int(offset), v)


def read_reg(offset):
    return int(pg.read(int(offset))) & 0xFFFFFFFF


def vdma_read(offset):
    return int(vdma.mmio.read(int(offset))) & 0xFFFFFFFF


def decode_dmasr(v):
    names = {
        0: "HALTED", 1: "IDLE", 4: "DMAIntErr", 5: "DMASlvErr", 6: "DMADecErr",
        8: "SGIntErr", 9: "SGSlvErr", 10: "SGDecErr",
        12: "IOC/FrmCnt IRQ", 13: "Delay IRQ", 14: "Error IRQ",
    }
    out = [name for bit, name in names.items() if v & (1 << bit)]
    frame_cnt_status = (v >> 16) & 0xFF
    if frame_cnt_status:
        out.append(f"IRQFrameCount/Sts={frame_cnt_status}")
    return out


def has_vdma_fatal_error(status):
    fatal = {"DMAIntErr", "DMASlvErr", "DMADecErr", "SGIntErr", "SGSlvErr", "SGDecErr", "Error IRQ"}
    for side in ("mm2s", "s2mm"):
        if any(x in fatal for x in status[side].get("DMASR_decode", [])):
            return True
    return False


def json_response(obj, status=200):
    body = json.dumps(obj, indent=2).encode("utf-8")
    return status, "application/json", body


def text_response(txt, status=200):
    return status, "text/plain", txt.encode("utf-8")


def read_json_body(handler):
    length = int(handler.headers.get("Content-Length", "0"))
    if length <= 0:
        return {}
    raw = handler.rfile.read(length)
    if not raw:
        return {}
    return json.loads(raw.decode("utf-8"))

# hardware init

def load_overlay():
    global ol, pg, vdma, hdmi_out

    print("Loading overlay:", OVERLAY_PATH, flush=True)
    ol = Overlay(OVERLAY_PATH)
    pg = ol.pixel_generator_0
    vdma = ol.video.axi_vdma
    hdmi_out = ol.video.hdmi_out

    print("Overlay loaded.", flush=True)
    print("IP keys:", flush=True)
    for k in sorted(ol.ip_dict.keys()):
        print(" ", k, flush=True)


def set_white_screen():
    
    write_reg(REG_DISPLAY_ENABLE, 0)
    write_reg(REG_TEST_PATTERN_ENABLE, 0)

    write_reg(REG_N_FEATURES, DEFAULT_N_FEATURES)
    write_reg(REG_H1_SIZE, DEFAULT_H1_SIZE)
    write_reg(REG_H2_SIZE, DEFAULT_H2_SIZE)
    write_reg(REG_AXIS_X, 0)
    write_reg(REG_AXIS_Y, 1)

    for i in range(DEFAULT_N_FEATURES):
        write_reg(REG_FEATURE_BASE + 4 * i, 0)

    write_reg(REG_Z_OFFSET, -255)
    write_reg(REG_Z_SCALE, 65536)
    write_reg(REG_Z_SHIFT, 0)

    write_reg(REG_OVERLAY_ENABLE, 0)
    write_reg(REG_WEIGHT_WE, 0)

    write_reg(REG_DISPLAY_ENABLE, 1)
    print("White screen registers set.", flush=True)


def tie_channels():
    s2mm_sa_before = [vdma_read(off) for off in (S2MM_SA1, S2MM_SA2, S2MM_SA3, S2MM_SA4)]
    vdma.readchannel.tie(vdma.writechannel)
    mm2s_sa_after = [vdma_read(off) for off in (MM2S_SA1, MM2S_SA2, MM2S_SA3, MM2S_SA4)]
    print("tie_channels: S2MM SA1-4 =", [hex32(v) for v in s2mm_sa_before], flush=True)
    print("tie_channels: MM2S SA1-4 after tie =", [hex32(v) for v in mm2s_sa_after], flush=True)
    if mm2s_sa_after != s2mm_sa_before:
        print("tie_channels WARNING: MM2S SA1-4 do not match S2MM SA1-4 after tie.", flush=True)


def start_video_high_level():
    global video_started, video_error
    video_started = False
    video_error = ""

    try:
        print("Configuring HDMI/VDMA", flush=True)

        hdmi_out._vdma = vdma
        hdmi_out.configure(VIDEO_MODE)

        try:
            vdma.readchannel.mode = VIDEO_MODE
            print("S2MM/readchannel mode set.", flush=True)
        except Exception as e:
            print("readchannel.mode warning:", repr(e), flush=True)

        try:
            vdma.writechannel.mode = VIDEO_MODE
            print("MM2S/writechannel mode set.", flush=True)
        except Exception as e:
            print("writechannel.mode warning:", repr(e), flush=True)

        try:
            vdma.readchannel.start()
            print("S2MM/readchannel started.", flush=True)
        except Exception as e:
            print("S2MM/readchannel start warning:", repr(e), flush=True)

        try:
            vdma.writechannel.start()
            print("MM2S/writechannel started.", flush=True)
        except Exception as e:
            print("writechannel start warning:", repr(e), flush=True)

        try:
            hdmi_out.start()
            print("HDMI output started.", flush=True)
        except Exception as e:
            print("hdmi_out.start warning:", repr(e), flush=True)

        try:
            tie_channels()
        except Exception as e:
            print("tie_channels warning:", repr(e), flush=True)

        video_started = True
        print("Video startup complete.", flush=True)

    except Exception:
        video_error = traceback.format_exc()
        print(video_error, flush=True)
        raise

#clean shutdwon
def shutdown_video():
    for name, fn in [
        ("hdmi_out.stop", lambda: hdmi_out.stop()),
        ("readchannel.stop", lambda: vdma.readchannel.stop()),
        ("writechannel.stop", lambda: vdma.writechannel.stop()),
    ]:
        try:
            fn()
            print(name, "ok", flush=True)
        except Exception as e:
            print(name, "ignored:", repr(e), flush=True)


def handle_shutdown_signal(signum, frame):
    print(f"\nReceived signal {signum}, shutting down cleanly", flush=True)
    try:
        shutdown_video()
    except Exception:
        print(traceback.format_exc(), flush=True)
    print("Clean shutdown done. Exiting.", flush=True)
    sys.exit(0)


def restart_video():
    global video_started, video_error
    video_started = False
    video_error = ""
    try:
        shutdown_video()
        time.sleep(0.2)
        start_video_high_level()
        return True, ""
    except Exception:
        video_error = traceback.format_exc()
        print(video_error, flush=True)
        return False, video_error

# debugg

def debug_counters():
    return {
        "TVALID": read_reg(DEBUG_TVALID),
        "TREADY": read_reg(DEBUG_TREADY),
        "HANDSHAKE": read_reg(DEBUG_HANDSHAKE),
        "TLAST": read_reg(DEBUG_TLAST),
        "TUSER_SOF": read_reg(DEBUG_TUSER),
        "FRAME": read_reg(DEBUG_FRAME),
        "FIFO_WR": read_reg(DEBUG_FIFO_WR),
        "FIFO_RD": read_reg(DEBUG_FIFO_RD),
        "MLP_VALID": read_reg(DEBUG_MLP_VALID),
        "NONZERO_TDATA": read_reg(DEBUG_NONZERO_TDATA),
        "WHITE_TDATA": read_reg(DEBUG_WHITE_TDATA),
        "LAST_TDATA": read_reg(DEBUG_LAST_TDATA),
        "CROSSHAIR_PIXELS": read_reg(DEBUG_CROSSHAIR_PIXELS),
        "AXIS_PIXELS": read_reg(DEBUG_AXIS_PIXELS),
        "TEST_PATTERN_ENABLE": read_reg(REG_TEST_PATTERN_ENABLE),
    }


def debug_delta(seconds=1.0):
    a = debug_counters()
    time.sleep(float(seconds))
    b = debug_counters()
    state_like = {"LAST_TDATA", "TEST_PATTERN_ENABLE"}
    return {k: (int(b[k]) - int(a[k]) if k not in state_like else int(b[k])) for k in a}


def vdma_status():
    out = {
        "video_started": video_started,
        "video_error": video_error,
        "mm2s": {
            "DMACR": hex32(vdma_read(MM2S_DMACR)),
            "DMASR": hex32(vdma_read(MM2S_DMASR)),
            "DMASR_decode": decode_dmasr(vdma_read(MM2S_DMASR)),
            "VSIZE": vdma_read(MM2S_VSIZE), "HSIZE": vdma_read(MM2S_HSIZE),
            "STRIDE": hex32(vdma_read(MM2S_STRIDE)),
            "SA1": hex32(vdma_read(MM2S_SA1)), "SA2": hex32(vdma_read(MM2S_SA2)),
            "SA3": hex32(vdma_read(MM2S_SA3)), "SA4": hex32(vdma_read(MM2S_SA4)),
        },
        "s2mm": {
            "DMACR": hex32(vdma_read(S2MM_DMACR)),
            "DMASR": hex32(vdma_read(S2MM_DMASR)),
            "DMASR_decode": decode_dmasr(vdma_read(S2MM_DMASR)),
            "VSIZE": vdma_read(S2MM_VSIZE), "HSIZE": vdma_read(S2MM_HSIZE),
            "STRIDE": hex32(vdma_read(S2MM_STRIDE)),
            "SA1": hex32(vdma_read(S2MM_SA1)), "SA2": hex32(vdma_read(S2MM_SA2)),
            "SA3": hex32(vdma_read(S2MM_SA3)), "SA4": hex32(vdma_read(S2MM_SA4)),
        },
        "PARK_PTR": hex32(vdma_read(PARK_PTR)),
    }
    try:
        out["pynq_channels"] = {
            "writechannel_running": bool(vdma.writechannel.running),
            "readchannel_running": bool(vdma.readchannel.running),
        }
    except Exception as e:
        out["pynq_channels_error"] = repr(e)
    out["fatal_error"] = has_vdma_fatal_error(out)
    return out


def pg_control_regs():
    regs = {
        "AXIS_X": REG_AXIS_X, "AXIS_Y": REG_AXIS_Y,
        "N_FEATURES": REG_N_FEATURES, "H1_SIZE": REG_H1_SIZE, "H2_SIZE": REG_H2_SIZE,
        "Z_OFFSET": REG_Z_OFFSET, "Z_SCALE": REG_Z_SCALE, "Z_SHIFT": REG_Z_SHIFT,
        "DISPLAY_ENABLE": REG_DISPLAY_ENABLE, "OVERLAY_ENABLE": REG_OVERLAY_ENABLE,
        "CROSSHAIR_X": REG_CROSSHAIR_X, "CROSSHAIR_Y": REG_CROSSHAIR_Y,
        "AXIS_X_POS": REG_AXIS_X_POS, "AXIS_Y_POS": REG_AXIS_Y_POS,
        "THICKNESS": REG_THICKNESS, "CROSS_COLOUR": REG_CROSS_COLOUR,
        "AXIS_COLOUR": REG_AXIS_COLOUR, "CROSS_HALF": REG_CROSS_HALF,
        "TEST_PATTERN_ENABLE": REG_TEST_PATTERN_ENABLE,
        "NONZERO_TDATA": DEBUG_NONZERO_TDATA, "WHITE_TDATA": DEBUG_WHITE_TDATA,
        "LAST_TDATA": DEBUG_LAST_TDATA, "CROSSHAIR_PIXELS": DEBUG_CROSSHAIR_PIXELS,
        "AXIS_PIXELS": DEBUG_AXIS_PIXELS,
    }
    return {name: hex32(read_reg(off)) for name, off in regs.items()}

#weight loading

def send_weight_byte(layer, neuron, col, byte_val):
    addr = (int(layer) << 14) | (int(neuron) << 8) | int(col)
    write_reg(REG_WEIGHT_ADDR, addr)
    write_reg(REG_WEIGHT_DATA, int(byte_val) & 0xFF)
    write_reg(REG_WEIGHT_WE, 1)
    write_reg(REG_WEIGHT_WE, 0)


def load_weights_blob(data, n, h1, h2, z_shift):
    if len(data) < 8:
        raise ValueError("weight data shorter than 8-byte header")

    z_offset = struct.unpack(">i", data[0:4])[0]
    z_scale = struct.unpack(">I", data[4:8])[0]
    raw = data[8:]
    expected = h1 * n + h1 + h2 * h1 + h2 + 3 * h2 + 3
    if len(raw) != expected:
        raise ValueError(f"bad weight payload length: got {len(raw)}, expected {expected}")

    print(
        f"Loading weights: n={n}, h1={h1}, h2={h2}, "
        f"z_shift={z_shift}, z_offset={z_offset}, z_scale={z_scale}",
        flush=True,
    )

    write_reg(REG_DISPLAY_ENABLE, 0)
    write_reg(REG_WEIGHT_WE, 0)
    write_reg(REG_N_FEATURES, n)
    write_reg(REG_H1_SIZE, h1)
    write_reg(REG_H2_SIZE, h2)
    write_reg(REG_Z_OFFSET, z_offset)
    write_reg(REG_Z_SCALE, z_scale)
    write_reg(REG_Z_SHIFT, z_shift)

    idx = 0
    for j in range(h1):
        for i in range(n):
            send_weight_byte(L1, j, i, raw[idx]); idx += 1
    for j in range(h1):
        send_weight_byte(L1, j, n, raw[idx]); idx += 1
    for j in range(h2):
        for i in range(h1):
            send_weight_byte(L2, j, i, raw[idx]); idx += 1
    for j in range(h2):
        send_weight_byte(L2, j, h1, raw[idx]); idx += 1
    for j in range(3):
        for i in range(h2):
            send_weight_byte(L3, j, i, raw[idx]); idx += 1
    for j in range(3):
        send_weight_byte(L3, j, h2, raw[idx]); idx += 1

    write_reg(REG_OVERLAY_ENABLE, 1)
    write_reg(REG_DISPLAY_ENABLE, 1)
    print("Loaded weight bytes:", idx, flush=True)
    return {
        "loaded_bytes": idx,
        "z_offset": z_offset,
        "z_scale": z_scale,
        "z_shift": z_shift,
        "readback": pg_control_regs(),
    }

# HTTP routing

def handle_get(path):
    if path == "/":
        return text_response("pynq_server running\n")
    if path == "/ping":
        return json_response({"ok": True, "server": "pynq_server"})
    if path == "/status":
        return json_response({
            "ok": True,
            "server": "pynq_server",
            "overlay_path": OVERLAY_PATH,
            "video": vdma_status(),
            "pg": pg_control_regs(),
            "debug": debug_counters(),
        })
    if path == "/debug":
        return json_response({
            "ok": True,
            "debug": debug_counters(),
            "delta_1s": debug_delta(1.0),
            "vdma": vdma_status(),
            "pg": pg_control_regs(),
        })
    if path == "/vdma":
        return json_response({"ok": True, "vdma": vdma_status()})
    if path == "/pg":
        return json_response({"ok": True, "pg": pg_control_regs()})
    if path.startswith("/reg/"):
        off = int(path.rsplit("/", 1)[1], 0)
        val = read_reg(off)
        return json_response({"ok": True, "offset": hex(off), "value": val, "value_hex": hex32(val)})
    return json_response({"ok": False, "error": f"unknown GET path {path}"}, 404)


def handle_post(path, handler):
    body = read_json_body(handler)

    if path == "/reg":
        off = int(body["offset"])
        val = int(body["value"])
        write_reg(off, val)
        return json_response({"ok": True, "offset": hex(off), "value": val & 0xFFFFFFFF, "readback": hex32(read_reg(off))})

    if path == "/batch_update":
        regs_written = 0
        features_written = 0

        for item in body.get("regs", []):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            off, val = int(item[0]), int(item[1])
            write_reg(off, val)
            regs_written += 1

        for item in body.get("features", []):
            if not isinstance(item, (list, tuple)) or len(item) != 2:
                continue
            idx, val = int(item[0]), int(item[1])
            if 0 <= idx < 128:
                write_reg(REG_FEATURE_BASE + idx * 4, val)
                features_written += 1

        ch = body.get("crosshair")
        if isinstance(ch, dict):
            if "x" in ch:
                cx = max(0, min(W - 1, int(ch["x"])))
                write_reg(REG_CROSSHAIR_X, cx)
                write_reg(REG_AXIS_X_POS, cx)
                regs_written += 2
            if "y" in ch:
                cy = max(0, min(H - 1, int(ch["y"])))
                write_reg(REG_CROSSHAIR_Y, cy)
                write_reg(REG_AXIS_Y_POS, cy)
                regs_written += 2

        return json_response({
            "ok": True,
            "regs_written": regs_written,
            "features_written": features_written,
            "crosshair": {"x": read_reg(REG_CROSSHAIR_X), "y": read_reg(REG_CROSSHAIR_Y)},
        })

    if path == "/load_weights":
        data = bytes.fromhex(body.get("hex", ""))
        result = load_weights_blob(data, int(body["n"]), int(body["h1"]), int(body["h2"]), int(body.get("z_shift", 0)))
        return json_response({"ok": True, **result})

    if path == "/white_screen":
        set_white_screen()
        return json_response({"ok": True, "pg": pg_control_regs(), "debug": debug_counters(), "vdma": vdma_status()})

    if path == "/test_pattern":
        enable = bool(body.get("enable", True))
        write_reg(REG_TEST_PATTERN_ENABLE, 1 if enable else 0)
        return json_response({"ok": True, "enabled": enable, "debug": debug_counters(), "vdma": vdma_status()})

    if path == "/video_restart":
        ok, err = restart_video()
        return json_response({"ok": ok, "error": err, "vdma": vdma_status(), "debug": debug_counters()}, 200 if ok else 500)

    if path == "/retie":
        tie_channels()
        return json_response({"ok": True, "vdma": vdma_status()})

    return json_response({"ok": False, "error": f"unknown POST path {path}"}, 404)

# HTTP server

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def send_payload(self, status, content_type, body):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        try:
            status, ctype, body = handle_get(path)
        except Exception as e:
            print(traceback.format_exc(), flush=True)
            status, ctype, body = json_response({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, 500)
        self.send_payload(status, ctype, body)

    def do_POST(self):
        path = urlparse(self.path).path
        try:
            status, ctype, body = handle_post(path, self)
        except Exception as e:
            print(traceback.format_exc(), flush=True)
            status, ctype, body = json_response({"ok": False, "error": str(e), "traceback": traceback.format_exc()}, 500)
        self.send_payload(status, ctype, body)

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.client_address[0], fmt % args), flush=True)

# main

def main():
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    signal.signal(signal.SIGTERM, handle_shutdown_signal)

    load_overlay()
    write_reg(REG_DISPLAY_ENABLE, 0)
    set_white_screen()
    start_video_high_level()
    write_reg(REG_DISPLAY_ENABLE, 1)

    print("Initial status:", json.dumps({
        "vdma": vdma_status(),
        "debug": debug_counters(),
        "pg": pg_control_regs(),
    }, indent=2), flush=True)
    print(f"Starting HTTP server on http://{HOST}:{PORT}", flush=True)
    HTTPServer((HOST, PORT), Handler).serve_forever()


if __name__ == "__main__":
    main()
