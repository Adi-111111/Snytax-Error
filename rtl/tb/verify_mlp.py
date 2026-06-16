"""
This verifies pixel_generator MLP output values.
Implements the exact integer arithmetic of the HDL.
  - Note: z_shift must be set to 32 in HDL for matching output representation
"""
import re, sys
import numpy as np

# Weight parsing
def parse_weights(path):
    """Return dict {(layer, neuron, col): signed_int8} from write_weight() calls."""
    w = {}
    pat = re.compile(
        r"write_weight\(2'd(\d+),\s*6'd(\d+),\s*8'd(\d+),\s*8'h([0-9A-Fa-f]+)\)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if m:
                layer  = int(m.group(1))
                neuron = int(m.group(2))
                col    = int(m.group(3))
                val    = int(m.group(4), 16)
                if val >= 128:
                    val -= 256 # to signed int8
                w[(layer, neuron, col)] = val
    return w

# Matching int to hardware truncation / sign extension
def to_int32(v):
    #Truncate a Python int to 32-bit signed (matches l1_buffer / l2_buffer [31:0])
    v = int(v) & 0xFFFFFFFF
    return v - 0x100000000 if v >= 0x80000000 else v

# Forward pass computation
def mlp_pixel(x_coord, y_coord, W):
    # Input builder
    INV_639 = 51          # 1/639 * 32768 ~ 51  (Q1.15 equivalent reciprocal)
    INV_479 = 68          # 1/479 * 32768 ~ 68
    x_norm = np.int16((x_coord * INV_639) & 0xFFFF)
    y_norm = np.int16((y_coord * INV_479) & 0xFFFF)
    # axis_x_select=0, axis_y_select=1, features[2..15]=16384 (regfile initialisation)
    x_in = [int(x_norm), int(y_norm)] + [16384] * 14

    BIAS = 32767   # BIAS_CONST_L1 / L23 ~ 1.0 in Q0.15

    # Layer 1: 64 neurons, 16 inputs + bias at col 16
    l1_acc = np.zeros(64, dtype=np.int64)
    for n in range(64):
        s = np.int64(0)
        for i in range(16):
            s += np.int64(x_in[i]) * np.int64(W.get((0, n, i), 0))
        s += np.int64(BIAS) * np.int64(W.get((0, n, 16), 0))
        l1_acc[n] = s
    l1_out = np.maximum(0, l1_acc) # ReLU (48-bit signed)
    # hardware stores l1_out[31:0] in l1_buffer
    l1_buf = np.array([to_int32(v) for v in l1_out], dtype=np.int64)

    # Layer 2: 32 neurons, 64 inputs + bias at col 64
    l2_acc = np.zeros(32, dtype=np.int64)
    for n in range(32):
        s = np.int64(0)
        for i in range(64):
            s += l1_buf[i] * np.int64(W.get((1, n, i), 0))
        s += np.int64(BIAS) * np.int64(W.get((1, n, 64), 0))
        l2_acc[n] = s
    l2_out = np.maximum(0, l2_acc) # ReLU (48-bit signed)
    # hardware uses l2_buffer[j][31:0] as L3 input
    l2_buf = np.array([to_int32(v) for v in l2_out], dtype=np.int64)

    # Layer 3: 3 neurons, 32 inputs + bias at col 32
    # (testbench only loads cols 0-16. Cols 17-32 are 0 from BRAM initialisation)
    l3_acc = np.zeros(3, dtype=np.int64)
    for n in range(3):
        s = np.int64(0)
        for i in range(32):
            s += l2_buf[i] * np.int64(W.get((2, n, i), 0))
        s += np.int64(BIAS) * np.int64(W.get((2, n, 32), 0))
        l3_acc[n] = s

    # Normalisation
    z_offset = -7229984 # 0xFF91ADE0 as signed int32
    z_scale  = 1
    diff   = l3_acc - np.int64(z_offset)
    scaled = diff   * np.int64(z_scale)
    result = scaled >> np.int64(16)
    rgb = np.clip(result, 0, 255).astype(np.int32) # clamp

    return int(rgb[0]), int(rgb[1]), int(rgb[2]), l3_acc

# Main
if __name__ == '__main__':
    path = 'C:/Users/rhysl/Documents/DesignProject/Snytax_11/tb_pixel_generator.sv'
    W = parse_weights(path)
    print(f"Parsed {len(W)} weight entries (L1={sum(1 for k in W if k[0]==0)}, "
          f"L2={sum(1 for k in W if k[0]==1)}, L3={sum(1 for k in W if k[0]==2)})\n")

    # Pixel (0,0)
    r, g, b, l3 = mlp_pixel(0, 0, W)
    print("Pixel (0,0)")
    print(f"  L3 raw acc:  R={l3[0]},  G={l3[1]},  B={l3[2]}")
    print(f"  After norm:  R={r},  G={g},  B={b}")
    print(f"  Simulation:  R=43, G=130, B=255")
    ok = (r == 43 and g == 130 and b == 255)
    print(f"  {'MATCH' if ok else 'MISMATCH'}\n")

    # First 4 pixels on line 0 (matches stream words 0,1,2)
    print("First 4 pixels on line 0")
    print(f"  {'px':>4}  {'R':>3} {'G':>3} {'B':>3}")
    for x in range(4):
        r, g, b, _ = mlp_pixel(x, 0, W)
        print(f"  ({x},0)  {r:3d} {g:3d} {b:3d}")

    print()
    print("Stream word 0 = {G[1], R[0], B[0], G[0]}")
    r0,g0,b0,_ = mlp_pixel(0,0,W)
    r1,g1,b1,_ = mlp_pixel(1,0,W)
    r2,g2,b2,_ = mlp_pixel(2,0,W)
    r3,g3,b3,_ = mlp_pixel(3,0,W)
    w0 = (g1 << 24) | (r0 << 16) | (b0 << 8) | g0
    w1 = (b2 << 24) | (g2 << 16) | (r1 << 8) | b1
    w2 = (r3 << 24) | (b3 << 16) | (g3 << 8) | r2
    print(f"Calculated word 0 = {w0:08x}  (sim: 822bff82)")
    print(f"Calculated word 1 = {w1:08x}  (sim: ff822bff)")
    print(f"Calculated word 2 = {w2:08x}  (sim: 2bff832b)")
    all_match = (f"{w0:08x}" == "822bff82" and
                 f"{w1:08x}" == "ff822bff" and
                 f"{w2:08x}" == "2bff832b")
    print(f"Packed words {'MATCH' if all_match else 'MISMATCH'}")