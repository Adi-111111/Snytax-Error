# Changes — z_shift normalization fix + weight export

All changes are relative to commit `288eb11`.

---

## Problem

PyTorch-trained weights produce `l3_acc` values in the ±500 billion range.  
`z_offset` is a 32-bit register (±2.1 billion) — fundamentally too small to centre that range.  
The original 3-stage pipeline `(l3_acc − z_offset) × z_scale >> 16` produced garbage output for any real trained network.

---

## HDL — `Learning/hdl/mlp_pipeline.sv`

### New port
```systemverilog
input [4:0] z_shift,
```
Sits alongside `z_offset` and `z_scale` in the normalisation port group.

### Normalization pipeline: 3 stages → 4 stages

**Before**
```
l3_acc  →  (subtract z_offset)  →  (× z_scale)  →  (>> 16 + clamp)  →  RGB
```
**After**
```
l3_acc  →  (>>> z_shift)  →  (subtract z_offset)  →  (× z_scale)  →  (>>> 16 + clamp)  →  RGB
```

Stage 1 arithmetically right-shifts `l3_acc` by `z_shift` bits, collapsing the ±500B range to ±hundreds before `z_offset` is applied. With `z_shift=32`, the shifted range fits comfortably in a 32-bit signed register.

### New pipeline registers
```systemverilog
reg signed [47:0] l3_shifted [H3_SIZE-1:0];  // stage 1 output
reg               l3_shift_valid;             // stage 1 handshake
```

### Bug fix — `l3_shift_valid` missing from reset/clear
`l3_shift_valid` added to both:
- the top-of-always-block self-clearing defaults
- the `!resetn` reset block

Without this, `l3_shift_valid` would stay high across a reset, causing the pipeline to re-fire spuriously.

### Bug fix — logical shift `>>` replaced with arithmetic shift `>>>`
```systemverilog
// Before (wrong for negative l3_scaled):
r_norm = l3_scaled[0] >> 16;

// After:
r_norm = l3_scaled[0] >>> 16;
```
`>>` is a logical shift (fills zeros). For a negative `l3_scaled`, this produces a large positive value that always clamps to 255 instead of 0. Any output neuron with net-negative activation would show 255 (fully saturated) rather than 0 (black). Fixed to `>>>` (arithmetic, fills sign bit).

### Bug fix — unsigned multiply on signed value
```systemverilog
// Before:
l3_scaled[j] <= l3_diff[j] * z_scale;

// After:
l3_scaled[j] <= $signed(l3_diff[j]) * $signed({1'b0, z_scale});
```
`z_scale` is declared `[31:0]` (unsigned). Multiplying a signed value by an unsigned operand causes Verilog to treat the entire expression as unsigned, corrupting negative `l3_diff` values. The fix prepends a zero bit and casts to signed so the multiply is always signed × signed.

---

## HDL — `Learning/hdl/pixel_generator.sv`

### New register — `regfile[74]` = `z_shift`
```systemverilog
regfile[74] = 32'd16;  // default: 16-bit shift (safe for existing test weights)
```

### New wire
```systemverilog
wire [4:0] z_shift = regfile[74][4:0];
```

### AXI-Lite address
| Register | Byte address | Description |
|----------|-------------|-------------|
| `regfile[74]` | `0x128` | `z_shift` (5-bit, unsigned) |

(existing: `0x120` = `z_offset`, `0x124` = `z_scale`)

### MLP instantiation
```systemverilog
.z_shift(z_shift),   // new
```

---

## New testbench — `Learning/hdl/tb_zshift.sv`

Tests the full 4-stage normalisation pipeline with Q4.4 weights.

**Weight setup** (all valid Q4.4 values):

| Layer | Weight | Q4.4 float |
|-------|--------|-----------|
| L1, L2 | `int8 = 16` | 1.0 |
| L3 R   | `int8 = 64` | 4.0 |
| L3 G   | `int8 = 32` | 2.0 |
| L3 B   | `int8 = -32` (0xE0) | -2.0 |

**Expected accumulator:** `l3_acc = [2^34, 2^33, -2^33]`

| Test | z_shift | z_offset | z_scale | Expected RGB |
|------|---------|----------|---------|-------------|
| 1 | 26 | 0 | 65536 | (255, 128, 0) |
| 2 | 27 | 0 | 65536 | (128, 64, 0) |
| 3 | 27 | -64 | 65536 | (192, 128, 0) |
| 4 | 0  | 0 | 65536 | (255, 255, 0) — saturate test |
| 5 | 26 | 0 | 32768 | (128, 64, 0) — z_scale test |

All 5 pass. The testbench also caught the `>>>` bug and the wrong L3 bank (3 vs 2) in the old `tb_mlp_pipeline.v`.

**Run:**
```bash
iverilog -g2012 -o sim_zshift tb_zshift.sv mlp_pipeline.sv weight_ram.sv mac_layer.sv
vvp sim_zshift
```

---

## Python — weight quantisation format

All training and export scripts changed from **per-layer adaptive int8** to **Q4.4 fixed-point**.

| | Adaptive int8 (before) | Q4.4 (after) |
|--|------------------------|--------------|
| Scale | `max_w / 127` per layer | `1/16` fixed |
| Float range | unbounded (full ±127 range used) | ±7.9375 |
| Formula | `round(w × 127 / max_w)` | `round(w × 16)` |

Files updated:
- `Learning/demo_iris.py` — `quantize()` and `hw_calibrate()` extract function
- `Learning/demo_blend.py` — `quantize()`
- `Learning/train_image.py` — `quantize_weights()`

**Training note:** add `weight_decay=1e-4` to the Adam optimizer to keep weights within the Q4.4 ±8 range and avoid clipping during quantization.

---

## New file — `Learning/export_weights.py`

Exports a trained PyTorch model to hardware-ready format.

**What it does:**
1. Quantizes all layers to Q4.4 (`round(w × 16)`)
2. Sweeps the full input grid through the exact fixed-point pipeline to find `l3_acc` range
3. Computes `z_shift`, `z_offset`, `z_scale`
4. Outputs **`mlp_weights.h`** — C header with `load_mlp_weights()` function for PS firmware
5. Outputs **`load_weights_pynq.py`** — PYNQ script using `MMIO` for Jupyter loading

**Weight address encoding written to `regfile[69]`:**
```
bits [15:14]  layer bank   (0=L1, 1=L2, 2=L3)
bits [13:8]   neuron index
bits  [7:0]   column       (0..N_inputs-1 for weights, N_inputs for bias)
```

**Usage:**
```python
from export_weights import export
export(model_q, feature_regs_float=feat_regs, axis_x=0, axis_y=1,
       base_addr=0x43C00000)   # set base_addr from Vivado Address Editor
```

**PS loading sequence (per weight):**
```
write regfile[69] = (layer << 14) | (neuron << 8) | col  # weight_addr
write regfile[70] = int8_value & 0xFF                     # weight_data
write regfile[71] = 1                                     # pulse we
write regfile[71] = 0
```
