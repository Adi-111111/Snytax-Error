#!/usr/bin/env python3
"""
Export a trained PyTorch MLP to the hardware weight format.

Weight format: raw signed int8 (NOT Q4.4).
  - Inputs to the MLP are Q1.15: x_fixed = round(x_float * 32768)
  - Weights are stored as int8: w_int8 = round(w_float * 127 / max_abs_weight)
  - The MAC computes x_int * w_int8 — the result is large, z_shift handles that
  - Biases use the same scale as the weights for their layer

Address encoding written to regfile[69]:
  bits [15:14] = layer bank  (0=L1, 1=L2, 2=L3)
  bits [13:8]  = neuron index
  bits [7:0]   = column (0..N_inputs-1 for weights, N_inputs for bias)

AXI-Lite register byte addresses (= reg_index * 4):
  0x114 = regfile[69]  weight_addr
  0x118 = regfile[70]  weight_data  (unsigned byte, two's-complement for negatives)
  0x11C = regfile[71]  weight_we    (pulse 1 then 0 to commit)
  0x120 = regfile[72]  z_offset     (signed 32-bit)
  0x124 = regfile[73]  z_scale      (unsigned 32-bit)
  0x128 = regfile[74]  z_shift      (unsigned, 5-bit)

Usage:
  from export_weights import export
  export(model_q, axis_x=0, axis_y=1, feature_regs=feature_regs)
"""

import math
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Architecture constants — must match pixel_generator.sv parameters
# ---------------------------------------------------------------------------
N_FEATURES = 16
H1_SIZE    = 64
H2_SIZE    = 32
H3_SIZE    = 3


# ---------------------------------------------------------------------------
# AXI-Lite register indices (regfile[n] sits at byte address n*4)
# ---------------------------------------------------------------------------
REG_WEIGHT_ADDR = 69
REG_WEIGHT_DATA = 70
REG_WEIGHT_WE   = 71
REG_Z_OFFSET    = 72
REG_Z_SCALE     = 73
REG_Z_SHIFT     = 74


# ---------------------------------------------------------------------------
# Quantise one linear layer to int8
# Weights and biases share the same per-layer scale = max_abs_weight / 127.
# Hardware multiplies bias by BIAS_CONST=32767 (≈2^15, same scale as Q1.15
# inputs), so it must be quantised with the weight scale to stay compatible.
# ---------------------------------------------------------------------------
Q44_SCALE = 16.0  # Q4.4: 4 integer bits + 4 fractional bits, 1 LSB = 1/16

def _quantise_layer(layer: nn.Linear):
    w = layer.weight.detach().numpy()      # (N_out, N_in)
    b = layer.bias.detach().numpy()        # (N_out,)
    w8 = np.round(w * Q44_SCALE).clip(-128, 127).astype(np.int8)
    b8 = np.round(b * Q44_SCALE).clip(-128, 127).astype(np.int8)
    return w8, b8                          # w8: (N_out, N_in), b8: (N_out,)


# ---------------------------------------------------------------------------
# Compute calibration registers for the normalization pipeline
# ---------------------------------------------------------------------------
def _calibrate(model_q, feature_regs_float, axis_x, axis_y, grid=64):
    """
    Sweep a pixel grid through the fixed-point pipeline and find the
    l3_acc range needed to set z_shift, z_offset, z_scale.
    """
    linears = [l for l in model_q.net if isinstance(l, nn.Linear)]
    assert len(linears) == 3, "expected 3 Linear layers"

    w8_1, b8_1 = _quantise_layer(linears[0])  # Q4.4: round(w * 16)
    w8_2, b8_2 = _quantise_layer(linears[1])
    w8_3, b8_3 = _quantise_layer(linears[2])

    BIAS = np.int64(32767)

    def trunc32(v):
        v = v & 0xFFFFFFFF
        return np.where(v >= 0x80000000, v - 0x100000000, v)

    axis_q = np.linspace(0, 32767, grid, dtype=np.int64)
    gx, gy = np.meshgrid(axis_q, axis_q, indexing='ij')
    N = grid * grid

    held = np.round(np.clip(feature_regs_float, 0, 0.9999) * 32768).astype(np.int64)
    x = np.tile(held, (N, 1))
    x[:, axis_x] = gx.flatten()
    x[:, axis_y] = gy.flatten()

    l1 = trunc32(np.maximum(0, x @ w8_1.astype(np.int64).T + BIAS * b8_1.astype(np.int64)))
    l2 = trunc32(np.maximum(0, l1 @ w8_2.astype(np.int64).T + BIAS * b8_2.astype(np.int64)))
    l3 = l2 @ w8_3.astype(np.int64).T + BIAS * b8_3.astype(np.int64)

    g_max = int(np.abs(l3).max())
    z_shift = max(0, math.ceil(math.log2(g_max / 255))) if g_max > 255 else 0

    shifted = l3 >> z_shift
    s_min   = int(shifted.min())
    s_max   = int(shifted.max())
    z_offset = s_min                                    # shift base to 0
    span     = max(s_max - s_min, 1)
    z_scale  = min(int(math.ceil(255 * 65536 / span)), 0xFFFFFFFF)

    return z_shift, z_offset, z_scale, l3


# ---------------------------------------------------------------------------
# Build the flat AXI write sequence
# ---------------------------------------------------------------------------
def _weight_writes(layer_bank: int, w8: np.ndarray, b8: np.ndarray):
    """
    Yield (reg_weight_addr_val, uint8_data) pairs for every weight and bias.
    w8: (N_neurons, N_inputs)   b8: (N_neurons,)
    """
    n_neurons, n_inputs = w8.shape
    for n in range(n_neurons):
        for c in range(n_inputs):
            addr = (layer_bank << 14) | (n << 8) | c
            yield addr, int(w8[n, c]) & 0xFF
        # bias is stored at column = n_inputs
        addr = (layer_bank << 14) | (n << 8) | n_inputs
        yield addr, int(b8[n]) & 0xFF


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_writes(model_q, feature_regs_float=None, axis_x=0, axis_y=1,
                 calib_grid=64):
    """
    Returns a dict with:
      'weights'  — list of (addr_val, uint8_data) for all three layers
      'z_shift'  — int
      'z_offset' — int (signed)
      'z_scale'  — int (unsigned)
    """
    linears = [l for l in model_q.net if isinstance(l, nn.Linear)]
    w1, b1 = _quantise_layer(linears[0])
    w2, b2 = _quantise_layer(linears[1])
    w3, b3 = _quantise_layer(linears[2])

    writes = []
    writes += list(_weight_writes(0, w1, b1))   # L1
    writes += list(_weight_writes(1, w2, b2))   # L2
    writes += list(_weight_writes(2, w3, b3))   # L3

    z_shift = z_offset = z_scale = 0
    if feature_regs_float is not None:
        z_shift, z_offset, z_scale, _ = _calibrate(
            model_q, feature_regs_float, axis_x, axis_y, calib_grid
        )

    return {
        'weights':  writes,
        'z_shift':  z_shift,
        'z_offset': z_offset & 0xFFFFFFFF,   # two's complement for C
        'z_scale':  z_scale,
    }


# ---------------------------------------------------------------------------
# Emit a C header — drop into PS firmware, call load_mlp_weights() at boot
# ---------------------------------------------------------------------------
def to_c_header(data: dict, base_addr: int = 0x43C00000,
                out_path: str = 'mlp_weights.h'):
    writes  = data['weights']
    z_shift  = data['z_shift']
    z_offset = data['z_offset']
    z_scale  = data['z_scale']

    lines = [
        '/* Auto-generated by export_weights.py — do not edit */',
        '#pragma once',
        '#include <stdint.h>',
        '',
        f'#define PIXEL_GEN_BASE  0x{base_addr:08X}UL',
        '',
        'static inline void _pg_reg_write(uint32_t idx, uint32_t val) {',
        '    volatile uint32_t *b = (volatile uint32_t *)PIXEL_GEN_BASE;',
        '    b[idx] = val;',
        '}',
        '',
        'static inline void _pg_write_weight(uint32_t addr_val, uint8_t data) {',
        '    _pg_reg_write(69, addr_val);   /* weight_addr */',
        '    _pg_reg_write(70, data);       /* weight_data */',
        '    _pg_reg_write(71, 1);          /* pulse we    */',
        '    _pg_reg_write(71, 0);',
        '}',
        '',
        'static inline void load_mlp_weights(void) {',
    ]

    for addr_val, byte in writes:
        lines.append(f'    _pg_write_weight(0x{addr_val:04X}u, 0x{byte:02X}u);')

    lines += [
        '',
        f'    _pg_reg_write(74, {z_shift}u);          /* z_shift  */',
        f'    _pg_reg_write(72, 0x{z_offset:08X}u);   /* z_offset (signed) */',
        f'    _pg_reg_write(73, {z_scale}u);           /* z_scale  */',
        '}',
    ]

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f'Wrote {out_path}  ({len(writes)} weight writes)')


# ---------------------------------------------------------------------------
# Emit a hex file — one line per entry: "ADDR DD"
#
# Format:
#   AAAA DD    — 4-hex address, 2-hex data byte, space-separated
#   The final 3 lines are norm registers: z_shift, z_offset, z_scale
#     written as: "REG_IDX XXXXXXXX" (8-hex 32-bit value)
#
# Load in Jupyter with the snippet printed to stdout by this function.
# ---------------------------------------------------------------------------
def to_hex_file(data: dict, out_path: str = 'weights.hex'):
    writes   = data['weights']
    z_shift  = data['z_shift']
    z_offset = data['z_offset']
    z_scale  = data['z_scale']

    lines = []
    for addr_val, byte in writes:
        lines.append(f'{addr_val:04X} {byte:02X}')

    # norm registers appended as special entries with reg index as "address"
    lines.append(f'# z_shift={z_shift}  z_offset=0x{z_offset:08X}  z_scale={z_scale}')
    lines.append(f'4A {z_shift:08X}')   # reg 74 = 0x4A
    lines.append(f'48 {z_offset:08X}')  # reg 72 = 0x48
    lines.append(f'49 {z_scale:08X}')   # reg 73 = 0x49

    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')

    print(f'Wrote {out_path}  ({len(writes)} weight entries)')
    print()
    print('--- Jupyter loading snippet ---')
    print(f'mmio = MMIO(BASE, 0x400)  # set BASE to your AXI-Lite address')
    print()
    print('def reg_write(idx, val):')
    print('    mmio.write(idx * 4, int(val) & 0xFFFFFFFF)')
    print()
    print(f'with open("{out_path}") as f:')
    print('    for line in f:')
    print('        line = line.strip()')
    print('        if not line or line.startswith("#"): continue')
    print('        parts = line.split()')
    print('        addr, data = int(parts[0], 16), int(parts[1], 16)')
    print('        if addr <= 0x8FFF:                  # weight entry')
    print('            reg_write(69, addr)             # weight_addr')
    print('            reg_write(70, data & 0xFF)      # weight_data')
    print('            reg_write(71, 1)                # pulse we')
    print('            reg_write(71, 0)')
    print('        else:                               # norm register (0x4A/48/49)')
    print('            reg_write(addr, data)')


# ---------------------------------------------------------------------------
# One-shot convenience wrapper
# ---------------------------------------------------------------------------
def export(model_q, feature_regs_float=None, axis_x=0, axis_y=1,
           base_addr=0x43C00000, calib_grid=64):
    """
    Quantise model_q, run calibration, write weights.hex + mlp_weights.h.
    Set base_addr to your AXI-Lite peripheral base from Vivado Address Editor.
    """
    data = build_writes(model_q, feature_regs_float, axis_x, axis_y, calib_grid)

    print(f'\nCalibration:')
    print(f'  z_shift  = {data["z_shift"]}')
    print(f'  z_offset = {data["z_offset"]} (0x{data["z_offset"]:08X})')
    print(f'  z_scale  = {data["z_scale"]}')
    print(f'  total weight writes = {len(data["weights"])}')
    print()

    to_hex_file(data)
    to_c_header(data, base_addr)


# ---------------------------------------------------------------------------
# Quick self-test with a tiny random model
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import copy

    class TinyMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(N_FEATURES, H1_SIZE), nn.ReLU(),
                nn.Linear(H1_SIZE,    H2_SIZE), nn.ReLU(),
                nn.Linear(H2_SIZE,    H3_SIZE),
            )

    def quantize(model):
        q = copy.deepcopy(model)
        with torch.no_grad():
            for layer in q.net:
                if isinstance(layer, nn.Linear):
                    w = layer.weight.data
                    s = w.abs().max() / 127.0
                    if s > 0:
                        layer.weight.data = (w / s).round().clamp(-127, 127) * s
        return q

    torch.manual_seed(42)
    model   = TinyMLP()
    model_q = quantize(model)

    feat_regs = np.zeros(N_FEATURES, dtype=np.float32)
    feat_regs[:4] = 0.5   # non-axis features held at 0.5

    export(model_q, feature_regs_float=feat_regs, axis_x=0, axis_y=1,
           base_addr=0x43C00000)
