#!/usr/bin/env python3
"""
Academic-style horizontal architecture diagrams for all 4 AOG Detection models.
Outputs 4 PNG files in outputs/architecture_diagrams/.

Models:
  1. Intensity Threshold Baseline
  2. Vanilla U-Net  (unet.py, 1-ch input, custom)
  3. U-Net + ResNet34  (smp.Unet, UNet1.py)
  4. U-Net++ + ResNet34  (smp.UnetPlusPlus, UNetPP_resnet34.py)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = "outputs/architecture_diagrams"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────── Color palette ────────────────────────────────────
C = {
    "input":   "#455A64",   # blue-gray
    "output":  "#4A148C",   # deep-purple
    "enc":     "#1565C0",   # blue  (vanilla encoder)
    "dec":     "#2E7D32",   # green (decoder)
    "bn":      "#BF360C",   # deep-orange (bottleneck)
    "resnet":  "#B71C1C",   # dark-red (ResNet34)
    "unetpp":  "#6A1B9A",   # purple (dense nodes)
    "preproc": "#0277BD",   # light-blue
    "thresh":  "#E65100",   # orange
    "morph":   "#00695C",   # teal
    "pool":    "#311B92",   # deep-purple (down-sample)
    "up":      "#1B5E20",   # dark-green (upsample / ConvT)
    "cat":     "#FF6F00",   # amber (concat symbol)
    "skip":    "#546E7A",   # gray-blue (skip arrows)
    "dense":   "#AD1457",   # pink (UNet++ cross-level)
}
W = "white"
DPI = 200


# ─────────────────────────── Primitive helpers ────────────────────────────────

def make_fig(fw, fh):
    fig, ax = plt.subplots(figsize=(fw, fh), facecolor="white")
    ax.set_xlim(0, fw)
    ax.set_ylim(0, fh)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def blk(ax, cx, cy, w, h, color, text, sub="", fs=8.5, sfs=6.2, tc=W, alpha=0.93):
    """Rounded-rectangle block with optional subtitle."""
    rect = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.06",
        facecolor=color, edgecolor=W,
        linewidth=1.6, alpha=alpha, zorder=3,
    )
    ax.add_patch(rect)
    if sub:
        ax.text(cx, cy + h * 0.16, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=4)
        ax.text(cx, cy - h * 0.22, sub, ha="center", va="center",
                fontsize=sfs, color=tc, zorder=4, alpha=0.88)
    else:
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=fs, fontweight="bold", color=tc, zorder=4)


def arr(ax, x1, y1, x2, y2, c="#37474F", lw=1.5, ms=11, ls="-", cs="arc3,rad=0"):
    ax.annotate(
        "", xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle="->", color=c, lw=lw,
            mutation_scale=ms, linestyle=ls,
            connectionstyle=cs,
        ),
        zorder=5,
    )


def cat_sym(ax, cx, cy, r=0.115):
    """Concat ⊕ symbol."""
    circ = plt.Circle((cx, cy), r, facecolor=C["cat"], edgecolor=W,
                      linewidth=1.2, zorder=4)
    ax.add_patch(circ)
    ax.text(cx, cy, "⊕", ha="center", va="center",
            fontsize=7.5, color=W, zorder=5, fontweight="bold")


def fig_title(ax, text):
    ax.text(0.5, 0.98, text, transform=ax.transAxes,
            ha="center", va="top", fontsize=12.5, fontweight="bold", color="#212121")


def draw_legend(ax, items, x0, y0, dx=2.15, bw=1.0, bh=0.44, fs=6.2):
    for i, (col, lbl) in enumerate(items):
        cx = x0 + i * dx
        r = FancyBboxPatch(
            (cx - bw / 2, y0 - bh / 2), bw, bh,
            boxstyle="round,pad=0.04", facecolor=col,
            edgecolor=W, linewidth=1, alpha=0.88, zorder=3,
        )
        ax.add_patch(r)
        ax.text(cx, y0 - bh / 2 - 0.22, lbl, ha="center", va="top",
                fontsize=fs, color="#424242")


# ══════════════════════════════════════════════════════════════════════════════
# Model 1 – Intensity Threshold Baseline
# ══════════════════════════════════════════════════════════════════════════════

def draw_intensity():
    fw, fh = 18, 4.2
    fig, ax = make_fig(fw, fh)
    fig_title(ax, "Model 1: Intensity Threshold Baseline")

    bw, bh = 2.0, 1.1
    y = 2.5

    steps = [
        (1.3,  C["input"],   "Input",           "(H×W, Grayscale)"),
        (3.7,  C["preproc"], "Median Blur",      "kernel = 3"),
        (6.0,  C["preproc"], "CLAHE",            "clip=2.0,  tile=8×8"),
        (8.4,  C["thresh"],  "Intensity\nThreshold", "τ ∈ [70, 180]  (grid-search)"),
        (10.9, C["morph"],   "Morph. Open",      "ellipse, k=3×3"),
        (13.4, C["morph"],   "Morph. Close",     "ellipse, k=3×3"),
        (15.8, C["morph"],   "Min-Area\nFilter", "A_min  (grid-search)"),
    ]
    for cx, col, txt, sub in steps:
        blk(ax, cx, y, bw, bh, col, txt, sub, fs=8, sfs=6)

    # arrows between blocks
    for i in range(len(steps) - 1):
        arr(ax, steps[i][0] + bw / 2, y, steps[i + 1][0] - bw / 2, y)

    # output block + arrow
    arr(ax, steps[-1][0] + bw / 2, y, 17.9 - 0.55, y)
    blk(ax, 17.6, y, 1.1, bh, C["output"], "Output\nMask", "(H×W, Binary)", fs=7.5, sfs=5.5)

    ax.text(fw / 2, 1.0,
            "τ and A_min are selected by exhaustive grid-search on validation set (selection metric: mean Dice)",
            ha="center", va="center", fontsize=8, color="#546E7A", style="italic")

    legend_items = [
        (C["preproc"], "Pre-\nprocessing"),
        (C["thresh"],  "Thresholding"),
        (C["morph"],   "Morphological\nOp."),
        (C["output"],  "Output"),
    ]
    draw_legend(ax, legend_items, x0=5.0, y0=0.38)
    fig.tight_layout(pad=0.4)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Model 2 – Vanilla U-Net  (unet.py, 1-ch grayscale input)
# ══════════════════════════════════════════════════════════════════════════════

def draw_vanilla_unet():
    fw, fh = 18, 10
    fig, ax = make_fig(fw, fh)
    fig_title(ax, "Model 2: Vanilla U-Net  (1-ch grayscale input, custom)")

    enc_x = 3.0
    dec_x = 15.0
    bw, bh = 2.2, 0.92

    # y-positions for levels 0-2 + bottleneck
    ys = [8.6, 6.8, 5.0, 2.8]   # enc/dec level 0,1,2  +  BN

    # ── Encoder ──────────────────────────────────────────────────────────────
    blk(ax, 0.95, ys[0], 1.35, bh, C["input"], "Input", "1×256×256", fs=8, sfs=6)
    enc_specs = [
        ("DoubleConv", "64ch · 256×256"),
        ("DoubleConv", "128ch · 128×128"),
        ("DoubleConv", "256ch · 64×64"),
    ]
    for i, (txt, sub) in enumerate(enc_specs):
        blk(ax, enc_x, ys[i], bw, bh, C["enc"], txt, sub)

    # MaxPool between encoder levels
    for i in range(2):
        my = (ys[i] + ys[i + 1]) / 2
        blk(ax, enc_x, my, 1.5, 0.45, C["pool"], "MaxPool2d", "÷2", fs=7.5, sfs=6)
        arr(ax, enc_x, ys[i] - bh / 2, enc_x, my + 0.22)
        arr(ax, enc_x, my - 0.22, enc_x, ys[i + 1] + bh / 2)

    # Pool from enc2 → BN  (goes down-then-right)
    pool_y = ys[2] - bh / 2 - 0.55
    blk(ax, enc_x, pool_y, 1.5, 0.45, C["pool"], "MaxPool2d", "÷2", fs=7.5, sfs=6)
    arr(ax, enc_x, ys[2] - bh / 2, enc_x, pool_y + 0.22)

    # ── Bottleneck ────────────────────────────────────────────────────────────
    bn_x = (enc_x + dec_x) / 2
    arr(ax, enc_x, pool_y - 0.22, bn_x - 1.3, ys[3], c="#37474F",
        cs="arc3,rad=0.25")
    blk(ax, bn_x, ys[3], 2.6, bh, C["bn"], "Bottleneck (DoubleConv)", "512ch · 32×32",
        fs=9, sfs=7)

    # ── Decoder ──────────────────────────────────────────────────────────────
    dec_specs = [
        ("DoubleConv", "256ch · 64×64"),
        ("DoubleConv", "128ch · 128×128"),
        ("DoubleConv", "64ch · 256×256"),
    ]
    for i, (txt, sub) in enumerate(dec_specs):
        level = 2 - i              # dec0 maps to enc2 (y=ys[2]), etc.
        blk(ax, dec_x, ys[level], bw, bh, C["dec"], txt, sub)
        cat_sym(ax, dec_x - bw / 2 - 0.14, ys[level])

    # ConvTranspose between decoder levels  (going up)
    for i in range(2):
        y_from = ys[2 - i]
        y_to   = ys[1 - i]
        my = (y_from + y_to) / 2
        blk(ax, dec_x, my, 1.7, 0.45, C["up"], "ConvT2d ×2", "", fs=7.5)
        arr(ax, dec_x, y_from + bh / 2, dec_x, my - 0.22)
        arr(ax, dec_x, my + 0.22, dec_x, y_to - bh / 2)

    # BN → first decoder block (diagonal)
    arr(ax, bn_x + 1.3, ys[3], dec_x - bw / 2 - 0.14 - 0.12, ys[2],
        c="#37474F", cs="arc3,rad=-0.25")

    # ── Output head ──────────────────────────────────────────────────────────
    blk(ax, fw - 1.0, ys[0], 1.4, bh, C["output"], "Output", "1×256×256\n(logits)", fs=7.5, sfs=5.5)
    arr(ax, dec_x + bw / 2, ys[0], fw - 1.0 - 0.7, ys[0])
    arr(ax, 0.95 + 0.68, ys[0], enc_x - bw / 2, ys[0])

    # ── Skip connections ──────────────────────────────────────────────────────
    for i in range(3):
        arr(ax, enc_x + bw / 2, ys[i], dec_x - bw / 2 - 0.3, ys[i],
            c=C["skip"], lw=1.2, ms=9, ls="--")
        mid_x = (enc_x + bw / 2 + dec_x - bw / 2) / 2
        ax.text(mid_x, ys[i] + 0.17, "skip", ha="center", va="bottom",
                fontsize=6.5, color=C["skip"], style="italic")

    # ── Loss annotation ───────────────────────────────────────────────────────
    ax.text(fw - 1.0, ys[0] - bh / 2 - 0.38,
            "Loss:\nBCE + Dice", ha="center", va="top",
            fontsize=7.5, color="#546E7A", style="italic")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        (C["input"],  "Input"),
        (C["enc"],    "Encoder\n(DoubleConv)"),
        (C["pool"],   "MaxPool2d"),
        (C["bn"],     "Bottleneck"),
        (C["up"],     "ConvT2d\n(Upsample)"),
        (C["dec"],    "Decoder\n(DoubleConv)"),
        (C["cat"],    "Concat (⊕)"),
        (C["skip"],   "Skip\nConnection"),
        (C["output"], "Output\nHead"),
    ]
    draw_legend(ax, legend_items, x0=0.8, y0=1.4, dx=1.92)
    fig.tight_layout(pad=0.4)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Model 3 – U-Net + ResNet34  (smp.Unet, UNet1.py)
# ══════════════════════════════════════════════════════════════════════════════

def draw_unet_resnet34():
    fw, fh = 20, 12
    fig, ax = make_fig(fw, fh)
    fig_title(ax,
        "Model 3: U-Net + ResNet34 Encoder  "
        "(smp.Unet, 3-ch input, ImageNet pretrained)")

    enc_x = 3.0
    dec_x = 17.0
    bw, bh = 2.4, 0.9

    # 5 encoder levels
    ys = [10.5, 9.0, 7.5, 6.0, 4.5, 2.6]   # enc0..enc4, BN_y

    enc_lbl = [
        ("ResNet34 Stem",   "64ch · 128×128"),
        ("ResNet34 Layer1", "64ch · 64×64"),
        ("ResNet34 Layer2", "128ch · 32×32"),
        ("ResNet34 Layer3", "256ch · 16×16"),
        ("ResNet34 Layer4", "512ch · 8×8"),
    ]
    dec_lbl = [
        ("Decoder Block", "256ch · 16×16"),
        ("Decoder Block", "128ch · 32×32"),
        ("Decoder Block", "64ch · 64×64"),
        ("Decoder Block", "64ch · 128×128"),
        ("Decoder Block", "32ch · 256×256"),
    ]

    # ── Encoder ──────────────────────────────────────────────────────────────
    blk(ax, 1.1, ys[0], 1.4, bh, C["input"], "Input", "3×256×256", fs=8, sfs=6)
    for i, (txt, sub) in enumerate(enc_lbl):
        blk(ax, enc_x, ys[i], bw, bh, C["resnet"], txt, sub, fs=8, sfs=6)

    # stride-2 between encoder stages
    for i in range(4):
        my = (ys[i] + ys[i + 1]) / 2
        blk(ax, enc_x, my, 1.5, 0.42, C["pool"], "stride-2", "", fs=7.5)
        arr(ax, enc_x, ys[i] - bh / 2, enc_x, my + 0.21)
        arr(ax, enc_x, my - 0.21, enc_x, ys[i + 1] + bh / 2)

    # enc4 → BN
    bn_x = (enc_x + dec_x) / 2
    arr(ax, enc_x, ys[4] - bh / 2, bn_x - 1.4, ys[5],
        cs="arc3,rad=0.28")
    blk(ax, bn_x, ys[5], 2.8, bh, C["bn"],
        "Encoder Output  (Bottleneck)", "512ch · 8×8", fs=9, sfs=7)

    # ── Decoder ──────────────────────────────────────────────────────────────
    for i, (txt, sub) in enumerate(dec_lbl):
        level = 4 - i
        blk(ax, dec_x, ys[level], bw, bh, C["dec"], txt, sub, fs=8, sfs=6)
        cat_sym(ax, dec_x - bw / 2 - 0.15, ys[level])

    # bilinear upsample ×2 between decoder levels
    for i in range(4):
        y_from = ys[4 - i]
        y_to   = ys[3 - i]
        my = (y_from + y_to) / 2
        blk(ax, dec_x, my, 1.7, 0.42, C["up"], "Bilinear ×2", "", fs=7.5)
        arr(ax, dec_x, y_from + bh / 2, dec_x, my - 0.21)
        arr(ax, dec_x, my + 0.21, dec_x, y_to - bh / 2)

    # BN → dec4 (bottom-most decoder)
    arr(ax, bn_x + 1.4, ys[5], dec_x - bw / 2 - 0.15 - 0.13, ys[4],
        cs="arc3,rad=-0.28")

    # ── Output head ──────────────────────────────────────────────────────────
    blk(ax, fw - 1.1, ys[0], 1.6, bh, C["output"], "Seg. Head\n(Sigmoid)", "1×256×256",
        fs=7.5, sfs=5.5)
    arr(ax, dec_x + bw / 2, ys[0], fw - 1.1 - 0.8, ys[0])
    arr(ax, 1.1 + 0.7, ys[0], enc_x - bw / 2, ys[0])
    ax.text(fw - 1.1, ys[0] - bh / 2 - 0.4,
            "Loss:\nBCE + Dice", ha="center", va="top",
            fontsize=7.5, color="#546E7A", style="italic")

    # ── Skip connections ──────────────────────────────────────────────────────
    for i in range(5):
        arr(ax, enc_x + bw / 2, ys[i], dec_x - bw / 2 - 0.3, ys[i],
            c=C["skip"], lw=1.2, ms=9, ls="--")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        (C["input"],   "Input"),
        (C["resnet"],  "ResNet34\nEncoder"),
        (C["pool"],    "Stride-2\nDown"),
        (C["bn"],      "Bottleneck"),
        (C["up"],      "Bilinear\nUpsample"),
        (C["dec"],     "Decoder\nBlock"),
        (C["cat"],     "Concat (⊕)"),
        (C["skip"],    "Skip\nConnection"),
        (C["output"],  "Sigmoid\nHead"),
    ]
    draw_legend(ax, legend_items, x0=0.9, y0=1.3, dx=2.05)
    fig.tight_layout(pad=0.4)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Model 4 – U-Net++ + ResNet34  (smp.UnetPlusPlus, UNetPP_resnet34.py)
# ══════════════════════════════════════════════════════════════════════════════

def draw_unetpp_resnet34():
    fw, fh = 24, 13
    fig, ax = make_fig(fw, fh)
    fig_title(ax,
        "Model 4: U-Net++ with ResNet34 Encoder  "
        "(smp.UnetPlusPlus, 3-ch input, ImageNet pretrained)")

    # UNet++ notation: x^{i,j}
    #   i = resolution level  (0 = finest, 4 = coarsest)
    #   j = column/step       (0 = encoder, 1,2,3,4 = dense nodes)
    #
    # Node x^{i,j} receives:
    #   - same-level: x^{i, j-1}
    #   - from below: Up( x^{i+1, j-1} )
    # Nodes exist for: 0 ≤ i ≤ 4,  1 ≤ j ≤ 4-i+1  (triangle)

    bw_enc = 2.1    # encoder block width
    bw_den = 1.7    # dense / decoder node width
    bh = 0.82

    # X-positions for columns j = 0..4
    col_xs = [2.8, 7.4, 11.8, 16.2, 20.6]

    # Y-positions for levels i = 0..4 + BN
    ys = [11.4, 9.8, 8.2, 6.6, 5.0, 3.2]   # i=0..4, BN

    enc_lbl = [
        ("ResNet34 Stem",   "64ch·128²"),
        ("ResNet34 L1",     "64ch·64²"),
        ("ResNet34 L2",     "128ch·32²"),
        ("ResNet34 L3",     "256ch·16²"),
        ("ResNet34 L4",     "512ch·8²"),
    ]

    # ── Encoder column (j=0) ─────────────────────────────────────────────────
    blk(ax, 1.0, ys[0], 1.4, bh, C["input"], "Input", "3×256×256", fs=7.5, sfs=5.5)
    for i, (txt, sub) in enumerate(enc_lbl):
        blk(ax, col_xs[0], ys[i], bw_enc, bh, C["resnet"], txt, sub, fs=7.5, sfs=5.5)

    # stride-2 down arrows
    for i in range(4):
        my = (ys[i] + ys[i + 1]) / 2
        blk(ax, col_xs[0], my, 1.4, 0.38, C["pool"], "stride-2", "", fs=7)
        arr(ax, col_xs[0], ys[i] - bh / 2, col_xs[0], my + 0.19)
        arr(ax, col_xs[0], my - 0.19, col_xs[0], ys[i + 1] + bh / 2)

    # Input → enc0
    arr(ax, 1.0 + 0.7, ys[0], col_xs[0] - bw_enc / 2, ys[0])

    # ── Dense nodes & decoder nodes ──────────────────────────────────────────
    # Nodes: (i, j) for j=1..4, i=0..(4-j)
    node_bw = {1: bw_den, 2: bw_den, 3: bw_den, 4: bw_den}
    node_col = {1: C["unetpp"], 2: C["unetpp"], 3: C["unetpp"], 4: C["dec"]}

    for j in range(1, 5):
        for i in range(5 - j):
            cx = col_xs[j]
            cy = ys[i]
            nw = node_bw[j]
            col = node_col[j]
            sub_map = {
                (0, 4): "64ch·128²",
                (1, 4): "64ch·64²",
                (2, 4): "128ch·32²",
                (3, 4): "256ch·16²",
            }
            sub = sub_map.get((i, j), "")
            label = f"x^{{{i},{j}}}"
            blk(ax, cx, cy, nw, bh, col, label, sub, fs=8, sfs=5.8)
            cat_sym(ax, cx - nw / 2 - 0.13, cy, r=0.10)

    # ── Draw connections ──────────────────────────────────────────────────────
    # For each node x^{i,j} (j>=1):
    #   a) horizontal skip from x^{i, j-1}  (same level)
    #   b) upsample arrow from x^{i+1, j-1} (level below)
    for j in range(1, 5):
        for i in range(5 - j):
            cx  = col_xs[j]
            cy  = ys[i]
            nw  = node_bw[j]
            pw  = bw_enc if j == 1 else node_bw[j - 1]   # prev column width

            # (a) same-level horizontal skip (dashed)
            arr(ax, col_xs[j - 1] + pw / 2, ys[i],
                cx - nw / 2 - 0.27, cy,
                c=C["skip"], lw=1.1, ms=8, ls="--")

            # (b) cross-level upsample from (i+1, j-1)  (curved arrow up)
            arr(ax, col_xs[j - 1], ys[i + 1] + bh / 2,
                cx - nw / 2 - 0.27, cy,
                c=C["dense"], lw=1.1, ms=8,
                cs="arc3,rad=-0.18")

    # ── Output head ──────────────────────────────────────────────────────────
    out_x = fw - 1.2
    blk(ax, out_x, ys[0], 1.6, bh, C["output"], "Seg. Head\n(Sigmoid)",
        "1×256×256", fs=7.5, sfs=5.5)
    arr(ax, col_xs[4] + bw_den / 2, ys[0], out_x - 0.8, ys[0])
    ax.text(out_x, ys[0] - bh / 2 - 0.42,
            "Loss: Focal\n+ Tversky",
            ha="center", va="top", fontsize=7, color="#546E7A", style="italic")

    # ── Column headers ────────────────────────────────────────────────────────
    headers = ["Encoder\n(j=0)", "Dense\n(j=1)", "Dense\n(j=2)",
               "Dense\n(j=3)", "Decoder\n(j=4)"]
    for j, hdr in enumerate(headers):
        ax.text(col_xs[j], ys[0] + bh / 2 + 0.52, hdr,
                ha="center", va="bottom", fontsize=8,
                color="#37474F", fontweight="bold")

    # ── Deep supervision note ─────────────────────────────────────────────────
    ax.text(fw / 2, 1.95,
            "Deep supervision: outputs from x^{0,1..4}; at inference, x^{0,4} is used.",
            ha="center", va="center", fontsize=8, color="#546E7A", style="italic")

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        (C["input"],   "Input"),
        (C["resnet"],  "ResNet34\nEncoder"),
        (C["pool"],    "Stride-2\nDown"),
        (C["unetpp"],  "Dense\nNode"),
        (C["dec"],     "Decoder\nOutput"),
        (C["cat"],     "Concat\n(⊕)"),
        (C["skip"],    "Same-level\nSkip"),
        (C["dense"],   "Cross-level\nUp-skip"),
        (C["output"],  "Sigmoid\nHead"),
    ]
    draw_legend(ax, legend_items, x0=1.0, y0=1.05, dx=2.55)
    fig.tight_layout(pad=0.4)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    jobs = [
        ("model1_intensity_baseline.png",  draw_intensity),
        ("model2_vanilla_unet.png",        draw_vanilla_unet),
        ("model3_unet_resnet34.png",       draw_unet_resnet34),
        ("model4_unetpp_resnet34.png",     draw_unetpp_resnet34),
    ]
    for fname, draw_fn in jobs:
        fig = draw_fn()
        path = os.path.join(OUTPUT_DIR, fname)
        fig.savefig(path, dpi=DPI, bbox_inches="tight",
                    facecolor="white", edgecolor="none")
        plt.close(fig)
        print(f"Saved: {path}")
    print("All 4 architecture diagrams saved.")
