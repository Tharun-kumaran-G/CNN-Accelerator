# 🐾 FPGA Cat/Dog Classifier — PYNQ Run Guide

> **Board IP:** `192.168.2.99` &nbsp;|&nbsp; **Platform:** Zynq-7020 on PYNQ  
> **Tool:** Vitis HLS + Vivado 2023.1 &nbsp;|&nbsp; **Interface:** JupyterLab

---

## 📋 Table of Contents

1. [Prerequisites & File Setup](#1-prerequisites--file-setup)
2. [Connect to the Board](#2-connect-to-the-board)
3. [Upload Files via JupyterLab](#3-upload-files-via-jupyterlab)
4. [Open & Run the Notebook](#4-open--run-the-notebook)
5. [Step-by-Step Code Walkthrough](#5-step-by-step-code-walkthrough)
6. [Running Inference](#6-running-inference)
7. [FPGA vs CPU Speed — Explained](#7-fpga-vs-cpu-speed--explained)
8. [Reading the Output Figure](#8-reading-the-output-figure)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Prerequisites & File Setup

### What you need on the board

Before running, make sure the following files exist on the PYNQ board at the correct paths:

```
/home/xilinx/pynq/overlays/real_detect/
├── real_detect.bit          ← FPGA bitstream
├── real_detect.hwh          ← Hardware handoff file
├── test8cat.jpg             ← Test image (or your own)
└── fpga_weights_for_pynq/
    └── fpga_weights/
        ├── conv1_w.npy
        ├── conv1_b.npy
        ├── conv2_w.npy
        ├── conv2_b.npy
        ├── conv3_w.npy
        ├── conv3_b.npy
        ├── fc_w.npy
        └── fc_b.npy

/home/xilinx/multi_weight_results/   ← auto-created by script
```

### Required Python packages (pre-installed on PYNQ)

| Package | Purpose |
|---|---|
| `pynq` | FPGA overlay loading & DMA buffers |
| `numpy` | Weight loading & CPU inference |
| `opencv-python` (`cv2`) | Image processing & visualization |
| `matplotlib` | Output figure generation |

> ✅ All packages above come pre-installed on **PYNQ image v2.7+**. No `pip install` needed.

---

## 2. Connect to the Board

### Step 1 — Connect via Ethernet

Plug an Ethernet cable between your PC and the PYNQ board.

Set your PC's Ethernet adapter to a **static IP** in the same subnet:

| Field | Value |
|---|---|
| IP Address | `192.168.2.1` |
| Subnet Mask | `255.255.255.0` |
| Gateway | `192.168.2.99` |

### Step 2 — Open JupyterLab in your browser

```
http://192.168.2.99
```

When prompted for a password, enter:

```
xilinx
```

> 💡 If the page doesn't load, verify the board is powered on and the link LED on the Ethernet port is lit.

---

## 3. Upload Files via JupyterLab

### Upload the inference script

1. In JupyterLab, navigate to `/home/xilinx/pynq/overlays/real_detect/`
2. Click the **Upload** button (↑ arrow icon) in the file browser panel
3. Select and upload:
   - `real_detect.bit`
   - `real_detect.hwh`
   - Your test image (e.g. `test8cat.jpg`)
   - The `fpga_weights/` folder contents

### Upload weight `.npy` files

Navigate into `fpga_weights_for_pynq/fpga_weights/` and upload all `.npy` weight files there.

> ⚠️ The `.bit` and `.hwh` files **must share the same filename stem** (`real_detect`) and be in the same folder, otherwise PYNQ cannot load the overlay.

---

## 4. Open & Run the Notebook

### Step 1 — Create a new notebook

In JupyterLab, click **File → New → Notebook** and select kernel **Python 3**.

### Step 2 — Paste the inference code

Copy the entire contents of `fpga_cat_dog_classifier.py` and paste it into the first cell of your notebook.

### Step 3 — Change the test image path (if needed)

Find this line near the top of the script and update it to point to your image:

```python
TEST_IMAGE = f"{BASE}/real_detect/test8cat.jpg"   # ← change image here
```

For example, to use a dog image:
```python
TEST_IMAGE = f"{BASE}/real_detect/my_dog.jpg"
```

### Step 4 — Run the cell

Press **`Shift + Enter`** or click the ▶ Run button.

Expected console output:
```
Overlay loaded
Image: test8cat.jpg  640×480
───────────────────────────────────────────────
  [FPGA  Weights  A]
    FPGA → Cat  Cat=482M  Dog=21M  margin=461M  12.3ms  81.3fps
    Running CPU inference (NumPy)...
    CPU  → Cat  284ms  3.5fps  [MATCH ✓]
    Speedup: FPGA is 23.1× faster than CPU

  Saved → /home/xilinx/multi_weight_results/combined_test8cat.jpg
Done.
```

---

## 5. Step-by-Step Code Walkthrough

Here is what each major section of the script does:

### 5.1 — Overlay Loading

```python
ol = Overlay(f"{BASE}/real_detect/real_detect.bit")
ip = ol.real_detector_0
```

Loads the FPGA bitstream onto the PL fabric and gets a handle to the `real_detector_0` IP block. This programs the Zynq PL in ~2–4 seconds.

---

### 5.2 — DMA Buffer Allocation

```python
img_buf     = allocate(shape=(4096,),    dtype=np.uint8)
conv1_w_buf = allocate(shape=(200_000,), dtype=np.int8)
# ... etc
```

Allocates contiguous physical memory buffers in DDR that both the ARM CPU and the FPGA accelerator can access directly — this is zero-copy DMA. No data is copied between separate memory regions.

---

### 5.3 — Register Address Writing

```python
REG = {
    "ctrl":    0x00,
    "image":   (0x10, 0x14),
    "conv1_w": (0x1C, 0x20),
    ...
}
```

These are the AXI-Lite control register offsets of the HLS IP. The script writes the **physical address** of each DMA buffer into the corresponding register pair, so the accelerator knows where to find the image and weights in DDR.

---

### 5.4 — Weight Loading

```python
load_weight_set("FPGA  Weights  A")
```

Loads quantized **INT8 weights** from `.npy` files on disk into the pre-allocated DMA buffers, then flushes the CPU cache so the FPGA sees the updated values.

---

### 5.5 — FPGA Inference

```python
ip.write(REG["ctrl"], 0x01)      # Start the accelerator
while (ip.read(REG["ctrl"]) & 0x02) == 0:
    time.sleep(0.0005)           # Poll ap_done bit
res_buf.invalidate()             # Invalidate CPU cache to read fresh results
cls = int(res_buf[0])            # 0 = Cat, 1 = Dog
```

The full inference pipeline (resize → conv1 → conv2 → conv3 → FC → output) runs **entirely on the FPGA fabric** while the ARM CPU simply waits on the done flag.

---

### 5.6 — CPU Inference (NumPy)

```python
cpu_forward(img_bgr, ws_name)
```

Runs the **same neural network** in pure NumPy on the ARM Cortex-A9 CPU for comparison. Uses float32 arithmetic. This is intentionally unoptimized (no NEON, no quantization) to show the raw CPU baseline.

---

## 6. Running Inference

### Change the input image

Edit the `TEST_IMAGE` variable at the top:

```python
TEST_IMAGE = "/home/xilinx/pynq/overlays/real_detect/your_image.jpg"
```

Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`

### Run on multiple images in a loop

Add this cell to your notebook to batch-process a folder:

```python
import glob

images = glob.glob("/home/xilinx/pynq/overlays/real_detect/*.jpg")
for img_path in images:
    TEST_IMAGE = img_path
    run()
```

### Adjust confidence display

If you want to see raw scores in the console, add after the `fpga_classify()` call:

```python
print(f"Raw scores — Cat: {score_cat}, Dog: {score_dog}, Margin: {margin}")
```

### Re-run without reloading the overlay

The overlay only needs to be loaded **once per session**. If you just want to re-run inference on a different image, re-run only the `run()` cell — do **not** re-run the `Overlay(...)` cell.

---

## 7. FPGA vs CPU Speed — Explained

### Why is the FPGA so much faster?

| Factor | CPU (ARM Cortex-A9) | FPGA (Zynq PL) |
|---|---|---|
| **Architecture** | Sequential, general-purpose | Massively parallel, custom pipeline |
| **Precision** | Float32 (32-bit) | INT8 quantized (8-bit) |
| **Clock speed** | 667 MHz (1 core) | 50 MHz fabric clock |
| **Parallelism** | 1 MAC at a time | All conv filters computed simultaneously |
| **Memory access** | Through L1/L2 cache | Direct DMA from DDR via HP0 port |
| **Typical latency** | ~250–350 ms | ~10–25 ms |
| **Typical FPS** | ~3–5 fps | ~50–100 fps |
| **Speedup** | 1× (baseline) | **~20–30× faster** |

### How the speedup is achieved

```
CPU path (sequential):
  Image → Conv1 (8 filters, one at a time)
        → Conv2 (16 filters, one at a time)
        → Conv3 (32 filters, one at a time)
        → FC layer
  Total: ~280 ms

FPGA path (parallel pipeline):
  Image ──┬─► Filter 0 ─┐
          ├─► Filter 1 ─┤
          ├─► Filter 2 ─┤ All 8 filters of Conv1
          ├─► ...       ─┤ computed at the SAME time
          └─► Filter 7 ─┘
          → Conv2 (16 parallel) → Conv3 (32 parallel) → FC
  Total: ~12 ms
```

### The two key FPGA advantages

**1. INT8 Quantization** — Instead of 32-bit floats, the FPGA uses 8-bit integers. This means:
- 4× more weights fit in the same memory bandwidth
- Integer arithmetic is ~4× faster than floating point on reconfigurable logic
- Minimal accuracy loss (typically < 1–2%)

**2. Spatial parallelism** — On a CPU, filters are applied one by one. On the FPGA, all filters in a layer have their own dedicated hardware multipliers running simultaneously. With 50 MHz clock × parallel MACs, total throughput far exceeds what a 667 MHz CPU can achieve sequentially.

### Realistic benchmark expectations

| Metric | Typical value |
|---|---|
| FPGA inference latency | 10 – 25 ms |
| FPGA throughput | 50 – 100 FPS |
| CPU inference latency | 250 – 400 ms |
| CPU throughput | 2 – 5 FPS |
| Speedup ratio | **15× – 30×** |
| Power (FPGA) | ~2.5 W |
| Power (CPU only) | ~1.5 W |
| Performance/Watt | FPGA wins by ~10× |

> 📌 The exact speedup depends on image size, weight quantization level, and whether the CPU uses NEON SIMD intrinsics. The NumPy baseline in this script represents worst-case CPU performance.

---

## 8. Reading the Output Figure

The saved figure at `/home/xilinx/multi_weight_results/combined_<image>.jpg` contains:

```
┌─────────────────────────────────────────┐
│  Title bar: filename | label | FPGA fps │
│             CPU result | speedup        │
├─────────────────────────────────────────┤
│  Detection panel with:                  │
│   • Tight bounding box around animal    │
│   • Label tag (Cat / Dog)               │
│   • Cat% ████░░░░ Dog% confidence bar   │
│   • FPGA row: ms | fps | conf | margin  │
│   • CPU  row: ms | fps | match/differ   │
│   • Dual latency bar (blue=FPGA,        │
│                        green=CPU)       │
├─────────────────────────────────────────┤
│  Conv1 feature maps  (8 filters)        │
├─────────────────────────────────────────┤
│  Conv2 feature maps  (16 filters)       │
├─────────────────────────────────────────┤
│  Conv3 feature maps  (32 filters)       │
└─────────────────────────────────────────┘
```

**Color coding:**
- 🟠 Orange box / label = **Cat** prediction
- 🟢 Green box / label = **Dog** prediction
- 🔵 Blue latency bar = FPGA inference time
- 🟢 Green latency bar = CPU inference time
- ✓ MATCH = FPGA and CPU agree on the class
- ✗ DIFFER = FPGA and CPU disagree (check quantization error)

---

## 9. Troubleshooting

### ❌ `FileNotFoundError: real_detect.bit`
Verify the bitstream is at exactly:
```
/home/xilinx/pynq/overlays/real_detect/real_detect.bit
```
And that `real_detect.hwh` is in the same folder.

---

### ❌ `TimeoutError: FPGA timeout`
The accelerator didn't assert `ap_done` within 5 seconds. Causes:
- Weights not loaded correctly → re-run `load_weight_set()`
- Buffer addresses not written → re-run `_write_all_addrs()`
- Bitstream mismatch with IP → regenerate bitstream from correct `.xpr`

---

### ❌ `cv2.imread()` returns `None`
The test image path is wrong or the file wasn't uploaded. Check:
```python
import os
print(os.path.exists(TEST_IMAGE))   # Should print True
```

---

### ❌ CPU inference returns `None`
The float32 or int8 `.npy` weight files aren't found in the `fpga_weights/` directory. Verify all 8 `.npy` files are uploaded:
```python
import os
wdir = "/home/xilinx/pynq/overlays/real_detect/fpga_weights_for_pynq/fpga_weights/"
print(os.listdir(wdir))
```

---

### ❌ Wrong classification result
- Try a cleaner, well-lit image with a single animal.
- Check that weights were trained on cats/dogs (not person/object)
- Verify the image is in RGB/BGR color (not already grayscale)

---

### 💡 Tip — Re-run just inference without reloading overlay

```python
# In a new cell — fast re-run:
load_weight_set("FPGA  Weights  A")
run()
```

No need to re-execute the overlay loading or buffer allocation cells.

---

*Guide written for PYNQ v2.7+ on Zynq-7020. Board IP: `192.168.2.99`.*
