"""
CPU-Only Cat/Dog Inference — No FPGA needed
Just run: python3 real_detect_inference.py --image ~/Downloads/test1dog.jpg --weights ~/Downloads/fpga_weights/
"""

import argparse
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np

CLASS_NAMES  = {0: "Cat", 1: "Dog"}
CLASS_COLORS = {0: (255, 160,  30),
                1: ( 50, 210,  80)}

ARCH = {
    "conv1": dict(out_ch=8,  in_ch=1,  k=3),
    "conv2": dict(out_ch=16, in_ch=8,  k=3),
    "conv3": dict(out_ch=32, in_ch=16, k=3),
}

# ── WEIGHT LOADING ────────────────────────────────────────────────────────────

def load_weight(weights_dir, key):
    """
    Try float32 version first, then int8, then auto-discover.
    """
    # Priority: float32 > int8 > auto-discover
    candidates = [
        f"{key}_float32.npy",
        f"{key}.npy",
    ]
    for fname in candidates:
        p = os.path.join(weights_dir, fname)
        if os.path.exists(p):
            return np.load(p).astype(np.float32)

    # Auto-discover by matching tokens
    all_files = [f for f in os.listdir(weights_dir) if f.endswith(".npy")]
    tokens = key.replace("_", " ").split()
    hits = [f for f in all_files if all(t in f.lower() for t in tokens)]
    # Prefer float32 hits
    float_hits = [h for h in hits if "float32" in h]
    chosen = float_hits[0] if float_hits else (hits[0] if hits else None)
    if chosen:
        p = os.path.join(weights_dir, chosen)
        return np.load(p).astype(np.float32)

    return None


# ── FORWARD PASS ──────────────────────────────────────────────────────────────

def _relu(x):
    return np.maximum(x, 0)

def _maxpool2(x):
    H, W = x.shape[-2], x.shape[-1]
    H2, W2 = H // 2, W // 2
    return x[..., :H2*2, :W2*2].reshape(*x.shape[:-2], H2, 2, W2, 2).max(axis=(-3, -1))

def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

def forward(img_bgr, weights_dir, verbose=True):
    def W(key):
        w = load_weight(weights_dir, key)
        if w is None and verbose:
            print(f"  [WARN] Weight not found: {key}")
        return w

    # Preprocess
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    x = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = x[np.newaxis, np.newaxis, :, :]   # (1, 1, 64, 64)

    t0 = time.time()

    # ── Conv layers ───────────────────────────────────────────────────────────
    for lk in ["conv1", "conv2", "conv3"]:
        a = ARCH[lk]
        out_ch, in_ch, k = a["out_ch"], a["in_ch"], a["k"]

        Wk = W(f"{lk}_w")
        Bk = W(f"{lk}_b")
        if Wk is None or Bk is None:
            print(f"  [ERROR] Missing weights for {lk} — cannot continue.")
            return None

        Wk = Wk.flatten()[:out_ch * in_ch * k * k].reshape(out_ch, in_ch, k, k)
        Bk = Bk.flatten()[:out_ch]

        pad = k // 2
        _, _, H, Wd = x.shape
        y  = np.zeros((1, out_ch, H, Wd), dtype=np.float32)
        xp = np.pad(x, ((0,0),(0,0),(pad,pad),(pad,pad)))

        for oc in range(out_ch):
            for ic in range(in_ch):
                for ki in range(k):
                    for kj in range(k):
                        y[:,oc,:,:] += Wk[oc,ic,ki,kj] * xp[:,ic,ki:ki+H,kj:kj+Wd]
            y[:,oc,:,:] += Bk[oc]

        x = _relu(_maxpool2(y))
        if verbose:
            print(f"  {lk}: {x.shape}  max={x.max():.4f}")

    # ── FC layer ──────────────────────────────────────────────────────────────
    Wf = W("fc_w")
    Bf = W("fc_b")
    if Wf is None or Bf is None:
        print("  [ERROR] Missing FC weights — cannot continue.")
        return None

    n_out  = int(Bf.flatten().shape[0])
    n_in   = len(Wf.flatten()) // n_out
    C      = x.shape[1]
    side   = int(round((n_in // C) ** 0.5))

    x_pool = np.zeros((1, C, side, side), dtype=np.float32)
    for c in range(C):
        x_pool[0, c] = cv2.resize(x[0, c], (side, side), interpolation=cv2.INTER_AREA)

    x_flat = x_pool.flatten()[:n_in]
    Wf2    = Wf.flatten()[:n_out * n_in].reshape(n_out, n_in)
    fc_out = _relu(Wf2 @ x_flat + Bf.flatten()[:n_out])

    if verbose:
        print(f"  fc:    {fc_out.shape}  max={fc_out.max():.4f}")

    # ── Output layer ──────────────────────────────────────────────────────────
    Wo = W("out_w")
    Bo = W("out_b")

    if Wo is not None and Bo is not None:
        Wo     = Wo.flatten()[:2 * n_out].reshape(2, n_out)
        logits = Wo @ fc_out + Bo.flatten()[:2]
        if verbose:
            print(f"  out:   logits = {logits}")
    else:
        # No out layer — use FC output. Take the two neurons with highest
        # variance as cat/dog proxy, then normalize via softmax.
        # More robust: treat FC as embedding, use last 2 values.
        logits = fc_out[-2:].copy()
        if verbose:
            print(f"  out:   (no out_w — using fc[-2:] as proxy) logits = {logits}")

    # Normalize scores to [0,1] via softmax so confidence is meaningful
    probs = softmax(logits)

    ms  = (time.time() - t0) * 1000
    cls = int(np.argmax(probs))

    return {
        "label":      CLASS_NAMES[cls],
        "cls":        cls,
        "score_cat":  float(probs[0]),
        "score_dog":  float(probs[1]),
        "margin":     float(abs(probs[1] - probs[0])),
        "latency_ms": ms,
    }


def benchmark(img_bgr, weights_dir, n_runs=5):
    print(f"\n  Running {n_runs} inference passes...")
    lats, last = [], None
    for i in range(n_runs):
        r = forward(img_bgr, weights_dir, verbose=(i == 0))
        if r is None:
            return None
        lats.append(r["latency_ms"])
        last = r

    last["best_ms"] = min(lats)
    last["avg_ms"]  = sum(lats) / len(lats)
    last["fps"]     = 1000.0 / min(lats)
    last["all_ms"]  = lats
    return last


# ── VISUALISE ─────────────────────────────────────────────────────────────────

def save_result_image(img_bgr, result, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    vis  = img_bgr.copy()
    H, W = vis.shape[:2]
    cls  = result["cls"]
    col  = CLASS_COLORS[cls]
    font = cv2.FONT_HERSHEY_SIMPLEX

    conf = max(result["score_cat"], result["score_dog"]) * 100

    # Border box
    cv2.rectangle(vis, (6, 6), (W-6, H-56), col, 3)

    # Label tag
    tag = f"{result['label']}  {conf:.1f}%"
    (tw, th), _ = cv2.getTextSize(tag, font, 0.8, 2)
    cv2.rectangle(vis, (6, 6), (6+tw+12, 6+th+16), col, -1)
    cv2.putText(vis, tag, (12, 6+th+6), font, 0.8, (0, 0, 0), 2)

    # Confidence bar
    bx, bw = 10, W - 20
    by = H - 66
    cat_w = int(bw * result["score_cat"])
    cv2.rectangle(vis, (bx, by), (bx+bw, by+14), (10,10,10), -1)
    cv2.rectangle(vis, (bx, by), (bx+cat_w, by+14), (30,150,255), -1)
    cv2.rectangle(vis, (bx+cat_w, by), (bx+bw, by+14), (40,200,70), -1)
    cv2.putText(vis, f"Cat {result['score_cat']*100:.1f}%", (bx+4, by+11), font, 0.38, (0,0,0), 1)
    cv2.putText(vis, f"Dog {result['score_dog']*100:.1f}%", (bx+cat_w+4, by+11), font, 0.38, (0,0,0), 1)

    # Metrics strip
    sy = H - 50
    cv2.rectangle(vis, (0, sy), (W, H), (15,15,25), -1)
    ms  = result["best_ms"]
    fps = result["fps"]
    cv2.putText(vis, f"CPU  {ms:.1f}ms  {fps:.1f}fps",
                (10, sy+20), font, 0.55, (120,255,160), 1)
    cv2.putText(vis,
                f"Cat={result['score_cat']:.4f}  Dog={result['score_dog']:.4f}  margin={result['margin']:.4f}",
                (10, sy+40), font, 0.42, (200,200,200), 1)

    img_rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
    col_hex = "#ffa020" if cls == 0 else "#32d050"

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#0a0a14")
    ax.imshow(img_rgb)
    ax.axis("off")
    ax.set_title(
        f"CPU Inference  →  {result['label']}  ({conf:.1f}% confidence)\n"
        f"Best: {ms:.1f}ms  |  {fps:.1f} FPS  |  Avg: {result['avg_ms']:.1f}ms",
        color=col_hex, fontsize=12, fontweight="bold", pad=10
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\n  Saved → {out_path}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="CPU-only cat/dog inference — no FPGA needed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--image",   required=True, help="Path to test image (.jpg/.png)")
    ap.add_argument("--weights", required=True, help="Folder containing .npy weight files")
    ap.add_argument("--runs",    type=int, default=5, help="Number of benchmark passes")
    ap.add_argument("--out",     default=None, help="Output image filename (auto if omitted)")
    ap.add_argument("--no-vis",  action="store_true", help="Skip saving result image")
    args = ap.parse_args()

    # Validate
    if not os.path.isfile(args.image):
        print(f"\n[ERROR] Image not found: {args.image}")
        sys.exit(1)
    if not os.path.isdir(args.weights):
        print(f"\n[ERROR] Weights folder not found: {args.weights}")
        sys.exit(1)

    img = cv2.imread(args.image)
    if img is None:
        print(f"[ERROR] Could not read image: {args.image}")
        sys.exit(1)

    npy_files = sorted(f for f in os.listdir(args.weights) if f.endswith(".npy"))
    print(f"\n  Image   : {args.image}  ({img.shape[1]}x{img.shape[0]})")
    print(f"  Weights : {args.weights}")
    print(f"  Files   : {npy_files}")

    result = benchmark(img, args.weights, n_runs=args.runs)
    if result is None:
        print("\n[FAILED] Check weight files above.")
        sys.exit(1)

    conf = max(result["score_cat"], result["score_dog"]) * 100
    print("\n" + "="*50)
    print(f"  Result     : {result['label']}")
    print(f"  Cat score  : {result['score_cat']:.4f}  ({result['score_cat']*100:.1f}%)")
    print(f"  Dog score  : {result['score_dog']:.4f}  ({result['score_dog']*100:.1f}%)")
    print(f"  Confidence : {conf:.1f}%")
    print(f"  Margin     : {result['margin']:.4f}")
    print(f"  Best       : {result['best_ms']:.2f} ms  ({result['fps']:.2f} FPS)")
    print(f"  Average    : {result['avg_ms']:.2f} ms")
    print(f"  All runs   : {[f'{v:.1f}' for v in result['all_ms']]} ms")
    print("="*50)

    if not args.no_vis:
        out = args.out or f"cpu_result_{os.path.basename(args.image)}"
        save_result_image(img, result, out)


if __name__ == "__main__":
    main()
