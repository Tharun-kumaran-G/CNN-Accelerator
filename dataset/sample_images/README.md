# Dataset — Sample Images

This folder contains sample images used for training, validation, and testing the Person/Object CNN classifier.

---

## Folder Structure

```
dataset/sample_images/
├── dog/
│   ├── dog_001.jpg
│   ├── dog_002.jpg
│   └── ...
├── cat/
│   ├── cat_001.jpg
│   ├── cat_002.jpg
│   └── ...
└── test/
    ├── test_001.jpg
    └── ...
```

---

## Dataset Details

| Property        | Value                     |
|-----------------|---------------------------|
| Input Size      | 64 × 64 pixels            |
| Color Space     | Grayscale                 |
| Classes         | 2 (Person, Object)        |
| Format          | JPEG / PNG                |
| Preprocessing   | CLAHE + normalization     |

---

## Preprocessing Pipeline

Before feeding images into the CNN, the following steps are applied on the ARM Cortex-A9 (PS side):

1. **Resize** — scale to 64×64
2. **Grayscale conversion** — if source is RGB
3. **CLAHE** — Contrast Limited Adaptive Histogram Equalization for better edge visibility
4. **Normalization** — pixel values scaled to INT8 range
5. **Foreground extraction** — adaptive thresholding + blob detection

---

## How to Add Your Own Images

Place images into the appropriate class folder (`dog/` or `cat/`), then re-run the training script in `models/cnn_model/` to retrain with the new data.

```bash
cd models/cnn_model/
python train.py --data ../../dataset/sample_images
```

---

## Notes

- Images shown here are representative samples only — not the full training set
- For best results, ensure your test images have a similar framing and background contrast to training images
- Avoid heavily blurred or extremely dark images — CLAHE helps but has limits
