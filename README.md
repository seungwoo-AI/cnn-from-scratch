# CNN‑from‑Scratch&#x20;

> **A minimal NumPy‑only CNN for 6 × 6 digit recognition (1 | 2 | 3)** — final project for *Numerical Optimization 2025*.
>
> 📄 *Full math derivations and extended discussion:* see `docs/Final Project 2025.pdf`.

---

## Why this repo might help you

* **Walk‑through code** – every layer, gradient, and update rule lives in a plain Python function you can step through in a debugger.
* **Framework‑free** – no PyTorch / TensorFlow; just the standard library + NumPy.
* **Self‑contained dataset** – 96 tiny 6 × 6 bitmaps bundled in one Excel sheet; a full train‑eval run finishes in seconds.
* **Six optimizers · three activations** ready to mix & match: *SGD · Momentum · NAG · RMSProp · Adam · AdamW*  ×  *ReLU · Sigmoid · Softmax*.
* **Readable CLI scripts** – one command trains, another plots the learning curve.

---

## Quick start (tested on Python 3.11 + NumPy 1.26)

```bash
# create an environment (conda / venv)
python -m pip install -r requirements.txt  # numpy pandas matplotlib

# train SimpleCNN with Adam for 300 epochs
python code/train_adam.py --epochs 300 --lr 0.001
# ➜ runs/adam-<timestamp>.log is created automatically

# visualise loss & accuracy
python code/plot_curve.py --log runs/adam-<timestamp>.log
```

*Use **`--help`** on any **`train_*.py`** script for flags such as batch size, seed, model variant.*

---

## Dataset

`data/handwriting_dataset.xlsx` contains **96 greyscale 6 × 6 bitmaps** and their labels.

| Split | # samples |
| ----- | --------- |
| Train | 72        |
| Val   | 12        |
| Test  | 12        |

`util_excel.py` converts each row into a `float32 (1, 6, 6)` tensor and the right‑most column into an integer label (1 | 2 | 3).

---

## Repository layout (one‑screen glance)

```text
code/              core source
 ├─ nn/            low‑level ops
 │   ├─ conv.py          # Conv2D forward / backward (5‑loop NumPy)
 │   ├─ activation.py    # ReLU · Sigmoid · Softmax‑CE
 │   └─ optimizer.py     # SGD ↔ AdamW — 6 algorithms
 ├─ models/        network builders
 │   └─ model.py        # SimpleCNN & EnhancedCNN
 ├─ utils/         helpers
 │   └─ util_excel.py   # Excel → (N,1,6,6) loader
 ├─ train_*.py     runnable experiments (one per optimiser)
 └─ plot_curve.py  matplotlib learning‑curve helper

data/handwriting_dataset.xlsx   96 labelled samples
LICENSE
README.md   (this file)
```

---

## How the pieces fit

1. **util\_excel.py** loads the Excel sheet → `(N, 1, 6, 6)` images + integer labels.
2. **model.py** assembles either:

   * *SimpleCNN* → `Conv (3×3) → Sigmoid → MaxPool (2×2) → Conv → Sigmoid → Softmax`.
   * *EnhancedCNN* → 3× `Conv + ReLU`, no pooling.
3. **optimizer.py** updates parameters via the chosen algorithm.
4. **train\_\*.py** : forward → loss → backward → update (default 300 epochs).
5. **plot\_curve.py** : turn the run log into a PNG for your report.

All tensors are explicit NumPy arrays – equations in *Appendix A* map 1‑to‑1 to code line numbers.

---

## For reviewers / instructors

* **Mathematical proof** – Appendix A shows back‑prop equations for Conv2D, MaxPool, ReLU, Sigmoid, Softmax‑CE with code references.
* **Reproducibility** – `train_all.py --seed 0‑9` regenerates every experiment table in the report.
* **Extensibility** – new layers drop in by subclassing the tiny `Layer` skeleton; see `nn/activation.py` for a template.
* **Code style** – PEP 8 compliant; formatted with `black` 24.3.

---

## Roadmap

* Batch‑norm & dropout layers.
* Cython / Numba speed‑ups for the 5‑loop convolution.
* Port the loader to MNIST or CIFAR‑10 (just swap `util_excel.py`).

---

© 2025 Seung‑Woo Lee — MIT License
