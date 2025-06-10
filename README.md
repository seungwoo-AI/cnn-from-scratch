# CNN‑from‑Scratch

> **A minimal NumPy‑only CNN for 6 × 6 digit recognition (1, 2, 3)** — built as the final project for *Numerical Optimization 2025*.
>
> 📄 *Full derivations, code listings, and discussion are in* `docs/Final Project 2025.pdf`.

---

## Why this repo might help you

* **Walk‑through code** — every layer, gradient, and update rule sits in a plain Python function you can single‑step in a debugger.
* **Framework‑free** — no PyTorch / TensorFlow; just the standard library + NumPy.
* **Self‑contained dataset** — 96 tiny 6 × 6 bitmaps live in one Excel sheet, so a full train‑eval run finishes in seconds  fileciteturn0file0.
* **Six optimizers + three activations** ready to mix & match: SGD · Momentum · NAG · RMSProp · Adam · AdamW + ReLU · Sigmoid · Softmax.
* **Readable scripts** — one command trains, another plots the learning curve.

---

## Quick start

```bash
# set up (conda, venv …)
python -m pip install -r requirements.txt  # numpy, pandas, matplotlib

# train SimpleCNN with Adam for 300 epochs
python code/train_adam.py --epochs 300 --lr 0.001

# visualise loss / accuracy
python code/plot_curve.py --log runs/adam-<timestamp>.log
```

*See `--help` on any `train_*.py` script for flags such as batch size, seed, or model variant.*

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

1. **util\_excel.py** loads the Excel sheet into `(N,1,6,6)` float32 images + integer labels.
2. **model.py** assembles either:

   * *SimpleCNN* → `Conv (3×3) → Sigmoid → MaxPool (2×2) → Conv → Sigmoid → Softmax`.
   * *EnhancedCNN* → 3× `Conv + ReLU`, no pooling.
3. **optimizer.py** updates parameters via the algorithm you choose.
4. **train\_\*.py** drives the loop: forward → loss → backward → update, for 300 epochs by default.
5. **plot\_curve.py** turns the run log into a PNG for your report.

Every forward / backward call returns explicit NumPy arrays so the math lines up with the equations in *Appendix A* of the report.

---

## For reviewers / instructors

* **Mathematical proof** — Appendix A shows back‑prop equations for Softmax‑CE, ReLU, Sigmoid, MaxPool, and Conv2D, each paired with the matching code line numbers.
* **Reproducibility** — `train_all.py --seed 0‑9` reproduces every experiment table in the report.
* **Extensibility** — new layers drop in by subclassing the simple `Layer` skeleton; see `nn/activation.py` for a template.

---

## Next steps (roadmap)

* Batch‑norm & dropout layers.
* Cython / Numba speed‑ups for the 5‑loop convolution.
* Port the loader to MNIST or CIFAR‑10 (just swap `util_excel.py`).

---

© 2025 Seung‑Woo Lee — MIT License
