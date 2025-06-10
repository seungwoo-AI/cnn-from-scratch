# CNN-from-Scratch

Minimal **NumPy-only** convolutional neural network that classifies 6 × 6 grayscale digits (1 – 3). Built for the *Numerical Optimization 2025* final project and included in my graduate‑school portfolio as a demonstration of pure‑Python deep‑learning engineering.

---

## 1 Why this repo matters

* **Framework‑free** – every forward/backward pass is handwritten in NumPy.
* **Lightweight yet complete** – <1 kLOC but supports SGD, RMSprop, Adam, CLI training, and plotting utilities.
* **Reproducible** – `train_all.py` recreates every experiment reported in the accompanying PDF.
* **Didactic** – heavy in‑line comments, mathematically faithful code, and LaTeX blocks for future self‑review.

---

## 2 Quick Start

```bash
git clone https://github.com/<your-id>/cnn-from-scratch.git
cd cnn-from-scratch

python -m venv .venv && source .venv/bin/activate      # Windows ▶ .venv\Scripts\activate
pip install -r requirements.txt

# run SimpleCNN + Adam for 200 epochs
python code/train_sgd.py --epochs 200 --optimizer adam
```

Logs and checkpoints land in `runs/`.

---

## 3 Dataset

`data/handwriting_dataset.xlsx` stores 96 samples of 6 × 6 grayscale digits plus one‑hot labels.
`util_excel.py` converts it to

$$
X \in \mathbb{R}^{96\times1\times6\times6},\; y \in \{1,2,3\}^{96}
$$

ready for convolution.

---

## 4 Experiments & Results

| Model      | Activation | Pool          | Optimizer                            | Val Acc (mean ± SD, 10 seeds) |
| ---------- | ---------- | ------------- | ------------------------------------ | ----------------------------- |
| S‑SGD      | Sigmoid    | 2 × 2 MaxPool | SGD (η = 0.9)                        | 96.25 ± 0.51 %                |
| S‑RMS      | Sigmoid    | MaxPool       | RMSprop (η = 0.001, ρ = 0.9)         | 93.33 ± 2.80 %                |
| **S‑Adam** | Sigmoid    | MaxPool       | Adam (η = 0.01, β₁ = 0.7, β₂ = 0.99) | **99.06 ± 0.98 %**            |
| E‑RMS      | ReLU       | –             | RMSprop                              | 83.13 ± 16.67 %               |
| E‑Adam     | ReLU       | –             | Adam                                 | 82.71 ± 21.91 %               |

Recreate every line in the table with:

```bash
python code/train_all.py --plot
```

---

## 5 Model math in one glance

$$
\begin{aligned}
a &= x * W_1 + b_1 \\
h_1 &= \sigma(a) \\
\hat{y} &= \operatorname{softmax}\bigl(\operatorname{flatten}(h_1) W_2 + b_2\bigr) \\
\mathcal{L} &= -\sum y\,\log \hat{y}
\end{aligned}
$$

Back‑propagated error:

$$
\delta^{(l)} = \bigl(W^{(l+1)}\bigr)^{\!\top} \delta^{(l+1)} \odot \sigma'\bigl(z^{(l)}\bigr).
$$

These formulas map 1‑to‑1 to `model.py`.

---

## 6 Project layout

```text
cnn-from-scratch/
 ├─ code/
 │   ├─ model.py          # CNN layers & backprop
 │   ├─ optimizer.py      # SGD, RMSprop, Adam
 │   ├─ train_sgd.py      # CLI trainer
 │   ├─ train_all.py      # reproduce all results
 │   └─ visualize.py      # Matplotlib plots
 ├─ data/
 │   └─ handwriting_dataset.xlsx
 ├─ runs/                 # generated after training
 ├─ docs/
 │   └─ abstract.md       # 1‑page summary of report
 ├─ requirements.txt
 └─ README.md
```

---

## 7 Further reading

* **Project portfolio page** (screenshots & high‑level summary) → [https://github.com/seungwoo-AI/seungwoo-AI.github.io/tree/main/Projects/cnn-from-scratch](https://github.com/seungwoo-AI/seungwoo-AI.github.io/tree/main/Projects/cnn-from-scratch).
* **Full 9‑page project report (PDF)** with derivations and ablations → [https://github.com/seungwoo-AI/seungwoo-AI.github.io/blob/main/Projects/cnn-from-scratch/Final%20Project\_2025.pdf](https://github.com/seungwoo-AI/seungwoo-AI.github.io/blob/main/Projects/cnn-from-scratch/Final%20Project_2025.pdf).
* The training‑loss figure (page 4) and back‑prop equations (appendix A) supply additional context.

---

## 8 Roadmap

* Batch‑norm & dropout layers.
* Cython / Numba speed‑ups for the 5‑loop convolution.
* Port the loader to MNIST or CIFAR‑10 (just swap `util_excel.py`).

---

## 9 License & citation

MIT © 2025 Seung‑Woo Lee

> Lee, S.‑W. (2025). *CNN‑from‑Scratch: A Minimal NumPy Implementation for 6 × 6 Digit Classification.* GitHub. <[https://github.com/](https://github.com/)/cnn-from-scratch>
