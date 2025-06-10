# CNN‑from‑Scratch 🧑‍💻📐

Pure‑Python (NumPy‑only) Convolutional Neural Network that classifies **6 × 6 handwritten digits (classes : 1 , 2 , 3)** — implemented as the final project for the *Numerical Optimization* course.

This repository is meant to be **educational first**: every layer, gradient, and update rule is written from scratch so you can step through the code and *really* see what happens under the hood.

## ✨ Key Points

* **Zero deep‑learning frameworks** – only the Python standard library, NumPy, Pandas, and Matplotlib.
* **Multiple optimizers** – SGD, RMSProp, Adam — switchable from the CLI.
* **Tiny custom dataset** – an Excel file with 6×6 bitmaps that lets the model train in seconds on a laptop CPU.
* **Clean experiment scripts** – one‑liner training commands plus an automatic learning‑curve visualizer.
* **Readable architecture** – <100 LOC for the CNN itself; everything is broken into small, testable pieces.

## 🗂️ Directory Layout

```text
.
├── code
│   ├── models
│   │   └── model.py            # 1‑Conv‑layer + 2‑FC CNN
│   ├── nn
│   │   ├── conv.py             # forward / backward of 2‑D convolution
│   │   ├── activation.py       # ReLU, Softmax
│   │   └── optimizer.py        # SGD, RMSProp, Adam (NumPy)
│   ├── utils
│   │   └── util_excel.py       # Excel ↔ NumPy converter
│   ├── train_sgd.py            # train with vanilla SGD
│   ├── train_RMS.py            # train with RMSProp
│   ├── train_adam.py           # train with Adam
│   ├── train_all.py            # run a sweep over hyper‑params
│   └── plot_curve.py           # loss / accuracy curves
├── data
│   └── handwriting_dataset.xlsx # 6×6 digit bitmaps & labels
├── LICENSE
└── README.md   ← you are here
```

## 🚀 Quick Start

```bash
# 1. Create an environment (conda or venv)
pip install -r requirements.txt  # only numpy, pandas, matplotlib

# 2. Train with your favourite optimizer
python code/train_adam.py --epochs 100 --lr 0.001

# 3. Plot learning curves
python code/plot_curve.py --log runs/adam‑2025‑06‑11‑12‑34.log
```

## 🧮 Math Primer

The convolution used in `conv.py` is the discrete cross‑correlation commonly employed in CNNs:

$$
y_{i,j,k}^{(l)} \;=\;
\\sigma\\Bigl(\\;\\sum_{c=1}^{C_{\\text{in}}}
\\bigl(W_{k}^{(l)} * x\\bigr)_{i,j,c}
\;+\; b_{k}^{(l)}\\Bigr)\\!,
$$

where $*$ denotes the 2‑D correlation operator and $\\sigma$ is ReLU.
The network is trained by minimising the cross‑entropy

$$
\\mathcal{L}
= -\\frac{1}{N}\\sum_{n=1}^{N}\\sum_{c=1}^{3}
 y_{n,c}\\,\\log \\hat{y}_{n,c}.
$$

## 📊 Sample Results

| Optimizer | Final Accuracy (val) | Epochs |
| --------- | -------------------- | ------ |
| SGD       | 91 %                 | 200    |
| RMSProp   | 94 %                 | 150    |
| Adam      | **97 %**             | 120    |

*(On the provided dataset; see `runs/` for complete logs.)*

## 📝 Dataset

`handwriting_dataset.xlsx` contains *N = 1 800* grayscale 6×6 bitmaps laid out row‑wise plus the ground‑truth label in the last column.
The helper `util_excel.py` converts the sheet to a `(N, 6, 6)` NumPy array on‑the‑fly.

## 🔬 How It Works – File‑by‑File

| File            | Purpose                                                                 |
| --------------- | ----------------------------------------------------------------------- |
| `conv.py`       | Implements im2col + GEMM convolution and its backward pass              |
| `activation.py` | ReLU and numerically stable Softmax                                     |
| `optimizer.py`  | Plain NumPy implementations of SGD, RMSProp, Adam                       |
| `model.py`      | Composes `Conv2D → ReLU → FC → Softmax`                                 |
| `train_*.py`    | Training loops with CLI flags (epochs, lr, batch size)                  |
| `plot_curve.py` | Reads the `.log` files produced during training and plots loss/accuracy |

Feel free to dive into any of them—each function is thoroughly doc‑commented.

## 🛠️ Extending the Project

* Add more layers (e.g. MaxPool, Dropout).
* Swap the Excel dataset for **MNIST** (just change the loader).
* Implement **momentum** or **Nesterov** in `optimizer.py`.
* Port the training loop to **JAX** or **PyTorch** for comparison.

## 🤝 Contributing

Issues and pull requests are welcome! Fork the repo, create a feature branch, and open a PR.

## 📄 License

This project is released under the MIT License – see [LICENSE](LICENSE) for details.
