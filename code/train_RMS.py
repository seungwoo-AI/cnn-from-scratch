# code/train_rmsprop.py
"""
RMSpropCNN  (ReLU + Conv3 + RMSprop) 학습 스크립트
───────────────────────────────────────────────
* 모델      : EnhancedCNN  (Conv3×3 → Conv3×3 → Conv2×2, ReLU, 풀링 없음)
* 옵티마이저: RMSprop      (lr = 1e-3, ρ = 0.9, eps = 1e-8)
* 에폭      : 300
* 데이터    : 6×6 손글씨 (총 96개)  ─ train 80 %, valid 20 %
"""

import numpy as np
from utils.util_excel import load_excel_dataset
from models.model import EnhancedCNN  ,SimpleCNN        # ReLU + 3-Conv 스택
from nn.optimizer import RMSprop
from nn.activation import softmax, cross_entropy_loss, cross_entropy_grad


def collect_params(model):
    """Conv2D 레이어의 (파라미터, grad) 튜플을 자동으로 모은다."""
    params = []
    for name in dir(model):
        layer = getattr(model, name)
        if hasattr(layer, "W"):                # Conv2D 라면
            params += [(layer.W, layer.dW),
                       (layer.b, layer.db)]
    return params


def train(seed: int = 0, lr: float = 1e-3, rho: float = 0.9):
    # 1) 데이터 로드
    X, y = load_excel_dataset(sheet_name="Data")
    np.random.seed(seed)

    # 2) Train / Valid 분할 (80 % / 20 %)
    idx = np.random.permutation(len(X))
    split = int(len(X) * 0.8)
    X_tr, y_tr = X[idx[:split]], y[idx[:split]]
    X_va, y_va = X[idx[split:]], y[idx[split:]]

    print(f"Train set: {X_tr.shape}, {y_tr.shape}")
    print(f"Valid set: {X_va.shape}, {y_va.shape}")

    # 3) 모델 & 옵티마이저
    model = SimpleCNN()
    optimizer = RMSprop(collect_params(model),
                        lr=1e-3, rho=rho, eps=1e-8)

    # 4) 학습 루프
    for epoch in range(300):
        probs = softmax(model.forward(X_tr))
        loss  = cross_entropy_loss(probs, y_tr)

        model.backward(cross_entropy_grad(probs, y_tr))
        optimizer.step()
        optimizer.zero_grad()

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/300  Train Loss: {loss:.4f}")

    # 5) 정확도
    acc_tr = (model.forward(X_tr).argmax(1) == y_tr).mean()
    acc_va = (model.forward(X_va).argmax(1) == y_va).mean()
    print(f"Final Training accuracy: {acc_tr:.4f}")
    print(f"Validation accuracy:     {acc_va:.4f}")


if __name__ == "__main__":
    train()
