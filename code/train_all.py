# code/train_all.py  ─ 5-model 10-run 실험 스크립트
import numpy as np
import pandas as pd

from utils.util_excel import load_excel_dataset
from models.model        import SimpleCNN, EnhancedCNN
from nn.optimizer        import SGD, RMSprop, Adam
from nn.activation       import softmax, cross_entropy_loss, cross_entropy_grad


# ──────────────────────────────────────────────────────────────
def collect_params(model):
    """Conv2D 레이어의 (파라미터,grad) 튜플 자동 수집"""
    params = []
    for name in dir(model):
        layer = getattr(model, name)
        if hasattr(layer, "W"):
            params += [(layer.W, layer.dW), (layer.b, layer.db)]
    return params


def run_once(model_cls, opt_cls, opt_kwargs, X, y, num_epoch=600):
    """모델 한 번 학습   → (정확도, [epoch별 loss 리스트]) 반환"""
    model = model_cls()
    optimizer = opt_cls(collect_params(model), **opt_kwargs)

    losses = []
    for _ in range(num_epoch):
        logits = model.forward(X)
        probs  = softmax(logits)
        loss   = cross_entropy_loss(probs, y)
        losses.append(loss)

        model.backward(cross_entropy_grad(probs, y))
        optimizer.step()
        optimizer.zero_grad()

    acc = (logits.argmax(1) == y).mean()
    return acc, losses


# ──────────────────────────────────────────────────────────────
def main():
    # 데이터 전체(96개) 한 번에 학습 — 시드별 섞기 필요 없으므로 생략
    X, y = load_excel_dataset()
    # 실험 목록: (표시 이름, 모델 클래스, 옵티마이저, kwargs)
    exps = [
        ("S-SGD",   SimpleCNN,  SGD,     dict(lr=0.9)),
        ("S-RMS",   SimpleCNN,  RMSprop, dict(lr=1e-3, rho=0.9, eps=1e-8)),
        ("S-Adam",  SimpleCNN,  Adam,    dict(lr=1e-2, beta1=0.7, beta2=0.99, eps=1e-8)),
        ("E-RMS",   EnhancedCNN, RMSprop,dict(lr=1e-3, rho=0.9, eps=1e-8)),
        ("E-Adam",  EnhancedCNN, Adam,   dict(lr=1e-3, beta1=0.7, beta2=0.99, eps=1e-8)),
    ]

    # 결과 저장용
    acc_table   = {name: [] for name, *_ in exps}
    loss_record = {name: [] for name, *_ in exps}   # 각 run의 loss 리스트

    # 10회 반복
    for seed in range(10):
        np.random.seed(seed)
        for name, m_cls, o_cls, o_kw in exps:
            acc, losses = run_once(m_cls, o_cls, o_kw, X, y)
            acc_table[name].append(acc)
            loss_record[name].append(losses)

    # --------------------- 정확도 요약 표 ----------------------
    def row(name, lst):
        arr = np.asarray(lst)
        return [name] + [f"{v:.3f}" for v in arr] + [f"{arr.mean():.4f}", f"{arr.std():.4f}"]

    header = ["Model"] + [f"Run{i+1}" for i in range(10)] + ["Mean", "Std"]
    data   = [row(name, acc_table[name]) for name, *_ in exps]
    df = pd.DataFrame(data, columns=header)

    print("\n=== 10-run Accuracy Summary ===")
    print(df.to_string(index=False))

    # (선택) Loss CSV 저장
    pd.to_pickle(loss_record, "loss_record.pkl")   # 필요 시 주석 해제


if __name__ == "__main__":
    main()
