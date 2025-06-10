# code/train_sgd.py
import numpy as np
from utils.util_excel import load_excel_dataset
from models.model import SimpleCNN
from nn.optimizer import SGD           # 필요 시 Adam, RMSprop 등으로 교체
from nn.activation import softmax, cross_entropy_loss, cross_entropy_grad

def train():
    # 1) 데이터 로드
    X, Y = load_excel_dataset(path='data/handwriting_dataset.xlsx', sheet_name='Data')
    # X.shape == (96, 1, 6, 6), Y.shape == (96,)

    # 2) 섞고(train/valid 분리)
    num_samples = X.shape[0]           # 총 샘플 개수: 96
    indices = np.random.permutation(num_samples)
    split_index = int(num_samples * 0.8)  # 80% 지점 (96 * 0.8 = 76.8 → 76개 학습/20개 검증)

    train_idx = indices[:split_index]  # 앞 76개 인덱스
    valid_idx = indices[split_index:]  # 뒤 20개 인덱스

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_valid, Y_valid = X[valid_idx], Y[valid_idx]

    # (확인용) 분리된 데이터 크기 출력
    print(f"Train set: X_train={X_train.shape}, Y_train={Y_train.shape}")
    print(f"Valid set: X_valid={X_valid.shape}, Y_valid={Y_valid.shape}")

    # 3) 모델 생성
    model = SimpleCNN()

    # 4) Optimizer에 넘겨줄 (파라미터, 그라디언트) 쌍 리스트 준비
    params = [
        (model.conv1.W, model.conv1.dW),
        (model.conv1.b, model.conv1.db),
        (model.conv2.W, model.conv2.dW),
        (model.conv2.b, model.conv2.db)
    ]
    optimizer = SGD(params, lr=0.9)  # 학습률은 필요에 따라 조정

    # 5) 학습 루프
    num_epochs = 300
    for epoch in range(num_epochs):
        # 5.1) 순전파 (학습 세트)
        logits = model.forward(X_train)      # (76, 3)

        # 5.2) softmax → 확률 계산
        probs = softmax(logits)             # (76, 3)

        # 5.3) 손실 계산
        loss = cross_entropy_loss(probs, Y_train)

        # 5.4) cross-entropy 기울기 계산
        d_logits = cross_entropy_grad(probs, Y_train)

        # 5.5) 역전파
        model.backward(d_logits)

        # 5.6) 파라미터 업데이트 & 그라디언트 초기화
        optimizer.step()
        optimizer.zero_grad()

        # 5.7) 10 에포크마다 학습 Loss 출력
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss:.4f}")

    # 6) 학습이 끝난 뒤, Train/Valid 정확도 계산
    # 6.1) 학습 세트 정확도
    logits_train = model.forward(X_train)          # (76, 3)
    preds_train = np.argmax(logits_train, axis=1)  # (76,)
    train_acc = np.mean(preds_train == Y_train)

    # 6.2) 검증 세트 정확도
    logits_valid = model.forward(X_valid)          # (20, 3)
    preds_valid = np.argmax(logits_valid, axis=1)  # (20,)
    valid_acc = np.mean(preds_valid == Y_valid)

    print(f"Final Training accuracy: {train_acc:.4f}")
    print(f"Validation accuracy: {valid_acc:.4f}")

if __name__ == "__main__":
    train()
