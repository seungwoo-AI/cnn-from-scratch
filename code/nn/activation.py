# code/nn/activation.py
# NumPy 기반 활성화 함수 및 손실 함수 모듈

import numpy as np

class ReLU:
    def __init__(self):
        # forward 시 x > 0인 위치를 저장하기 위한 마스크
        self.mask = None

    def forward(self, x):
        """
        x: numpy array (arbitrary shape)
        반환: ReLU(x) = max(0, x)
        """
        self.mask = (x > 0).astype(np.float32)
        return x * self.mask

    def backward(self, d_out):
        """
        d_out: 상위 계층에서 내려온 gradient (same shape as x)
        반환: ReLU backward gradient
        """
        return d_out * self.mask

class Sigmoid:
    def __init__(self):
        # forward 시 출력값을 저장하여, backward에서 미분 시 사용
        self.out = None

    def forward(self, x):
        """
        x: numpy array (arbitrary shape)
        반환: Sigmoid(x) = 1 / (1 + exp(-x))
        """
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, d_out):
        """
        d_out: 상위 계층에서 내려온 gradient (same shape as x)
        반환: Sigmoid backward gradient = d_out * sigmoid(x) * (1 - sigmoid(x))
        """
        return d_out * (self.out * (1 - self.out))

def softmax(x):
    """
    x: numpy array, shape = (batch, num_classes)
    반환: 확률 분포, shape = (batch, num_classes)
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    exps = np.exp(shifted_logits)
    return exps / np.sum(exps, axis=1, keepdims=True)

def cross_entropy_loss(probs, labels):
    """
    probs: numpy array, shape = (batch, num_classes)
    labels: numpy array, shape = (batch,), 값은 정수 인덱스
    반환: 평균 손실 (scalar)
    """
    batch_size = probs.shape[0]
    correct_logprobs = -np.log(probs[np.arange(batch_size), labels] + 1e-12)
    loss = np.sum(correct_logprobs) / batch_size
    return loss

def cross_entropy_grad(probs, labels):
    """
    probs: numpy array, shape = (batch, num_classes)
    labels: numpy array, shape = (batch,), 값은 정수 인덱스
    반환: logits에 대한 gradient, shape = (batch, num_classes)
    """
    batch_size = probs.shape[0]
    d_logits = probs.copy()
    d_logits[np.arange(batch_size), labels] -= 1
    return d_logits / batch_size
