import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # Xavier 초기화
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(1. / (in_channels * kernel_size**2))
        self.biases = np.zeros(out_channels)

    def forward(self, x):
        self.input = x  # (batch, in_channels, H, W)
        batch_size, _, H, W = x.shape
        KH, KW = self.kernel_size, self.kernel_size
        OH, OW = H - KH + 1, W - KW + 1

        out = np.zeros((batch_size, self.out_channels, OH, OW))

        for b in range(batch_size):
            for oc in range(self.out_channels):
                for i in range(OH):
                    for j in range(OW):
                        region = x[b, :, i:i+KH, j:j+KW]  # (in_channels, KH, KW)
                        out[b, oc, i, j] = np.sum(region * self.weights[oc]) + self.biases[oc]
        return out

    def backward(self, dout, lr=0.01):
        # 간단한 버전 (추후 구현)
        pass

class ReLU:
    def forward(self, x):
        self.mask = (x > 0)
        return x * self.mask

    def backward(self, dout):
        return dout * self.mask


class Flatten:
    def forward(self, x):
        self.orig_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self.orig_shape)

class FullyConnected:
    def __init__(self, in_dim, out_dim):
        self.W = np.random.randn(in_dim, out_dim) * np.sqrt(2. / in_dim)
        self.b = np.zeros(out_dim)

    def forward(self, x):
        self.input = x
        return x @ self.W + self.b

    def backward(self, dout, lr=0.01):
        dW = self.input.T @ dout
        db = np.sum(dout, axis=0)
        dx = dout @ self.W.T

        # 업데이트
        self.W -= lr * dW
        self.b -= lr * db

        return dx

def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)  # 안정성
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy(pred, label):
    eps = 1e-9
    return -np.mean(np.sum(label * np.log(pred + eps), axis=1))

def cross_entropy_grad(pred, label):
    return (pred - label) / pred.shape[0]
