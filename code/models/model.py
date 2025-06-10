# model.py
import numpy as np
from nn.conv import Conv2D, MaxPool2D
from nn.activation import ReLU, Sigmoid

class SimpleCNN:
    def __init__(self):
        # 1) conv1: in_channels=1, out_channels=3, kernel_size=3
        self.conv1 = Conv2D(in_channels=1, out_channels=3, kernel_size=3)
        self.sigmoid1 = Sigmoid()
        self.pool1 = MaxPool2D(kernel_size=2, stride=2)

        # 2) conv2: in_channels=3, out_channels=3, kernel_size=2
        self.conv2 = Conv2D(in_channels=3, out_channels=3, kernel_size=2)
        self.sigmoid2 = Sigmoid()

        # conv2의 출력 채널 개수(=3)를 그대로 클래스 개수로 사용
        # conv2의 출력 크기는 (batch, 3, 1, 1)이므로 reshape → (batch, 3)

    def forward(self, x):
        """
        순전파(Forward pass)
        x: numpy array of shape (batch, 1, 6, 6)
        반환: logits: numpy array of shape (batch, 3)
        """
        # 첫 번째 conv → Sigmoid → MaxPool
        out = self.conv1.forward(x)    # → (batch, 3, 4, 4)
        out = self.sigmoid1.forward(out)  # → (batch, 3, 4, 4)
        out = self.pool1.forward(out)  # → (batch, 3, 2, 2)

        # 두 번째 conv → Sigmoid
        out = self.conv2.forward(out)  # → (batch, 3, 1, 1)
        out = self.sigmoid2.forward(out)  # → (batch, 3, 1, 1)

        # 로짓(logit) 형태로 변환
        batch_size = out.shape[0]
        logits = out.reshape(batch_size, -1)  # → (batch, 3)
        return logits

    def backward(self, d_logits):
        """
        역전파(Backward pass)
        d_logits: (batch, 3) shape, 로짓에 대한 그라디언트
        """
        batch_size = d_logits.shape[0]

        # (batch,3) → (batch,3,1,1) 형태로 바꾼 뒤, Sigmoid2/backward → conv2/backward → pool1/backward → Sigmoid1/backward → conv1/backward
        d_out = d_logits.reshape(batch_size, 3, 1, 1)

        # conv2 backward
        d_out = self.sigmoid2.backward(d_out)   # → (batch, 3, 1, 1)
        d_out = self.conv2.backward(d_out)      # → (batch, 3, 2, 2)

        # pool1 backward
        d_out = self.pool1.backward(d_out)      # → (batch, 3, 4, 4)

        # sigmoid1 backward
        d_out = self.sigmoid1.backward(d_out)   # → (batch, 3, 4, 4)

        # conv1 backward
        _ = self.conv1.backward(d_out)          # → (batch, 1, 6, 6)
        # 최종 dX는 반환하지 않음

# models/model.py
class EnhancedCNN:
    """
    Conv1(1→4, ReLU) → Conv2(4→4, ReLU) → Conv3(4→3, ReLU) → Flatten
    풀링 없음 : 6→4→2→1
    """
    def __init__(self):
        self.c1 = Conv2D(1, 4, 3)
        self.a1 = ReLU()
        self.c2 = Conv2D(4, 4, 3)
        self.a2 = ReLU()
        self.c3 = Conv2D(4, 3, 2)
        self.a3 = ReLU()

    def forward(self, x):
        out = self.a1.forward(self.c1.forward(x))
        out = self.a2.forward(self.c2.forward(out))
        out = self.a3.forward(self.c3.forward(out))     # (N,3,1,1)
        return out.reshape(len(x), 3)                   # logits

    def backward(self, dlogits):
        d = dlogits.reshape(len(dlogits), 3, 1, 1)
        d = self.c3.backward(self.a3.backward(d))
        d = self.c2.backward(self.a2.backward(d))
        _ = self.c1.backward(self.a1.backward(d))       # dX는 사용 안 함
