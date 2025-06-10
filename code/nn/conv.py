# nn/conv.py
import numpy as np

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size):
        """
        - in_channels: 입력 채널 수 (흑백이미지 → 1)
        - out_channels: 출력 채널(필터) 수
        - kernel_size: 정사각형 커널 크기 (예: 3 → 3×3)
        """
        self.in_c = in_channels
        self.out_c = out_channels
        self.k = kernel_size

        # Xavier 초기화
        limit = np.sqrt(1.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * limit
        self.b = np.zeros(out_channels, dtype=np.float32)

        # <— Gradient를 반드시 배열로 미리 초기화합니다.
        self.dW = np.zeros_like(self.W, dtype=np.float32)
        self.db = np.zeros_like(self.b, dtype=np.float32)

        # backward 시 input을 저장하기 위해서만 사용
        self.x = None

    def forward(self, x):
        """
        x: numpy array of shape (batch, in_c, h_in, w_in)
        반환: out of shape (batch, out_c, h_out, w_out)
        """
        self.x = x  # backward에서 사용할 수 있도록 저장
        batch, _, h_in, w_in = x.shape
        h_out = h_in - self.k + 1
        w_out = w_in - self.k + 1

        out = np.zeros((batch, self.out_c, h_out, w_out), dtype=np.float32)

        for n in range(batch):
            for oc in range(self.out_c):
                for ic in range(self.in_c):
                    for i in range(h_out):
                        for j in range(w_out):
                            window = x[n, ic, i : i + self.k, j : j + self.k]  # (k, k)
                            out[n, oc, i, j] += np.sum(window * self.W[oc, ic])
                out[n, oc, :, :] += self.b[oc]

        return out

    def backward(self, d_out):
        """
        d_out: 상위 계층(또는 손실함수)으로부터 내려오는 gradient,
               shape = (batch, out_c, h_out, w_out)
        반환: dx (shape = (batch, in_c, h_in, w_in))
        """
        x = self.x
        batch, _, h_in, w_in = x.shape
        _, _, h_out, w_out = d_out.shape

        # gradient 초기화 (매 backward 호출마다 새로 채워야 함)
        # 이전 에포크에 남아 있던 기울기를 지우고 다시 계산하기 위해서
        self.dW.fill(0.0)
        self.db.fill(0.0)
        dx = np.zeros_like(x, dtype=np.float32)

        for n in range(batch):
            for oc in range(self.out_c):
                for i in range(h_out):
                    for j in range(w_out):
                        grad_val = d_out[n, oc, i, j]  # 스칼라
                        self.db[oc] += grad_val

                        for ic in range(self.in_c):
                            window = x[n, ic, i : i + self.k, j : j + self.k]  # (k, k)
                            self.dW[oc, ic] += grad_val * window
                            dx[n, ic, i : i + self.k, j : j + self.k] += grad_val * self.W[oc, ic]

        return dx

class MaxPool2D:
    def __init__(self, kernel_size, stride):
        """
        kernel_size: 풀링 크기 (예: 2 → 2×2)
        stride: 풀링 보폭 (예: 2)
        """
        self.k = kernel_size
        self.s = stride

        # backward를 위해 저장
        self.x = None
        self.max_idx = None

    def forward(self, x):
        """
        x: numpy array of shape (batch, c, h, w)
        반환: out of shape (batch, c, h_out, w_out)
        """
        self.x = x
        batch, c, h, w = x.shape
        h_out = h // self.k
        w_out = w // self.k

        out = np.zeros((batch, c, h_out, w_out), dtype=np.float32)
        self.max_idx = {}

        for n in range(batch):
            for ch in range(c):
                for i in range(h_out):
                    for j in range(w_out):
                        h_start = i * self.s
                        w_start = j * self.s
                        window = x[n, ch, h_start : h_start + self.k, w_start : w_start + self.k]
                        m = np.max(window)
                        out[n, ch, i, j] = m

                        flat_index = np.argmax(window)
                        mi, mj = np.unravel_index(flat_index, (self.k, self.k))
                        self.max_idx[(n, ch, i, j)] = (mi, mj)

        return out

    def backward(self, d_out):
        """
        d_out: 상위 계층(또는 손실함수)으로부터 내려오는 gradient,
               shape = (batch, c, h_out, w_out)
        반환: dx (shape = (batch, c, h, w))
        """
        x = self.x
        batch, c, h, w = x.shape
        h_out = h // self.k
        w_out = w // self.k

        dx = np.zeros_like(x, dtype=np.float32)

        for n in range(batch):
            for ch in range(c):
                for i in range(h_out):
                    for j in range(w_out):
                        mi, mj = self.max_idx[(n, ch, i, j)]
                        h_start = i * self.s
                        w_start = j * self.s
                        dx[n, ch, h_start + mi, w_start + mj] = d_out[n, ch, i, j]

        return dx
