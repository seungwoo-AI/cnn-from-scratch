# optimizer.py
# NumPy 기반 Optimizer 모듈: SGD, Momentum, NAG(인터페이스 일관화), Adagrad, RMSprop, Adam

import numpy as np

class SGD:
    def __init__(self, params, lr=0.01):
        """
        - params: [(param_array, grad_array), ...]
        - lr: 학습률 (learning rate)
        """
        self.params = params
        self.lr = lr

    def step(self):
        """
        각 파라미터에 대해 p = p - lr * grad 연산을 수행
        """
        for p, grad in self.params:
            p -= self.lr * grad

    def zero_grad(self):
        """
        각 gradient 배열을 0으로 초기화
        """
        for _, grad in self.params:
            grad.fill(0.0)

class Momentum:
    def __init__(self, params, lr=0.01, momentum=0.9):
        """
        - params: [(param_array, grad_array), ...]
        - lr: 학습률
        - momentum: 모멘텀 계수 (gamma)
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        # velocity 초기화: params마다 동일한 형태의 제로 배열 생성
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def step(self):
        """
        v_t = gamma * v_{t-1} + lr * grad
        p = p - v_t
        """
        for i, (p, grad) in enumerate(self.params):
            self.v[i] = self.momentum * self.v[i] + self.lr * grad
            p -= self.v[i]

    def zero_grad(self):
        for _, grad in self.params:
            grad.fill(0.0)

class NAG:
    def __init__(self, params, lr=0.01, momentum=0.9):
        """
        - params: [(param_array, grad_array), ...]
        - lr: 학습률
        - momentum: 모멘텀 계수 (gamma)

        이 구현은 lookahead 계산 대신 '일반 Nesterov 근사' 방식을 사용하여,
        SGD/NAG 인터페이스를 통일합니다.
        """
        self.params = params
        self.lr = lr
        self.momentum = momentum
        # velocity 초기화
        self.v = [np.zeros_like(p) for p, _ in self.params]

    def step(self):
        """
        Nesterov 근사 방식:
        p = p - gamma * v_prev - lr * grad
        v_t = gamma * v_prev + lr * grad
        """
        for i, (p, grad) in enumerate(self.params):
            # 이전 속도 보존
            v_prev = self.v[i].copy()
            # 현재 기울기로 속도 업데이트
            self.v[i] = self.momentum * v_prev + self.lr * grad
            # 파라미터를 'lookahead' 위치가 아닌 현재 기울기로 직접 업데이트
            p -= (self.momentum * v_prev + self.lr * grad)

    def zero_grad(self):
        for _, grad in self.params:
            grad.fill(0.0)

class Adagrad:
    def __init__(self, params, lr=0.01, eps=1e-8):
        """
        - params: [(param_array, grad_array), ...]
        - lr: 초기 학습률
        - eps: 작은 값 (0 나눗셈 방지용)
        """
        self.params = params
        self.lr = lr
        self.eps = eps
        # r_t: squared gradient 누적값
        self.r = [np.zeros_like(p) for p, _ in self.params]

    def step(self):
        """
        r_t = r_{t-1} + grad^2
        p = p - lr / (sqrt(r_t) + eps) * grad
        """
        for i, (p, grad) in enumerate(self.params):
            self.r[i] += grad * grad
            adjusted_lr = self.lr / (np.sqrt(self.r[i]) + self.eps)
            p -= adjusted_lr * grad

    def zero_grad(self):
        for _, grad in self.params:
            grad.fill(0.0)

class RMSprop:
    def __init__(self, params, lr=0.001, rho=0.9, eps=1e-8):
        """
        - params: [(param_array, grad_array), ...]
        - lr: 초기 학습률
        - rho: 지수이동평균 계수
        - eps: 작은 값 (0 나눗셈 방지)
        """
        self.params = params
        self.lr = lr
        self.rho = rho
        self.eps = eps
        # r_t: 지수이동평균된 squared gradient 값
        self.r = [np.zeros_like(p) for p, _ in self.params]

    def step(self):
        """
        r_t = rho * r_{t-1} + (1 - rho) * grad^2
        p = p - lr / (sqrt(r_t) + eps) * grad
        """
        for i, (p, grad) in enumerate(self.params):
            self.r[i] = self.rho * self.r[i] + (1 - self.rho) * (grad * grad)
            adjusted_lr = self.lr / (np.sqrt(self.r[i]) + self.eps)
            p -= adjusted_lr * grad

    def zero_grad(self):
        for _, grad in self.params:
            grad.fill(0.0)

class Adam:
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        """
        - params: [(param_array, grad_array), ...]
        - lr: 초기 학습률
        - beta1: 1차 모멘텀 계수
        - beta2: 2차 모멘텀 계수
        - eps: 작은 값 (0 나눗셈 방지)
        """
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        # 1차, 2차 모멘텀 초기값
        self.m = [np.zeros_like(p) for p, _ in self.params]
        self.v = [np.zeros_like(p) for p, _ in self.params]
        self.t = 0

    def step(self):
        """
        t += 1  
        m_t = beta1 * m_{t-1} + (1 - beta1) * grad  
        v_t = beta2 * v_{t-1} + (1 - beta2) * grad^2  
        m_hat = m_t / (1 - beta1^t)  
        v_hat = v_t / (1 - beta2^t)  
        p = p - lr * m_hat / (sqrt(v_hat) + eps)  
        """
        self.t += 1
        for i, (p, grad) in enumerate(self.params):
            # 1차 모멘텀
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grad
            # 2차 모멘텀
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grad * grad)
            # 편향 보정
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            # 파라미터 업데이트
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self):
        for _, grad in self.params:
            grad.fill(0.0)
