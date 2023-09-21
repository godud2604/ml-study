"""
- Sigmoid 계층의 계산 그래프 : 순전파의 출력 y만으로 역전파를 계산할 수 있다
"""

class Sigmoid:
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out

    def backward(self, dout):
        # 순전파의 출력을 인스턴스 out에 보관했다가, 역전파 계싼 때 그 값을 사용한다.
        dx = dout * (1.0 - self.out) * self.out

        return dx