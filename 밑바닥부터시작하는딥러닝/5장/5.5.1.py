class Relu:
    def __init__(self):
        # relu 클래스는 mask라는 인스턴스 변수를 가진다.
        # => mask는 true/false로 구성된 넘파이 배열로, 순전파의 입력인 x의 원소 값이 0 이하일 때는 true,
        #    그 외(0보다 큰 원소)는 false로 유지한다.
        self.mask = None

    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out
    
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx
    
    