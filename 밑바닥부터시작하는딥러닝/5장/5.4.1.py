"""
5.4 단순한 계층 구현하기

5.4.1 곱셈 계층
"""

class MulLayer:
    def __init__(self):
        # 인스턴스 변수인 x와 y를 초기화한다.
        # => 두 변수는 순전파 시의 입력 값을 유지하기 위해서 사용한다.
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y

        return out
    
    def backward(self, dout):
        dx = dout * self.y 
        dy = dout * self.x

        return dx, dy
    

apple = 100
apple_num = 2
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)

print(price) # 220

# 역전파
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(apple, apple_num)
dapple_price, dapple_num = mul_apple_layer.backward(dapple_price)
