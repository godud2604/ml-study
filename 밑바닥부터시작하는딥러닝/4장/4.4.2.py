"""
4.4.2 신경망에서의 기울기
"""

import sys, os
sys.path.append(os.pardir)
import numpy as np

# 형상이 2x3인 가중치 매개변수 하나를 인스턴스 변수로 갖는다.
# 메서드는 2개인데, 예측을 수행하는 predict, 손실 함수의 값을 구하는 loss
class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3) # 정규분포로 초기화
    
    def predict(self, x):
        return np.dot(x, self.W)
    
    # x: 입력 데이터 / t: 정답 레이블
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(x, y)

        return loss