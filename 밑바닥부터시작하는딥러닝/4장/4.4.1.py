"""
4.4.1 경사법 (경사 하강법)

- 기계학습 문제 대부분은 학습 단계에서 최적의 매개변수를 찾는 것.
- 신경망 역시 최적의 매개변수(가중치와 편향)를 학습 시에 찾아야 한다.
    - 최적이란, 손실 함수가 최솟값이 될 때의 매개변수 값이다.
- 이런 상황에서, 기울기를 잘 이용해 함수의 최솟값(또는 가능한 한 작은 값)을 찾으려는 것이 경사법이다.
- 실제로 복잡한 함수에서는 기울기가 가리키는 방향에 최솟값이 없는 경우가 대부분이다.
"""

def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradient(f, x)
        # 기울기에 학습률을 곱한 값으로 갱신하는 처리를 step_num번 반복
        x -= lr * grad

    return x


def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)