import numpy as np

# 오차 제곱합(SSE: Sum of Squares for error)
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
t = [0,0,1,0,0,0,0,0,0,0]

def sum_of_squares_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)

print(sum_of_squares_error(np.array(y), np.array(t)))

# 교차 엔트로피 오차(CEE: Cross Entropy Error)
def cross_entropy_error(y, t):
    delta = 1e-7  # log(0) 방지
    return -np.sum(t * np.log(y + delta))

print(cross_entropy_error(np.array(y), np.array(t)))