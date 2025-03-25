import numpy as np

# 넘파이 배열 생성하기
x = np.array([1.0, 2.0, 3.0])
print(x)

# 넘파이의 산술 연산
x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
print(x + y)
print(x - y)
print(x * y)
print(x / y)

# 넘파이의 N차원 배열
A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
print(A.dtype)

B = np.array([[3, 0], [0, 6]])
print(A + B)
print(A * B)

# 브로드캐스트
A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)

# 원소 접근
X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])
print(X[0][1])
