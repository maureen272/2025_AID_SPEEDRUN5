import numpy as np

def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))

X = np.array([1.0, 0.5])
W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
B1 = np.array([0.1, 0.2, 0.3])

print(W1.shape) # (2, 3)
print(X.shape) # (2,)
print(B1.shape) # (3,)

A1 = np.dot(X, W1) + B1  #[0.3, 0.7, 1.1]
print(A1)

Z1 = sigmoid(A1)  #[0.57444252, 0.66818777, 0.75026011]
print(Z1) # [0.57444252 0.66818777 0.75026011]

W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
B2 = np.array([0.1, 0.2])

print(Z1.shape) # (3,)
print(W2.shape) # (3, 2)
print(B2.shape) # (2,)

A2 = np.dot(Z1, W2) + B2  #[0.8, 1.0]
Z2 = sigmoid(A2)  #[0.68997448, 0.73105858]
print(A2) # [0.8 1.0]
print(Z2) # [0.68997448 0.73105858]

def identity_function(x):
    """Identity function"""
    return x

W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
B3 = np.array([0.1, 0.2])

A3 = np.dot(Z2, W3) + B3  
Y = identity_function(A3)  
print(A3) 
print(Y) 