import numpy as np
# A = np.array([[1, 2], [3, 4]])
# B = np.array([[5, 6], [7, 8]])
# array = np.dot(A, B)
# print(array)

# A = np.array([[1,2,3], [4,5,6]])
# B = np.array([[1,2], [3,4], [5,6]])
# array = np.dot(A, B)
# print(array)

X = np.array([1, 2])
W = np.array([[1, 3, 5], [2, 4, 6]])
shapeX = X.shape
shapeW = W.shape
Y = np.dot(X, W)
print(shapeX, shapeW, Y)