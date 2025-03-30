import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle
'''
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

print(x_train.shape) # (60000, 784)
print(t_train.shape) # (60000,)
print(x_test.shape) # (10000, 784)
print(t_test.shape) # (10000,)

def img_show(img): # 이미지 출력 함수
    pil_img = Image.fromarray(np.uint8(img)) # uint8은 0~255의 정수형으로 변환 -> 이유는 이미지의 픽셀값이 0~255이기 때문
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
img = x_train[0] # x_train의 첫번째 이미지
label = t_train[0]
print(label) 

print(img.shape) # (784,)
img = img.reshape(28, 28) # 원래의 28x28 이미지로 변형
print(img.shape) # (28, 28)
img_show(img) # 5
'''
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

def init_network():
    with open(os.path.dirname(__file__) + "/../dataset/mnist.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def sigmoid(x):
    """Sigmoid function"""
    return 1 / (1 + np.exp(-x))

def softmax(x):
    """Softmax function"""
    exp_x = np.exp(x - np.max(x)) # overflow 방지
    return exp_x / np.sum(exp_x, axis=1, keepdims=True) # axis=1: 행 방향으로 합계, keepdims=True: 차원 유지


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data() # x: 테스트 데이터, t: 정답 레이블
network = init_network() # 네트워크 초기화
accuracy_cnt = 0 # 정확도 카운트
for i in range(len(x)):
    y = predict(network, x[i]) # 예측값
    p = np.argmax(y) # 가장 큰 값의 인덱스(예측 레이블) -> 이유는 softmax 함수의 출력값이 확률 분포이기 때문
    if p == t[i]: # 예측 레이블과 정답 레이블 비교
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x))) # 정확도 출력