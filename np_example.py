import numpy as np
import math
import time 
import random as rand
import matplotlib.pyplot as plt

TRAIN_SET_SIZE = 50
PI = math.pi
N = 5
epsilon = 0.05
epoch = 50000

c = np.ndarray(N)
W = np.ndarray(N)
V = np.ndarray(N)
global b 
b = float(0)

def sigmoid(x: float) -> float:
    return (1.0 / (1.0 + math.exp(-x)))

def f_theta(x: float) -> float:
    result = b
    for i in range(N):
        result += V[i] * sigmoid(c[i] + W[i] *x)
    return result

def train(x: float, y: float):
    for i in range(N):
        W[i] = W[i] - epsilon * 2 * (f_theta(x) - y) * V[i] * x * (1 - sigmoid(c[i] + W[i] * x)) * sigmoid(c[i] + W[i] * x)
    
    for i in range(N):
        V[i] = V[i] - epsilon * 2 * (f_theta(x) - y) * sigmoid(c[i] + W[i] * x)

    global b
    b = b - epsilon * 2 * (f_theta(x) - y)

    for i in range(N):
        c[i] = c[i] - epsilon * 2 * (f_theta(x) - y) * V[i] * (1 - sigmoid(c[i] + W[i] * x)) * sigmoid(c[i] + W[i] * x)
    
def main():
    rand.seed(time.time())
    start = time.perf_counter()
    for i in range(N):
        W[i] = 2 * rand.random() / 2147483647 - 1
        V[i] = 2 * rand.random() / 2147483647 - 1
        c[i] = 2 * rand.random() / 2147483647 - 1

    trainSet = np.empty((TRAIN_SET_SIZE, 2))
    
    for i in range(TRAIN_SET_SIZE):
        trainSet[i] = np.array(((i * 2 * PI / TRAIN_SET_SIZE), math.sin(i * 2 * PI / TRAIN_SET_SIZE)))

    for i in range(epoch):
        for j in range(TRAIN_SET_SIZE):
            train(trainSet[j][0], trainSet[j][1])
        print("Epoch: ", i)
    total = time.perf_counter() - start
    print("Training took: ", total, " seconds.")

    x = np.empty(0) # input
    y1 = np.empty(0) # true labels
    y2 = np.empty(0) # preds

    for i in range(1000):
        np.append(x, i * 2 * PI / 1000)
        np.append(y1, math.sin(i * 2 * PI / 1000))
        np.append(y2, f_theta(i * 2 * PI / 1000))
    
    plt.figure(figsize=(10, 7))
    plt.scatter(x, y1, c="b", s=4, label="True Data")
    plt.scatter(x, y2, c="g", s=4, label="Predictions")
    print(x, y1, y2)
    #plt.savefig()
    plt.show()
main()