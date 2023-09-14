from numpy import *
from random import *
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

def vectomat(A,B):
    C = zeros([len(A),len(B)])
    for i in range(len(A)):
        for j in range(len(B)):
            C[i][j] = A[i]*B[j]
    return C

def FFTnor(X,Fs):
    N = len(X)
    #T = 1/Fs
    Xf = fft(X)
    Xf = 2.0/N * abs(Xf[0:N//2])
    Xf = Xf/max(Xf)
    #freq = fftfreq(N, T)[:N//2]
    #plt.plot(freq, Xf)
    #plt.grid()
    #plt.show()
    return Xf

def loaddata(Filename,N):
    file = open(Filename)
    content = file.readlines()
    rows = len(content)
    data = content[63:rows]
    data = [data[i].replace('\t', '') for i in range(size(data))]
    data = [data[i].rstrip() for i in range(size(data))]
    data = [data[i].split() for i in range(size(data))]
    for i in range(size(data,0)):
        for j in range(size(data,1)):
            data[i][j] = float(data[i][j])
    data = [data[i][0] for i in range(size(data,0))]
    if size(data) >= N:
        data = data[0:N-1]
    if size(data) < N:
        M = N-size(data)
        data.extend(zeros(M).tolist())
    return data

def sigmoid(V):
    Y = 1 / (1 + exp(-V))
    return Y

def softmax(V):
    EV = exp(V)
    Y  = EV / sum(EV)
    return Y

def dropout(Y,r):
    L = size(Y)
    Yd  = zeros(L)
    num = L*(1-r)
    num = round(num)
    idy = sample(range(0, L), int(num))
    Yd[idy] = 1/(1-r)
    for i in range(L):
        Y[i] = Y[i]*Yd[i]
    return Y

def DNNtest(W1,W2,W3,W4,X):
    V1 = matmul(W1,transpose(X))
    Y1 = sigmoid(V1)
    V2 = matmul(W2,Y1)
    Y2 = sigmoid(V2)
    V3 = matmul(W3,Y2)
    Y3 = sigmoid(V3)
    V4 = matmul(W4,Y3)
    Y4 = softmax(V4)
    return Y4

def DNNtrain(W1,W2,W3,W4,X,D,r,alpha):
    V1 = matmul(W1,X) # turn to row vector
    Y1 = sigmoid(V1)
    Y1 = dropout(Y1,r)
    V2 = matmul(W2,Y1)
    Y2 = sigmoid(V2)
    Y2 = dropout(Y2,r)
    V3 = matmul(W3,Y2)
    Y3 = sigmoid(V3)
    Y3 = dropout(Y3,r)
    V4 = matmul(W4,Y3)
    Y4 = softmax(V4)
    E4  = subtract(D, Y4)
    Delta4 = E4
    E3 = matmul(transpose(W4),Delta4)
    Sigdev = [i1 * i2 for i1,i2 in zip(Y3,(1-Y3))]
    Delta3 = [i1 * i2 for i1,i2 in zip(Sigdev, E3)] 
    E2 = matmul(transpose(W3),Delta3)
    Sigdev = [i1 * i2 for i1,i2 in zip(Y2,(1-Y2))]
    Delta2 = [i1 * i2 for i1,i2 in zip(Sigdev, E2)] 
    E1 = matmul(transpose(W2),Delta2)
    Sigdev = [i1 * i2 for i1,i2 in zip(Y1,(1-Y1))] 
    Delta1 = [i1 * i2 for i1,i2 in zip(Sigdev, E1)]
    W4 = W4 + alpha*vectomat(Delta4,Y3)
    W3 = W3 + alpha*vectomat(Delta3,Y2)
    W2 = W2 + alpha*vectomat(Delta2,Y1)
    W1 = W1 + alpha*vectomat(Delta1,X)
    return W1,W2,W3,W4,Y4
  
def DNNtest(W1,W2,W3,W4,X,D):
    V1 = matmul(W1,transpose(X))
    Y1 = sigmoid(V1)
    V2 = matmul(W2,Y1)
    Y2 = sigmoid(V2)
    V3 = matmul(W3,Y2)
    Y3 = sigmoid(V3)
    V4 = matmul(W4,Y3)
    Y4 = softmax(V4)
    Y4 = around(Y4)
    P  = (D == Y4.tolist())
    return Y4,P

def DNN(W1,W2,W3,W4,X):
    V1 = matmul(W1,transpose(X))
    Y1 = sigmoid(V1)
    V2 = matmul(W2,Y1)
    Y2 = sigmoid(V2)
    V3 = matmul(W3,Y2)
    Y3 = sigmoid(V3)
    V4 = matmul(W4,Y3)
    Y4 = softmax(V4)
    Y4 = around(Y4)
    return Y4

    
    
    
