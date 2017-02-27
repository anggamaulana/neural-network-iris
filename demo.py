import numpy as np
import time

n_hidden = 4
n_in = 4
n_out = 3

learning_rate = 0.01
momentum = 0.9

np.random.seed(0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def tanh_prime(x):
    return  1 - np.tanh(x)**2

def train(x, t, V, W, bv, bw):

    # forward
    A = np.dot(x, V) + bv
    Z = np.tanh(A)

    B = np.dot(Z, W) + bw
    Y = sigmoid(B)

    # backward
    Ew = Y - t
    Ev = tanh_prime(A) * np.dot(W, Ew)

    dW = np.outer(Z, Ew)
    dV = np.outer(x, Ev)

    # cross-entropy loss
    loss = -np.mean ( t * np.log(Y) + (1 - t) * np.log(1 - Y) )

    # Note that we use error for each layer as a gradient
    # for biases

    return  loss, (dV, dW, Ev, Ew)

def predict(x, V, W, bv, bw):
    A = np.dot(x, V) + bv
    B = np.dot(np.tanh(A), W) + bw
    hasil = (sigmoid(B) > 0.5).astype(int)
    if hasil[0]==1:
        return 'Iris-setosa'
    elif hasil[1]==1:
        return 'Iris-versicolor'
    elif hasil[2]==1:
        return 'Iris-virginica'
    else:
        return 'tidak tahu'

# Setup initial parameters
# Note that initialization is cruxial for first-order methods!

V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

bv = np.zeros(n_hidden)
bw = np.zeros(n_out)

params = [V,W,bv,bw]

# Generate some data
datatraining = np.loadtxt('iris.data-feature.txt',delimiter=',')
datalabel = np.loadtxt('iris.data-feature-label.txt',delimiter=',')

print datatraining
print datalabel

# Train
for epoch in range(500):
    err = []
    upd = [0]*len(params)

    t0 = time.clock()
    for i in range(datatraining.shape[0]):
        loss, grad = train(datatraining[i], datalabel[i], *params)

        for j in range(len(params)):
            params[j] -= upd[j]

        for j in range(len(params)):
            upd[j] = learning_rate * grad[j]

        err.append( loss )


    
    print "Epoch: %d, Loss: %.8f, Time: %.4fs" % (
                epoch, np.mean( err ), time.clock()-t0 )

# Hasil model
print predict(datatraining[0], *params)
print predict(datatraining[140], *params)

