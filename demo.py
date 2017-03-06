import numpy as np
import time

# formasi NN 4-4-3 untuk MLP
n_hidden = 4
n_in = 4
n_out = 3

learning_rate = 0.01
momentum = 0.9

np.random.seed(0)

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def train(x, t, V, W, bv, bw):

    
    # forward
    # hitung input dengan bobot hidden layer matriks 1x4 dot 4x4 menghasilkan matriks 1x4
    A = np.dot(x, V) + bv
    Z = sigmoid(A)

   
    # hitung hidden layer dengan bobot layer output matriks 1x4 dot 4x3 menghasilkan matriks 1x3
    B = np.dot(Z, W) + bw
    Y = sigmoid(B)
    

    # backward error ke layer sebelumnya
    # eror pada lapisan output
    Ew = Y - t
    # error pada lapisan hidden A adalah matriks 4,1 dikali dengan W=4x3 Ew=3,1
    # print "x"
    # print sigmoid(A)
    # print np.shape([4,5,6])
    # print "x"
    Ev = sigmoid(A) * np.dot(W, Ew)

    # error untuk lapisan output di dot outer dengan hasil dari layer hidden
    dW = np.outer(Z, Ew)
    # error untuk lapisan hidden di dot outer dengan hasil dari layer input
    dV = np.outer(x, Ev)

    # cross-entropy loss tambahan rumus belum digunakan di kelas bu afia
    # loss = -np.mean ( t * np.log(Y) + (1 - t) * np.log(1 - Y) )

    #euclidean square, jarak euclidean antara hasil dan fakta E(y,y') dari wikipedia
    loss =0.5*(np.sum(np.power(Y-t,2)))


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

# Setup initial parameters MLP
# weight layer hidden
V = np.random.normal(scale=0.1, size=(n_in, n_hidden))
# weight layer output
W = np.random.normal(scale=0.1, size=(n_hidden, n_out))

# Setup initial parameters SLP
W2 = np.random.normal(scale=0.1, size=(n_in, n_out))

print V
print W

# bias
bv = np.zeros(n_hidden)
bw = np.zeros(n_out)
# bias SLP
bw2 = np.zeros(n_out)

params = [V,W,bv,bw]

# split 120 training:30 testing
datasumber = np.loadtxt('iris.data-feature.txt',delimiter=',')
datasumberlabel = np.loadtxt('iris.data-feature-label.txt',delimiter=',')

kelas1=datasumber[0:40,:]
kelas2=datasumber[50:90,:]
kelas3=datasumber[100:140,:]

label1=datasumberlabel[0:40,:]
label2=datasumberlabel[50:90,:]
label3=datasumberlabel[100:140,:]

# Test data
kelastest1=datasumber[40:50,:]
kelastest2=datasumber[90:100,:]
kelastest3=datasumber[140:150,:]

labeltest1=datasumberlabel[40:50,:]
labeltest2=datasumberlabel[90:100,:]
labeltest3=datasumberlabel[140:150,:]

# print kelas1.shape

datatraining=np.concatenate((kelas1,kelas2,kelas3),axis=0)

datalabel = np.concatenate((label1,label2,label3),axis=0)

datatest=np.concatenate((kelastest1,kelastest2,kelastest3),axis=0)
datalabeltest = np.concatenate((labeltest1,labeltest2,labeltest3),axis=0)

# print datatraining.shape
# print datalabel.shape

# Train
for epoch in range(10000):
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
print "PREDIKSI DATATEST"
# print datatest
# print datalabeltest
print params
for i in range(datatest.shape[0]):
    print predict(datatest[i], *params)
# print "prediksi baris ke 139: "
# print predict(datatraining[140], *params)
# print "prediksi baris ke 140: "
# print predict(datatraining[141], *params)

