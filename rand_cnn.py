from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import numpy as np
import tensorflow as tf
import math
import random

#%% parameters
# input image dimensions
img_rows, img_cols = 28, 28
batch_size = 128
num_classes = 2
epochs = 5
input_shape = ( img_rows, img_cols,1)

#%% Build the baseline model
def build_model():
    model = Sequential()
    model.add(Conv2D(8, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(16, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model

#%% Read MNIST data and create train test for each model
# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Look for 0 and 8 digits
where_is_0 = list(np.where(y_train == 0)[0])
where_is_8 = list(np.where(y_train == 8)[0])
x_train_0 = x_train[where_is_0]
x_train_8 = x_train[where_is_8]

ms=10 # number of traning samples

x_train_model_1 = np.vstack((x_train_0[:ms] , x_train_8[:ms]))
x_train_model_2 = np.vstack((x_train_0[ms:2*ms] , x_train_8[ms:2*ms]))
x_train_model_3 = np.vstack((x_train_0[2*ms:3*ms] , x_train_8[2*ms:3*ms]))
x_train_model_4 = np.vstack((x_train_0[3*ms:4*ms] , x_train_8[3*ms:4*ms]))
x_train_model_5 = np.vstack((x_train_0[4*ms:5*ms] , x_train_8[4*ms:5*ms]))
x_train_model_6 = np.vstack((x_train_0[5*ms:6*ms] , x_train_8[5*ms:6*ms]))
x_train_model_7 = np.vstack((x_train_0[6*ms:7*ms] , x_train_8[6*ms:7*ms]))
x_train_model_8 = np.vstack((x_train_0[7*ms:8*ms] , x_train_8[7*ms:8*ms]))
x_train_model_9 = np.vstack((x_train_0[8*ms:9*ms] , x_train_8[8*ms:9*ms]))
x_train_model_10 = np.vstack((x_train_0[9*ms:10*ms] , x_train_8[9*ms:10*ms]))

# y_train is the same for all models: 1000 samples of 0 and 1000 samples of 8
y_train = np.array(ms*[0] + ms*[1])
y_train = keras.utils.to_categorical(y_train, num_classes)

where_is_0 = list(np.where(y_test == 0)[0])
where_is_8 = list(np.where(y_test == 8)[0])

x_test_0 = x_test[where_is_0[:100]]
x_test_8 = x_test[where_is_8[:100]]

x_test = np.vstack((x_test_0 , x_test_8))
y_test = np.array(100*[0] + 100*[1])
y_test = keras.utils.to_categorical(y_test, num_classes)

#%% energy model
def compound_poissrnd( lambda1,lambda2,K,L):  
    M=abs(np.random.poisson(lambda1))
    E=np.zeros((K,L))
    for m in range(0,M):
        E=E+np.random.poisson(lambda2,(K,L))
    return E

#%% Main
# wireless parameters
K=10
N=128
L=300
mc=1 # monte carlo iterations
alphaa=3.7
fc=2.5*10**9
d0=1
G0=(3*10**8/(4*math.pi*d0*fc))**2
Tout=1
pc=1
Bmax=200
Ec=pc*Tout
Efl=1
Efix=Efl+Ec
lambda1=5 
lambda2=0.5 
N0=1.6*10**(-12)
siDB=25
a=0.006
SI0=10**(siDB/20)
numb_sck=[]
# cycling
rr=10 #test every rr rounds
accuracy_m=np.zeros((int(L/rr),mc))
nm=0
for m in range(0,mc):
        print(m)
        H=(1/math.sqrt(2))*(np.random.randn(N,K,L)+1j*np.random.randn(N,K,L))
        d=G0*np.random.randint(50,500,K)**(-alphaa)  
        D=np.diag(d)
        E=compound_poissrnd(lambda1,lambda2,K,L+1)
        B=E[0:K,0]
        n_tot=0
        accuracy = []
        test_loss = []
        new_config = [] 
        r=0
        for l in range(0,L):  
            ur= np.random.randint(0,2,K)
            G=np.dot(H[:,:,l],D**(1/2))
            A1=np.dot((G.conj()).transpose(),G)
            A=np.linalg.inv(A1)                            
            nl=0
            us=[]
            xk=np.zeros(K)
            xs=np.zeros(K)
            for k in range(0,K):
              if ur[k]==1:  
                pk=A[k,k].real*N0*SI0              
                xk[k]=pk*Tout*+Efix
                aa=B[k]/xk[k]
                bb=(Bmax-B[k]-E[k,l+1])/xk[k]
                if 1 <=aa and -1 <=bb:
                   nl=nl+1
                   us.append(k+1)
                   xs[k]=1
            for k in range(0,K):      
                if l<L-1:
                   B[k]=min(Bmax,B[k]-xs[k]*xk[k]+E[k,l+1])
            #FL
            config=[]
            # slave 1 
            if 1 in us:    
               slave1 = build_model()
               if l>0 and len(new_config)>0:
                  slave1.set_weights(new_config)
               slave1.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.sgd(lr=a),metrics=['accuracy'])
               slave1.fit(x_train_model_1,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
               config.append(slave1.get_weights())     
            # slave 2 
            if 2 in us:    
               slave2 = build_model()
               if l>0 and len(new_config)>0:
                  slave2.set_weights(new_config)
               slave2.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.sgd(lr=a),metrics=['accuracy'])
               slave2.fit(x_train_model_2,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
               config.append(slave2.get_weights()) 
            # slave 3 
            if 3 in us:    
               slave3 = build_model()
               if l>0 and len(new_config)>0:
                  slave3.set_weights(new_config)
               slave3.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.sgd(lr=a),metrics=['accuracy'])
               slave3.fit(x_train_model_3,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
               config.append(slave3.get_weights())
            # slave 4 
            if 4 in us:    
               slave4 = build_model()
               if l>0 and len(new_config)>0:
                  slave4.set_weights(new_config)
               slave4.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.sgd(lr=a),metrics=['accuracy'])
               slave4.fit(x_train_model_4,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
               config.append(slave4.get_weights()) 
            # slave 5 
            if 5 in us:    
               slave5 = build_model()
               if l>0 and len(new_config)>0:
                  slave5.set_weights(new_config)
               slave5.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.sgd(lr=a),metrics=['accuracy'])
               slave5.fit(x_train_model_5,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
               config.append(slave5.get_weights()) 
            # slave 6 
            if 6 in us:    
               slave6 = build_model()
               if l>0 and len(new_config)>0:
                  slave6.set_weights(new_config)
               slave6.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.sgd(lr=a),metrics=['accuracy'])
               slave6.fit(x_train_model_6,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
               config.append(slave6.get_weights()) 
            # slave 7 
            if 7 in us:    
               slave7 = build_model()
               if l>0 and len(new_config)>0:
                  slave7.set_weights(new_config)
               slave7.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.sgd(lr=a),metrics=['accuracy'])
               slave7.fit(x_train_model_7,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
               config.append(slave7.get_weights()) 
            # slave 8 
            if 8 in us:    
               slave8 = build_model()
               if l>0 and len(new_config)>0:
                  slave8.set_weights(new_config)
               slave8.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.sgd(lr=a),metrics=['accuracy'])
               slave8.fit(x_train_model_8,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
               config.append(slave8.get_weights())
            # slave 9 
            if 1 in us:    
               slave9 = build_model()
               if l>0 and len(new_config)>0:
                  slave9.set_weights(new_config)
               slave9.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.sgd(lr=a),metrics=['accuracy'])
               slave9.fit(x_train_model_9,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
               config.append(slave9.get_weights()) 
            # slave 10 
            if 10 in us:    
               slave10 = build_model()
               if l>0 and len(new_config)>0:
                  slave10.set_weights(new_config)
               slave10.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.sgd(lr=a),metrics=['accuracy'])
               slave10.fit(x_train_model_10,y_train,batch_size=batch_size,epochs=epochs,verbose=1)
               config.append(slave10.get_weights()) 
            # master         
            if len(config)>0:
               new_config = []
               for k in range(8):   
                   x = np.zeros(config[0][k].shape)
                   for conf in config:
                       x += conf[k]
                   new_config.append(x/len(config))
            # test every rr rounds
            if l % rr == 0: 
               master = build_model()
               if len(new_config)>0:
                  master.set_weights(new_config)
               master.compile(loss=keras.losses.binary_crossentropy,optimizer=keras.optimizers.sgd(lr=a),metrics=['accuracy'])
               score = master.evaluate(x_test, y_test, verbose=0)
               accuracy.append(score[1])
               test_loss.append(score[0])
               accuracy_m[r,m]=score[1]
               r=r+1   
            n_tot=n_tot+nl
        nm=nm+n_tot/L
numb_sck=nm/mc  
accuracy_final=np.mean(accuracy_m, axis=1)                 
np.savetxt('rand_cnn.txt', accuracy_final, fmt='%.5f')
                 

            
    
    
    
    
    
    
    
    

    
    
    
    
    
    
    
    
    





