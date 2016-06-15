from __future__ import print_function
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.core import Dense, Activation, Dropout, TimeDistributedDense
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
import numpy as np
import cPickle
import sys
import time
import os

#saving a list of losses over each batch during training
class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('val_loss'))
history = LossHistory()

#write log to file
def writefile(path, string):
    f = file(path,"a")
    f.write(string)
    f.close()

#file path
tpath = sys.argv[1]
vpath = sys.argv[2]
dpath = sys.argv[3]
epath = sys.argv[4]

#generate dict
vob = cPickle.load(open(dpath, "r"))
vobindex = {value:key for key, value in vob.items()}
print ('the size of dict: ', len(vob))

maxlen = 31
diclen = len(vob)

emb = cPickle.load(open(epath,"r"))
W = [emb]
print ('the size of emb:', emb.shape)

def read_data(path):
    dataset = cPickle.load(open(path, "r"))
    width = len(dataset)
    X = np.zeros((width, maxlen), dtype=np.int16)
    Y = np.zeros((width, maxlen, diclen), dtype=np.bool)

    for j in xrange(width):
        temp = len(dataset[j])
        for i in xrange(temp):
            X[j,i] = dataset[j][i]
            if i < temp-1:
                Y[j,i,dataset[j][i+1]] = 1
            else:
                Y[j,i,vob.get('</s>')] = 1
    return X,Y

print ("read train and val dataset...")
X,Y = read_data(tpath)

print ("the train dataset size is: ", X.shape," labels: ",Y.shape) 

#build the model: 2 stacked LSTM
tic = time.time()
print('Build model...')
model = Sequential()
model.add(Embedding(len(vob),300, mask_zero=True))
model.add(LSTM(512, return_sequences=True))
model.add(LSTM(512, return_sequences=True))
model.add(TimeDistributedDense(len(vob)))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

print('compile completed in',time.time()-tic)

if os.path.exists('model.weights'):
     print('Loading existing weights')
     model.load_weights('model.weights')
def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))
'''
hist_nll = [2.11]
# train the model, output generated text after each iteration 
for iteration in range(1,500):
    result = 'Iteration'+str(iteration)+'\n'
    print('Iteration', iteration)
    model.fit(X, Y, batch_size=128, nb_epoch=1, validation_data=(X,Y), callbacks=[history])
    result += 'the nll of valset is: ' + str(history.losses[0])+'\n'
    print('the nll of valset is :', history.losses)
    if hist_nll[iteration-1] > history.losses[0]:
        result += 'better result'+str(history.losses[0])+'\n'
        print('better result: ', history.losses[0])
        model.save_weights('model.weights', overwrite=True)
        hist_nll.append(history.losses[0])
    else:
        result += 'worse result'+str(history.losses[0])+'\n'+'stop...'
        print('worse result: ', history.losses[0])
        print('stop...')
        break
    writefile('./result.txt',result)
'''
checkpointer = ModelCheckpoint(filepath="./weights.hdf5",verbose=0, save_best_only=True)
earlystopping = EarlyStopping(patience = 0, verbose=0)
model.fit(X, Y, batch_size=128, nb_epoch=200, validation_data=(X,Y), callbacks=[checkpointer])
'''
tic1 = time.time()
ev = model.evaluate(X,Y,batch_size=128)
print('evaluate completed in',time.time()-tic1)
'''
'''
def safe_pickle(obj, filename):
    with open(filename, 'wb') as f:
        cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
fx = "./X.npy"
np.save(fx, X)
fy = "./Y.npy"
np.save(fy, Y)
'''
