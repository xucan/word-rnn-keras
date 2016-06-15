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

dpath = "../train_dict.pkl"
#generate dict
vob = cPickle.load(open(dpath, "r"))
vobindex = {value:key for key, value in vob.items()}

maxlen = 31
diclen = len(vob)

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

if os.path.exists('weights.hdf5'):
     print('Loading existing weights')
     model.load_weights('weights.hdf5')
def sample(a, temperature=1.0):
    # helper function to sample an index from a probability array
    a = np.log(a) / temperature
    a = np.exp(a) / np.sum(np.exp(a))
    return np.argmax(np.random.multinomial(1, a, 1))

#iteration = 1
while True:
#    print('Iteration', iteration)
    wait = raw_input("please press enter")
    for diversity in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,1.2]:
        print()
        print('----- diversity:', diversity)

        sentence = ['<s>']
        print('----- Generating with seed: "' + sentence[0] + '"')
        sys.stdout.write(sentence[0])

        for iteration in range(100):
            x = np.zeros((1, len(sentence)))
            for t, word in enumerate(sentence):
                x[0,t] = vob.get(word,1)

            preds = model.predict(x, verbose=0)[0][-1]
            next_index = sample(preds, diversity)
#            next_index = np.argmax(preds)
            next_word = vobindex.get(next_index)

            sentence.append(next_word)
            sys.stdout.write(' ')
            sys.stdout.write(next_word)
            sys.stdout.flush()
            if next_word == '</s>': break
        print()






























