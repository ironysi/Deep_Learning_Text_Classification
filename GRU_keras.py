
# coding: utf-8

# In[5]:


import pandas as pd
import numpy as np

# preprocessing
from sklearn import preprocessing

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

import gensim
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from gensim.utils import simple_preprocess

# Model
from keras.layers import Input, GRU, Dense, GlobalMaxPool1D, Dropout, Embedding
from keras.callbacks import EarlyStopping, Callback
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers


# In[6]:


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


# In[7]:


le = preprocessing.LabelEncoder()
labels = le.fit_transform(train_data['author'])

val_data = train_data.sample(frac=0.2, random_state=42)
train_data = train_data.drop(val_data.index)


# In[8]:


texts = train_data.text
NUM_WORDS = 20000

tokenizer = Tokenizer(num_words=NUM_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
tokenizer.fit_on_texts(texts)

# transfer sentences into sequences of word indexes
sequences_train = tokenizer.texts_to_sequences(texts)
sequences_valid = tokenizer.texts_to_sequences(val_data.text)

word_index = tokenizer.word_index

print(sequences_train[0])
print('\n')
print(sequences_valid[0])
print('\n')
print('Found %s unique tokens.' % len(word_index))


# In[9]:


X_train = pad_sequences(sequences_train)
X_val = pad_sequences(sequences_valid, maxlen=X_train.shape[1])

y_train = to_categorical(np.asarray(labels[train_data.index]))
y_val = to_categorical(np.asarray(labels[val_data.index]))

print('Shape of X train: {0} and X validation tensor: {1}'.format(X_train.shape, X_val.shape) )
print('Shape of label train: {0}  and validation tensor: {1}'.format(y_train.shape, y_val.shape) )


# In[10]:


# Load pretrained word vectors
word_vectors = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

EMBEDDING_DIM = 300

# vocab size will be either size of word_index, or Num_words (whichever is smaller)
vocabulary_size = min(len(word_index) + 1, NUM_WORDS) 

embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))


# In[11]:


for word, i in word_index.items():
    if i>=NUM_WORDS:
        continue
    try:
        # get vector for each word 
        embedding_vector = word_vectors[word]
        # save vector into embedding matrix
        embedding_matrix[i] = embedding_vector
    except KeyError:
        # generate random vector if the word was not found in pretrained vectors
        embedding_matrix[i] = np.random.normal(0, np.sqrt(0.25), EMBEDDING_DIM)

# to free up some memory
del(word_vectors)


# In[12]:


# create Keras embedding layer
# note use of embedding matrix that we previously created as weights 
embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)
# 'trainable = false' eliminates 6 000 000 (vocab_size*emb_size) trainable parameters! 
# (highly recommended for testing purposes)


# In[13]:


inp = Input(shape=(X_train.shape[1],))

emb = embedding_layer(inp)
gru = GRU(100, activation='relu', return_sequences=True, name='lstm_layer')(emb)
maxPool = GlobalMaxPool1D()(gru)
drop0 = Dropout(0.3)(maxPool)
hidden = Dense(50, activation="relu")(drop0)
drop1 = Dropout(0.1)(hidden)
output = Dense(3, activation='softmax')(drop1)


# In[14]:


model = Model(inp, output)

adam = Adam(lr=1e-3)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])

early_stop = EarlyStopping(monitor='val_loss')


# In[15]:


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
training_history = LossHistory()


# In[16]:


model.summary()


# In[ ]:


model.fit(X_train, y_train, batch_size=1000, epochs=10, verbose=1, validation_data=(X_val, y_val),
         callbacks=[training_history, early_stop])


# In[ ]:


import pickle

history = training_history.losses

# save callbacks (progress)
pickle.dump(history, 'models/GRU_train_callbacks.pickle')

# save model as model.json
with open("models/GRU_model.json", "w") as json_file:
    
# save weights to HDF5
model.save_weights("models/GRU_model.h5")

