
# coding: utf-8

# In[1]:



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
from keras.layers import Dense, Input, GlobalMaxPooling1D, Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, Dropout, concatenate, Embedding
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping, Callback
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers


# In[2]:


train_data = pd.read_csv('data/train.csv')
test_data = pd.read_csv('data/test.csv')


# # Data

# In[3]:


train_data.head(3)


# In[4]:


train_data.sort_values(by='id').head(6)
# Id column doesn


# In[5]:


print('shape')
print(train_data.shape,test_data.shape)

print('\ntrain data missing?')
print(train_data.isnull().sum())

print('\ntest data missing?')
print(test_data.isnull().sum())


# In[6]:


le = preprocessing.LabelEncoder()

labels = le.fit_transform(train_data['author'])


# In[7]:


val_data = train_data.sample(frac=0.2, random_state=42)
train_data = train_data.drop(val_data.index)


# In[8]:


print(train_data.shape)
print(val_data.shape)


# # Data Preprocessing

# ### Transform text to indexes

# In[9]:


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


# In[10]:


# Transfers sequences into standardized vector size
# if no maxlen is given, maxlen will be lenght of longes sentence in dataset
# this means 861 for us
X_train = pad_sequences(sequences_train)
X_val = pad_sequences(sequences_valid, maxlen=X_train.shape[1])

y_train = to_categorical(np.asarray(labels[train_data.index]))
y_val = to_categorical(np.asarray(labels[val_data.index]))

print('Shape of X train: {0} and X validation tensor: {1}'.format(X_train.shape, X_val.shape) )
print('Shape of label train: {0}  and validation tensor: {1}'.format(y_train.shape, y_val.shape) )


# ### Import embeddings (pretrained from google)

# In[11]:


# Load pretrained word vectors
word_vectors = KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin', binary=True)

EMBEDDING_DIM = 300

# vocab size will be either size of word_index, or Num_words (whichever is smaller)
vocabulary_size = min(len(word_index) + 1, NUM_WORDS) 

embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))


# In[12]:


# showcase of how does the word_index dict looks like
for word, i in word_index.items():
    print('word: {0} \t idx: {1} '.format(word, i))
    if i >= 4:
        break


# In[13]:


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


# In[14]:


# create Keras embedding layer
# note use of embedding matrix that we previously created as weights 
embedding_layer = Embedding(vocabulary_size,
            
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)
# 'trainable = false' eliminates 6 000 000 (vocab_size*emb_size) trainable parameters! 
# (highly recommended for testing purposes)


# # Model

# In[15]:


sequence_length = X_train.shape[1]
filter_sizes = [3, 4, 5]
num_filters = 100
drop = 0.4

inputs = Input(shape=(sequence_length,))
embedding = embedding_layer(inputs)
reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)

conv_0 = Conv2D(filters=num_filters, kernel_size=(filter_sizes[0], EMBEDDING_DIM),
                activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),
                activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),
                activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)

maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)

merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
flatten = Flatten()(merged_tensor)
reshape = Reshape((3 * num_filters, ))(flatten)
dropout = Dropout(drop)(flatten)
output = Dense(units=3, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(dropout)

# this creates a model that includes
model = Model(inputs, output)


# In[16]:


model.summary()


# In[23]:


adam = Adam(lr=1e-3)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])

early_stop = EarlyStopping(monitor='val_loss')


# In[24]:


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        
training_history = LossHistory()


# In[ ]:


model.fit(X_train, y_train, batch_size=1000, epochs=50, verbose=1, validation_data=(X_val, y_val),
         callbacks=[early_stop, training_history])  # starts training


# In[ ]:


import pickle

history = training_history.losses

# save callbacks (progress)
pickle.dump(history, 'models/CNN_train_callbacks.pickle')

# save model as model.json
with open("models/CNN_model.json", "w") as json_file:
    
# save weights to HDF5
model.save_weights("models/CNN_model.h5")



# In[53]:


sequences_test = tokenizer.texts_to_sequences(test_data.text)

X_test = pad_sequences(sequences_test, maxlen=X_train.shape[1])
y_pred = model.predict(X_test)


# In[55]:


# save model as model.json
import pickle

    pickle.dump(y_pred, 'models/CNN_model_predictions.pickle')

