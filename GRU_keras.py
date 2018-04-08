import numpy as np
import pickle

from keras.layers import Input, GRU, Dense, GlobalMaxPool1D, Dropout, Embedding
from keras.callbacks import EarlyStopping, Callback
from keras.optimizers import Adam
from keras.models import Model

embedding_matrix = np.load('data/processed/embedding_matrix' + '.npy')

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')

X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')

X_test = np.load('data/processed/X_test.npy')

vocabulary_size = embedding_matrix.shape[0]
EMBEDDING_DIM = 300

# create Keras embedding layer
# note use of embedding matrix that we previously created as weights 
embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)
# 'trainable = false' eliminates 6 000 000 (vocab_size*emb_size) trainable parameters!

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

print(model.summary())


class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


training_history = LossHistory()
early_stop = EarlyStopping(monitor='val_loss')

model.fit(X_train, y_train, batch_size=1000, epochs=10, verbose=1, validation_data=(X_val, y_val),
          callbacks=[training_history, early_stop])

# save callbacks (progress)
with open('models/CNN_train_callbacks.pickle', 'wb') as file:
    pickle.dump(training_history.losses, file)
    file.close()

# save model as model.json
with open('models/CNN_model.pickle', 'wb') as file:
    json_model = model.to_json()
    pickle.dump(json_model, file)

# save weights to HDF5
model.save_weights("models/CNN_model.h5")
