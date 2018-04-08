import pickle
import numpy as np

from keras import regularizers
from keras.callbacks import EarlyStopping, Callback
from keras.layers import Dense, Input, Conv2D, MaxPooling2D, Dropout, \
    concatenate, Embedding
from keras.layers.core import Reshape, Flatten
from keras.models import Model
from keras.optimizers import Adam

embedding_matrix = np.load('data/processed/embedding_matrix' + '.npy')

X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')

X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')

X_test = np.load('data/processed/X_test.npy')

# Model parameters
vocabulary_size = embedding_matrix.shape[0]
sequence_length = X_train.shape[1]
filter_sizes = [3, 4, 5]
num_filters = 100
drop = 0.5
EMBEDDING_DIM = 300

# create Keras embedding layer
# note use of embedding matrix that we previously created as weights
embedding_layer = Embedding(vocabulary_size,

                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)
# 'trainable = false' eliminates 6 000 000 (vocab_size*emb_size) trainable parameters!

inputs = Input(shape=(sequence_length,))
embedding = embedding_layer(inputs)
reshape = Reshape((sequence_length, EMBEDDING_DIM, 1))(embedding)

conv_0 = Conv2D(filters=num_filters, kernel_size=(filter_sizes[0], EMBEDDING_DIM),
                activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),
                activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)
conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),
                activation='relu', kernel_regularizer=regularizers.l2(0.01))(reshape)

maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1, 1))(conv_0)
maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1, 1))(conv_1)
maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1, 1))(conv_2)

merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
flatten = Flatten()(merged_tensor)
reshape = Reshape((3 * num_filters,))(flatten)
dropout = Dropout(drop)(flatten)
output = Dense(units=3, activation='softmax', kernel_regularizer=regularizers.l2(0.01))(dropout)

model = Model(inputs, output)
print(model.summary())

adam = Adam(lr=1e-3)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])


class LossHistory(Callback):
    def on_train_begin(self, logs=None):
        if logs is None:
            logs = {}
        self.losses = []

    def on_batch_end(self, batch, logs=None):
        if logs is None:
            logs = {}
        self.losses.append(logs.get('loss'))


early_stop = EarlyStopping(monitor='val_loss')
training_history = LossHistory()

model.fit(X_train, y_train, batch_size=1000, epochs=50, verbose=1, validation_data=(X_val, y_val),
          callbacks=[early_stop, training_history])

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
