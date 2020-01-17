import keras
from keras.layers import Dense, Conv1D, BatchNormalization, MaxPooling1D, Dropout, ELU, TimeDistributed
from keras.layers import Flatten, Bidirectional, Input, LSTM, GRU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
import numpy as np
import os

# Training parameters
batch_size = 128
epochs = 200
num_classes = 4

# Load the data.
from util.IO_util import read_json
corpus = read_json('ser_traindev.tar.gz!/train.json')
(x_train, y_train), (x_test, y_test) = corpus.load_data()
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# Input image dimensions.
input_shape = x_train.shape[1:]

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
print('y_train shape:', y_train.shape)

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

def lr_schedule(epoch):
    """Learning Rate Schedule

    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.

    # Arguments
        epoch (int): The number of epochs

    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

cnn = Sequential()
cnn.add(Conv1D(32, kernel_size=8, activation='relu', input_shape=(26, 1)))
cnn.add(BatchNormalization(axis=1))
cnn.add(ELU())
cnn.add(MaxPooling1D(pool_size=3, strides=2))

cnn.add(Dropout(0.1))
cnn.add(Conv1D(64, kernel_size=4, activation='relu'))
cnn.add(BatchNormalization(axis=1))
cnn.add(ELU())
cnn.add(MaxPooling1D(pool_size=4, strides=4))
cnn.add(Dropout(0.1))
cnn.add(Flatten())

rnn = Sequential()
rnn.add(Bidirectional(LSTM(units=128, input_shape=(None, 1707, 64))))

dense = Sequential()
dense.add(Dropout(0.3))
dense.add(Dense(4, activation='softmax'))

main_input = Input(shape=input_shape)

model = TimeDistributed(cnn)(main_input)
model = rnn(model)
model = dense(model)

from keras.models import Model
final_model = Model(inputs=main_input, outputs=model)
final_model.compile(loss='categorical_crossentropy',
                    optimizer=Adam(learning_rate=lr_schedule(epochs)),
                    metrics=['accuracy'])
final_model.build(input_shape=input_shape)
final_model.summary()

# Prepare model model saving directory.
save_dir = os.path.join(os.getcwd(), 'saved_models')
model_name = 'model.{epoch:03d}.h5'
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, model_name)

# Prepare callbacks for model saving and for learning rate adjustment.
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_acc',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]


final_model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_data=(x_test, y_test),
          shuffle=True,
          callbacks=callbacks)

from keras.models import load_model
import time
final_model.save('model' + str(time.time()) + '.h5')
pred = read_json('ser_traindev.tar.gz!/dev.json')
x_pred = pred.load_data()
x_pred = np.reshape(pred.load_data(), (3342, 1707, 26, 1))
predict = final_model.predict(x_pred)
predict = np.argmax(predict, axis=1)

import json
with open('re.json', 'w') as f:
    s = json.dumps(predict.tolist())
    f.write(s)

# Score trained model.
scores = final_model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
