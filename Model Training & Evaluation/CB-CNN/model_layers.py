from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
import tensorflow as tf


def get_CB_CNN_model(num_class):
  model = Sequential()

  early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

  model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(96,96, 1))) #L1
  model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))  #l2
  model.add(MaxPooling2D(pool_size=(2, 2)))  #  1
  model.add(Dropout(0.25))

  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))  # l3
  model.add(MaxPooling2D(pool_size=(2, 2)))  # 2
  model.add(Conv2D(128, kernel_size=(3, 3), activation='relu')) # l4
  model.add(MaxPooling2D(pool_size=(2, 2))) # 2
  model.add(Dropout(0.25))

  model.add(Flatten())
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(num_class, activation='softmax'))

  return model