#!/usr/bin/env python
# coding: utf-8

# ### To DO:
# - Hyperparameters for optimizers
# - Possibly test all parameters at the same time
# - Apply best to CIFAR

# In[45]:


#! pip install keras
#! pip install tensorflow
#! pip install torch
#! pip install tensorflow[and-cuda]


# In[2]:


import keras as keras
from __future__ import print_function
from keras.datasets import mnist
from keras.models import Sequential
from keras.optimizers import RMSprop
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import cifar10
from itertools import product
from tensorflow.keras.losses import categorical_crossentropy



# initializers
from keras.initializers import Zeros  
from keras.initializers import RandomNormal, RandomUniform  
from keras.initializers import glorot_normal, glorot_uniform 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.initializers import GlorotUniform
from keras.initializers import he_normal, he_uniform  
from keras.initializers import lecun_normal, lecun_uniform  
from tensorflow.keras import regularizers
from tensorflow.keras.initializers import GlorotUniform, Zeros, RandomNormal, RandomUniform, HeNormal, HeUniform, LecunNormal, LecunUniform
from keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


# # Task 1.1 

# In[47]:


# mnist_mlp.py
batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[48]:


# mnist_cnn.py
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# # Task  1.2 MLP Tuning

# In[49]:


# mnist_mlp.py + different initilization methods
batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
seed_value = 42
results = pd.DataFrame(columns=['Initialization Method', 'Test Loss', 'Test Accuracy'])
# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

initilization_methods = [Zeros(), 
                       RandomNormal(seed=seed_value), 
                       RandomUniform(seed=seed_value), 
                       glorot_uniform(seed=seed_value), 
                       glorot_normal(seed=seed_value), 
                       he_normal(seed=seed_value), 
                       he_uniform(seed=seed_value), 
                       lecun_normal(seed=seed_value), 
                       lecun_uniform(seed=seed_value)]

for method in initilization_methods:
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,), kernel_initializer=method))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu', kernel_initializer=method))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=method))

    # Extract name of method
    method_name = method.__class__.__name__
    
    model.summary()

    
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])

    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    
    score = model.evaluate(x_test, y_test, verbose=0)
    
    new_result = pd.DataFrame({'Initialization Method': [method_name], 'Test Loss': [score[0]], 'Test Accuracy': [score[1]]})
    results = pd.concat([results, new_result], ignore_index=True)
    
    print(results)


# In[50]:


# mnist_mlp.py + different activation functions
batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
seed_value = 42
results = pd.DataFrame(columns=['Activation Method', 'Test Loss', 'Test Accuracy'])

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

activation_methods = ['relu', 'sigmoid', 'tanh', 'linear', 'softmax']

for method in activation_methods:
    model = Sequential()
    model.add(Dense(512, activation=method, input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation=method))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    
    method_name = method
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=RMSprop(),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    new_result = pd.DataFrame({'Activation Method': [method_name], 'Test Loss': [score[0]], 'Test Accuracy': [score[1]]})
    results = pd.concat([results, new_result], ignore_index=True)
    
    print(results)


# In[51]:


# mnist_mlp.py + different optimizers
batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
seed_value = 42
results = pd.DataFrame(columns=['Optimizer', 'Test Loss', 'Test Accuracy'])

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

optimizers = [keras.optimizers.Adam, keras.optimizers.SGD, keras.optimizers.RMSprop]


for optimizer in optimizers:
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    # Use the optimizer's name as method_name
    method_name = optimizer.__name__
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer(),
                  metrics=['accuracy'])
    
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))
    
    score = model.evaluate(x_test, y_test, verbose=0)
    new_result = pd.DataFrame({'Optimizer': [method_name], 'Test Loss': [score[0]], 'Test Accuracy': [score[1]]})
    results = pd.concat([results, new_result], ignore_index=True)
    
    print(results)


# In[52]:


# mnist_mlp.py + different regularization techniques
batch_size = 128
num_classes = 10
epochs = 20

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape (10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')
seed_value = 42
results = pd.DataFrame(columns=['Regularization Method', 'Test Loss', 'Test Accuracy'])

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

regularization_methods = ['None', 'L1', 'L2', 'Dropout']

for method in regularization_methods:
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(784,)))

    if method == 'L1':
        model.add(Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l1(0.01)))
    elif method == 'L2':
        model.add(Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.01)))
    elif method == 'Dropout':
        model.add(Dropout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))

    # Use the regularization method as method_name
    method_name = method

    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=keras.optimizers.RMSprop(),
                  metrics=['accuracy'])

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        verbose=1,
                        validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test, verbose=0)
    
    new_result = pd.DataFrame({'Regularization Method': [method_name], 'Test Loss': [score[0]], 'Test Accuracy': [score[1]]})
    results = pd.concat([results, new_result], ignore_index=True)

    print(results)


# In[53]:


# mnist_mlp.py + different regularization techniques
num_classes = 10
epochs = 20

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Define hyperparameter values to test
learning_rates = [0.00001, 0.1]
momentum_values = [0, 0.9]
epsilon_values = [1e-8, 1e-4]
nesterov_values = [False, True]
batch_sizes = [64, 256]
optimizers = [keras.optimizers.Adam, keras.optimizers.SGD, keras.optimizers.RMSprop]

# Create a DataFrame to store results
results = pd.DataFrame(columns=['Optimizer', 'Learning Rate', 'Momentum', 'Epsilon', 'Nesterov', 'Batch Size', 'Test Loss', 'Test Accuracy'])

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)


# Test different hyperparameters
for optimizer in optimizers:
    for lr in learning_rates:
        for momentum in momentum_values:
            for epsilon in epsilon_values:
                for nesterov in nesterov_values:
                    for batch_size in batch_sizes:
                        model = Sequential()
                        model.add(Dense(512, activation='relu', input_shape=(784,)))
                        model.add(Dropout(0.2))
                        model.add(Dense(512, activation='relu'))
                        model.add(Dropout(0.2))
                        model.add(Dense(10, activation='softmax'))

                        optimizer_name = optimizer.__name__
                        optimizer_instance = optimizer(learning_rate=lr)

                        if optimizer_name == 'SGD':
                            optimizer_instance.momentum = momentum
                            optimizer_instance.nesterov = nesterov
                        elif optimizer_name == 'RMSprop':
                            optimizer_instance.epsilon = epsilon

                        model.compile(loss='categorical_crossentropy',
                                      optimizer=optimizer_instance,
                                      metrics=['accuracy'])

                        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=5, verbose=0, validation_data=(x_test, y_test))
                        score = model.evaluate(x_test, y_test, verbose=0)

                        new_result = pd.DataFrame({'Optimizer': [optimizer_name], 'Learning Rate': [lr], 'Momentum': [momentum], 'Epsilon': [epsilon], 'Nesterov': [nesterov], 'Batch Size': [batch_size], 'Test Loss': [score[0]], 'Test Accuracy': [score[1]]})
                        results = pd.concat([results, new_result], ignore_index=True)



# Display the results
pd.set_option('display.max_rows', None)  # To display all rows
pd.set_option('display.max_columns', None)  # To display all columns
best_results = results.sort_values(by='Test Accuracy', ascending=False).head(5)
print("Top 5 Best Performing Configurations:")
print(best_results)
print(results)


# # Parameters results:
# - Initialization method: GlorotUniform 
# - Activation function: Relu
# - Optimizer: Adam
# - Regulazation technique: Dropout
# - -----------------------------------------
# 
# # Top 5 Best Performing Configurations:
#     Optimizer  Learning Rate Momentum       Epsilon Nesterov Batch Size  \
#     62       SGD            0.1      0.9  1.000000e-04     True         64   
#     60       SGD            0.1      0.9  1.000000e-04    False         64   
#     58       SGD            0.1      0.9  1.000000e-08     True         64   
#     50       SGD            0.1        0  1.000000e-08     True         64   
#     54       SGD            0.1        0  1.000000e-04     True         64   
# 
#     Test       Loss        Test Accuracy  
#     62           0.072707         0.9780  
#     60           0.073003         0.9778  
#     58           0.070178         0.9777  
#     50           0.073058         0.9767  
#     54           0.075574         0.9765  

# # Task 1.2 CNN Tuning

# In[54]:


# mnist_cnn.py + initilazation methods
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

initilization_methods = [Zeros(), 
                       RandomNormal(seed=seed_value), 
                       RandomUniform(seed=seed_value), 
                       glorot_uniform(seed=seed_value), 
                       glorot_normal(seed=seed_value), 
                       he_normal(seed=seed_value), 
                       he_uniform(seed=seed_value), 
                       lecun_normal(seed=seed_value), 
                       lecun_uniform(seed=seed_value)]

# Create a DataFrame to store results
results = pd.DataFrame(columns=['Initialization Method', 'Test Loss', 'Test Accuracy'])

for method in initilization_methods:
    # Create a new model for each initialization method
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_initializer=method))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer=method))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer=method))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', kernel_initializer=method))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    # Append the results to the DataFrame
    new_row = {'Initialization Method': method.__class__.__name__, 'Test Loss': score[0], 'Test Accuracy': score[1]}
    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

# Display the results
print(results)


# In[55]:


# mnist_cnn.py + different activation functions

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

activation_methods = ['relu', 'sigmoid', 'tanh', 'linear', 'softmax']

# Create a DataFrame to store results
results = pd.DataFrame(columns=['Activation Method', 'Test Loss', 'Test Accuracy'])

for activation_method in activation_methods:
    # Create a new model for each activation function
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation=activation_method, input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation=activation_method))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation=activation_method))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    # Append the results to the DataFrame
    new_row = {'Activation Method': activation_method, 'Test Loss': score[0], 'Test Accuracy': score[1]}
    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

# Display the results
print(results)


# In[56]:


# mnist_cnn.py + different optimizers

batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

optimizers = [keras.optimizers.Adam, keras.optimizers.SGD, keras.optimizers.RMSprop]

# Create a DataFrame to store results
results = pd.DataFrame(columns=['Optimizer', 'Test Loss', 'Test Accuracy'])

for optimizer in optimizers:
    # Create a new model for each optimizer
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=optimizer(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    # Append the results to the DataFrame
    new_row = {'Optimizer': optimizer.__name__, 'Test Loss': score[0], 'Test Accuracy': score[1]}
    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

# Display the results
print(results)


# In[57]:


# mnist_cnn.py + different regularization methods
batch_size = 128
num_classes = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype ('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

regularization_methods = ['None', 'L1', 'L2', 'Dropout']

# Create a DataFrame to store results
results = pd.DataFrame(columns=['Regularization Method', 'Test Loss', 'Test Accuracy'])

for reg_method in regularization_methods:
    # Create a new model for each regularization method
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape, kernel_regularizer=None))
    
    if reg_method == 'L1':
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    elif reg_method == 'L2':
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    else:
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=None))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    if reg_method == 'Dropout':
        model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    # Append the results to the DataFrame
    new_row = {'Regularization Method': reg_method, 'Test Loss': score[0], 'Test Accuracy': score[1]}
    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

# Display the results
print(results)


# In[60]:


# mnist_cnn.py + different hyperparameters
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# Define hyperparameter values to test
learning_rates = [0.00001, 0.1]
momentum_values = [0, 0.9]
epsilon_values = [1e-8, 1e-4]
nesterov_values = [False, True]
batch_sizes = [64, 256]
optimizers = [keras.optimizers.Adam, keras.optimizers.SGD, keras.optimizers.RMSprop]

# Create a DataFrame to store results
results = pd.DataFrame(columns=['Learning Rate', 'Momentum', 'Epsilon', 'Nesterov', 'Batch Size', 'Regularization', 'Test Loss', 'Test Accuracy'])

# Test different hyperparameters
for lr, momentum, epsilon, nesterov, batch_size, reg_method in product(learning_rates, momentum_values, epsilon_values, nesterov_values, batch_sizes, regularization_methods):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), kernel_regularizer=None))
    
    if reg_method == 'L1':
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l1(0.01)))
    elif reg_method == 'L2':
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=regularizers.l2(0.01)))
    else:
        model.add(Conv2D(64, (3, 3), activation='relu', kernel_regularizer=None))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))

    if reg_method == 'Dropout':
        model.add(Dropout(0.5))
    
    model.add(Dense(num_classes, activation='softmax'))

    optimizer = keras.optimizers.Adadelta(learning_rate=lr, rho=momentum, epsilon=epsilon)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=optimizer, metrics=['accuracy'])
    
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=0, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=0)

    # Append the results to the DataFrame
    new_row = {'Learning Rate': lr, 'Momentum': momentum, 'Epsilon': epsilon, 'Nesterov': nesterov, 'Batch Size': batch_size, 'Regularization': reg_method, 'Test Loss': score[0], 'Test Accuracy': score[1]}
    results = pd.concat([results, pd.DataFrame([new_row])], ignore_index=True)

# Display the results
pd.set_option('display.max_rows', None)  # To display all rows
pd.set_option('display.max_columns', None)  # To display all columns 
best_results = results.sort_values(by='Test Accuracy', ascending=False).head(5)
print("Top 5 Best Performing Configurations:")
print(best_results)
print(results)


# # Parameters results:
# - Initialization method: HeNormal
# - Activation function: linear
# - Optimizer: Adam
# - Regulazation technique: L2
# - -----------------------------------------
# # Top 5 Best Performing Configurations:
#         Learning Rate Momentum  Epsilon Nesterov Batch Size Regularization
#     115            0.1      0.9   0.0001    False         64        Dropout   
#     123            0.1      0.9   0.0001     True         64        Dropout   
#     88             0.1        0   0.0001     True         64           None   
#     83             0.1        0   0.0001    False         64        Dropout   
#     82             0.1        0   0.0001    False         64             L2   
# 
# 
#      Test Loss  Test Accuracy  
#     115   0.028825         0.9912  
#     123   0.029777         0.9910  
#     88    0.036239         0.9899  
#     83    0.031306         0.9898  
#     82    0.040284         0.9897 

# # CIFAR-10
# 
# ## MLP
#     Optimizer  Learning Rate Momentum       Epsilon Nesterov Batch Size  \
#       52       SGD            0.1        0  1.000000e-04    False         64   
#       60       SGD            0.1      0.9  1.000000e-04    False         64   
#       48       SGD            0.1        0  1.000000e-08    False         64   

# In[10]:


# MLP with BEST PARAMETERS

# Set the best-performing parameters
batch_size = 128
num_classes = 10
epochs = 20
initializer = tf.keras.initializers.GlorotUniform()
activation = 'relu'
optimizer = SGD()
dropout_rate = 0.2

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to integers (no one-hot encoding is needed)
y_train = y_train.reshape(-1)
y_test = y_test.reshape(-1)

# Create the MLP model
model = Sequential([
    Flatten(input_shape=(32, 32, 3)),  # Flatten the 32x32x3 image
    Dense(512, activation=activation, kernel_initializer=initializer),
    Dropout(dropout_rate),
    Dense(512, activation=activation, kernel_initializer=initializer),
    Dropout(dropout_rate),
    Dense(num_classes, activation='softmax')  # 10 output classes for CIFAR-10
])

# Compile the model
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',  # Use sparse categorical cross-entropy for label format
              metrics=['accuracy'])

# Print a summary of the model's architecture
model.summary()

# Train the model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy:.2%}')


# In[11]:


# MLP with BEST HYPERPARAMETERS

# Define the best-performing hyperparameters
hyperparameter_configs = [
    {
        'learning_rate': 0.1,
        'momentum': 0,
        'nesterov': False
    },
    {
        'learning_rate': 0.1,
        'momentum': 0.9,
        'nesterov': False
    },
    {
        'learning_rate': 0.1,
        'momentum': 0,
        'nesterov': False
    }
]

results = []

for config in hyperparameter_configs:
    batch_size = 64
    num_classes = 10
    epochs = 20
    initializer = tf.keras.initializers.GlorotUniform()
    activation = 'relu'
    dropout_rate = 0.2

    # Load and preprocess the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    # Create the MLP model
    model = Sequential([
        Flatten(input_shape=(32, 32, 3)),
        Dense(512, activation=activation, kernel_initializer=initializer),
        Dropout(dropout_rate),
        Dense(512, activation=activation, kernel_initializer=initializer),
        Dropout(dropout_rate),
        Dense(num_classes, activation='softmax')
    ])

    optimizer = SGD(learning_rate=config['learning_rate'], momentum=config['momentum'], nesterov=config['nesterov'])

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    # Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(x_test, y_test)
    results.append(accuracy)

# Print the accuracy for all three models
for i, accuracy in enumerate(results):
    print(f'Model {i + 1} - Test accuracy: {accuracy:.2%}')



# # CNN 

# In[12]:


# CNN with BEST PARAMETERS

# Set the best-performing hyperparameters
batch_size = 128
num_classes = 10
epochs = 12
initializer = tf.keras.initializers.HeNormal()
activation = 'linear'
regularization = l2(0.01)

# Load and preprocess the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Convert labels to one-hot encoded vectors
y_train = tf.keras.utils.to_categorical(y_train, num_classes)
y_test = tf.keras.utils.to_categorical(y_test, num_classes)

# Create the CNN model
model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation=activation, input_shape=(32, 32, 3), kernel_initializer=initializer, kernel_regularizer=regularization),
    Conv2D(64, (3, 3), activation=activation, kernel_initializer=initializer, kernel_regularizer=regularization),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation=activation, kernel_initializer=initializer, kernel_regularizer=regularization),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model with the specified optimizer
optimizer = Adam()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Print a summary of the model's architecture
model.summary()

# Train the model
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

# Evaluate the model on the test dataset
loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy:.2%}')


# In[22]:


# CNN with BEST HYPERPARAMETERS

# Define a list of hyperparameter combinations to test
hyperparameter_configs = [
    {'learning_rate': 0.1, 'momentum': 0.9, 'epsilon': 0.0001, 'nesterov': False, 'batch_size': 64, 'regularization': 'Dropout'},
    {'learning_rate': 0.1, 'momentum': 0.9, 'epsilon': 0.0001, 'nesterov': True, 'batch_size': 64, 'regularization': 'Dropout'},
    {'learning_rate': 0.1, 'momentum': 0, 'epsilon': 0.0001, 'nesterov': True, 'batch_size': 64, 'regularization': 'None'}
]

results = []

for config in hyperparameter_configs:
    # Extract hyperparameters
    learning_rate = config['learning_rate']
    momentum = config['momentum']
    epsilon = config['epsilon']
    nesterov = config['nesterov']
    batch_size = config['batch_size']
    regularization = config['regularization']

    # Load and preprocess the CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Convert labels to one-hot encoded vectors
    num_classes = 10
    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes)

    # Create the CNN model
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    # Define the optimizer based on the specified hyperparameters
    if nesterov:
        optimizer = SGD(learning_rate=learning_rate, momentum=momentum, nesterov=True)
    else:
        optimizer = Adam(learning_rate=learning_rate, epsilon=epsilon)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Print a summary of the model's architecture
    model.summary()

    # Train the model
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(x_test, y_test))

    # Evaluate the model on the test dataset
    loss, accuracy = model.evaluate(x_test, y_test)
    results.append({'Configuration': config, 'Test Accuracy': accuracy})

# Print results for all combinations
for result in results:
    print(f'Configuration: {result["Configuration"]}')
    print(f'Test accuracy: {result["Test Accuracy"]:.2%}')


# In[25]:


# Set random seeds for reproducibility  SGD 12 EPOCHS - BEST PEFORMING HYPER PARAM ON CIFAR
np.random.seed(42)
tf.random.set_seed(42)

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the CNN model
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Create the SGD optimizer with specified hyperparameters (omitting epsilon)
optimizer = SGD(learning_rate=0.1, momentum=0.0, nesterov=True)

# Compile the model with the customized optimizer
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=12, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


# In[31]:


# Set random seeds for reproducibility 24 EPOCHS
np.random.seed(42)
tf.random.set_seed(42)

# Load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train, x_test = x_train / 255.0, x_test / 255.0

# Define the CNN model
model = keras.Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# Create the SGD optimizer with specified hyperparameters (omitting epsilon)
optimizer = SGD(learning_rate=0.1, momentum=0.0, nesterov=True)

# Compile the model with the customized optimizer
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=24, batch_size=64, validation_data=(x_test, y_test))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


# 
