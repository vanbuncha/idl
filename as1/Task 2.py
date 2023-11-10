#!/usr/bin/env python
# coding: utf-8

# In[73]:


import numpy as np
from tensorflow import keras
from tensorflow.keras import Input, layers, Model
from sklearn.model_selection import train_test_split
import tensorflow as tf


# In[4]:


labels = np.load("resources/labels.npy")
data = np.load("resources/images.npy")


# In[69]:


def map_minutes(labels):
    max_categories = 720
    labels_dict = {}
    unique_counter = 0
    
    for i in range(len(labels)):
        if str(labels[i]) not in labels_dict:
            labels_dict.update({str(labels[i]):unique_counter})
            unique_counter += 1
            
    return labels_dict


# In[70]:


def split_categories(labels, labels_dict, num_categories):
    max_categories = 720
    split_interval = max_categories / num_categories
    split_labels_dict = {}
    counter = -1
    
    for label in labels:
        if labels_dict[str(label)] % split_interval == 0 and str(label) not in split_labels_dict:
            counter += 1
            
        split_labels_dict.update({str(label):counter})
    
    numeric_labels = []
    
    for label in labels:
        numeric_labels.append(split_labels_dict[str(label)])
    
    return split_labels_dict, np.asarray(numeric_labels)
    


# In[62]:


max_categories = 720
num_categories = 360

unique_labels_dict = map_minutes(labels)
labels_dict, y = split_categories(labels, unique_labels_dict, num_categories)

X = data


# In[63]:


in_shape = (75, 75, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[64]:


X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

print("Shape of X_train: ", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")


# In[65]:


y_train = keras.utils.to_categorical(y_train, num_categories)
y_test = keras.utils.to_categorical(y_test, num_categories)

print("Shape of y_train: ", y_train.shape)


# In[ ]:





# In[66]:


model = keras.Sequential(
    [
        keras.Input(shape = in_shape),
        layers.Conv2D(32, kernel_size=(4,4), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(4,4), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(5,5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_categories, activation="softmax")
    ]
)
#Best for 24 classes
#  keras.Input(shape = in_shape),
#         layers.Conv2D(32, kernel_size=(4,4), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Conv2D(64, kernel_size=(4,4), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Conv2D(64, kernel_size=(5,5), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_categories, activation="softmax")

#Best for 48 classes
#     keras.Input(shape = in_shape),
#         layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Conv2D(64, kernel_size=(4,4), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Flatten(),
#         layers.Dropout(0.5),
#         layers.Dense(num_categories, activation="softmax")

#Best for 96, 192 classes
#         keras.Input(shape = in_shape),
#         layers.Conv2D(32, kernel_size=(4,4), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Conv2D(64, kernel_size=(4,4), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Conv2D(64, kernel_size=(5,5), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Flatten(),
#         layers.Dense(128, activation="relu"),
#         layers.Dropout(0.5),
#         layers.Dense(num_categories, activation="softmax")

model.summary()


# In[71]:


def common_sense_accuracy_classification(y_true, y_pred):
    
    #Obtain a tensor filled with the predicted categories, instead of the probability distrubutions provided by softmax
    y_pred_cats = tf.cast(tf.argmax(y_pred, axis=1), tf.float32)
    y_true_cats = tf.cast(tf.argmax(y_true, axis=1), tf.float32)

    #Seperate the higher and lower values within the true and the pred. 
    min_elements = tf.math.minimum(y_pred_cats, y_true_cats)
    max_elements = tf.math.maximum(y_pred_cats, y_true_cats)
    
    #Standard difference = abs(min-max)
    standard_difference = tf.math.abs(tf.math.subtract(max_elements, min_elements))
    
    #Circular difference = number of classes - standard difference
    circular_difference = tf.math.abs(tf.subtract(standard_difference, num_categories))
    
    
    #Select the lowest value between the differences to determine common sense error
    common_sense_error = tf.math.minimum(standard_difference, circular_difference)
    
    #Convert the common sense error into minutes. Since the exact time is not known, the time distance between categories is taken to be the length of time in a category
    common_sense_error = tf.multiply(common_sense_error, 720/num_categories)

    return common_sense_error


# In[68]:


batch_size = 256
epochs = 100

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy", common_sense_accuracy_classification])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[69]:


score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("Test Common Sense Error", score[2])


# # Regression

# In[20]:


def map_regression(data):
    new_data = []
    for item in data:
        decimal = int(item[0]) + (int(item[1])/60)
        new_data.append(decimal)
        
    return new_data


num_categories = 1

X = data
y = np.array(map_regression(labels))

in_shape = (75, 75, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

print("Shape of X_train: ", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")


# In[39]:


def common_sense_accuracy_regression(y_true, y_pred):

    #Seperate the higher and lower values within the true and the pred. 
    min_elements = tf.math.minimum(y_pred, y_true)
    max_elements = tf.math.maximum(y_pred, y_true)
    
    max_value = 12 + (59/60)
    
    #Standard difference = abs(min-max)
    standard_difference = tf.math.abs(tf.math.subtract(max_elements, min_elements))
    
    #Circular difference = number of classes - standard difference
    circular_difference = tf.math.abs(tf.subtract(standard_difference, max_value))
    
    
    #Select the lowest value between the differences to determine common sense error
    common_sense_error = tf.math.minimum(standard_difference, circular_difference)
    
    #Convert the common sense error into minutes. Since the exact time is not known, the time distance between categories is taken to be the length of time in a category
    common_sense_error = tf.multiply(common_sense_error, 60)

    return common_sense_error


# In[66]:


model = keras.Sequential(
[
        keras.Input(shape = in_shape),
        layers.Conv2D(32, kernel_size=(4,4), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(5,5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Conv2D(64, kernel_size=(5,5), activation="relu"),
        layers.MaxPooling2D(pool_size=(2,2)),
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_categories, activation="linear")
    ]
)

model.summary()

#Best:
# [
#         keras.Input(shape = in_shape),
#         layers.Conv2D(32, kernel_size=(4,4), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Conv2D(64, kernel_size=(5,5), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Conv2D(64, kernel_size=(5,5), activation="relu"),
#         layers.MaxPooling2D(pool_size=(2,2)),
#         layers.Flatten(),
#         layers.Dense(256, activation="relu"),
#         layers.Dense(256, activation="relu"),
#         layers.Dense(128, activation="relu"),
# layers.Dense(128, activation="relu"),
# layers.Dense(128, activation="relu"),
#         layers.Dropout(0.5),
#         layers.Dense(num_categories, activation="linear")
#     ]


# In[67]:


batch_size = 256
epochs = 100

model.compile(loss="mean_squared_error", optimizer="adam", metrics=["accuracy", common_sense_accuracy_regression])

model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)


# In[68]:


score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
print("Test Common Sense Error", score[2])


# # Multi-Head

# In[72]:


max_categories = 720
num_categories = 192

unique_labels_dict = map_minutes(labels)
labels_dict, y_class = split_categories(labels, unique_labels_dict, num_categories)
y_reg = np.array(map_regression(labels))


X = data

in_shape = (75, 75, 1)

X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(X, y_class, y_reg, test_size=0.2)

X_train = np.array(X_train, dtype=np.float32)
X_test = np.array(X_test, dtype=np.float32)

X_train = X_train.astype("float32") / 255
X_test = X_test.astype("float32") / 255

X_train = np.expand_dims(X_train, -1)
X_test = np.expand_dims(X_test, -1)

print("Shape of X_train: ", X_train.shape)
print(X_train.shape[0], "train samples")
print(X_test.shape[0], "test samples")

y_class_train = keras.utils.to_categorical(y_class_train, num_categories)
y_class_test = keras.utils.to_categorical(y_class_test, num_categories)

print("Shape of y_train: ", y_class_train.shape)


# In[90]:


input_layer = Input(shape=in_shape, name="input_layer")
conv1 = layers.Conv2D(32, kernel_size=(4,4), activation="relu", name="conv1")(input_layer)
pool1 = layers.MaxPooling2D(pool_size=(2,2), name="pool1")(conv1)
conv2 = layers.Conv2D(64, kernel_size=(4,4), activation="relu", name="conv2")(pool1)
pool2 = layers.MaxPooling2D(pool_size=(2,2), name="pool2")(conv2)
conv3 = layers.Conv2D(64, kernel_size=(5,5), activation="relu", name="conv3")(pool2)
pool3 = layers.MaxPooling2D(pool_size=(2,2), name="pool3")(conv3)

#Class Head
class_flatten = layers.Flatten(name="class_flatten")(pool3)
class_dense1 = layers.Dense(128, activation="relu", name="class_dense1")(class_flatten)
class_drop = layers.Dropout(0.5, name="class_drop")(class_dense1)
class_out = layers.Dense(num_categories, activation="softmax", name="class_out")(class_drop)

#Reg Head
reg_flatten = layers.Flatten(name="reg_flatten")(pool3)
reg_dense1 = layers.Dense(256, activation="relu", name="reg_dense1")(reg_flatten)
reg_dense2 = layers.Dense(256, activation="relu", name="reg_dense2")(reg_dense1)
reg_dense3 = layers.Dense(128, activation="relu", name="reg_dense3")(reg_dense2)
reg_dense4 = layers.Dense(128, activation="relu", name="reg_dense4")(reg_dense3)
reg_dense5 = layers.Dense(128, activation="relu", name="reg_dense5")(reg_dense4)
reg_drop = layers.Dropout(0.5, name="reg_drop")(reg_dense5)
reg_out = layers.Dense(1, activation="linear", name="reg_out")(reg_drop)

model = Model(inputs=input_layer, outputs=[class_out, reg_out])
model.summary()


# In[93]:


model.compile(optimizer="adam", loss={"class_out":"categorical_crossentropy", "reg_out":"mean_squared_error"}, metrics={"class_out":[common_sense_accuracy_classification], "reg_out":[common_sense_accuracy_regression]})

epochs=100
batch_size = 128

model.fit({"input_layer":X_train}, {"class_out": y_class_train, "reg_out": y_reg_train}, epochs=epochs, batch_size=batch_size)


# In[94]:


score = model.evaluate({"input_layer":X_test},{"class_out":y_class_test, "reg_out": y_reg_test}, verbose=0)
print("Loss", score[0])
print("Classification Common Sense Accuracy: ", score[1])
print("Regression Common Sense Accuracy: ", score[2])


# In[ ]:




