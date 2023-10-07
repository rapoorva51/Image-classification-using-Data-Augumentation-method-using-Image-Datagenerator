#!/usr/bin/env python
# coding: utf-8

# # Image classification using Data Augumentation method using Image Datagenerator
# Checked the accuracy of 13 Pretrained models.

# In[1]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir(r"C:\Users\HP\Downloads\imgdataset\Multi-class Weather Dataset"))


# In[2]:


import keras
import cv2
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import VGG16
from keras.models import load_model,model_from_json
from keras.applications.vgg16 import preprocess_input


# In[3]:


import tensorflow as tf
import PIL


# In[4]:


path = r"C:\Users\HP\Downloads\imgdataset\Multi-class Weather Dataset"

train_datagen = ImageDataGenerator(rescale = 1./255,horizontal_flip = True,shear_range = 10,zoom_range = 0.2,preprocessing_function =preprocess_input )
train_generator = train_datagen.flow_from_directory(path+"/train",target_size = (224,224),shuffle = True,class_mode = 'categorical')

validation_datagen = ImageDataGenerator(rescale = 1./255,preprocessing_function = preprocess_input)
validation_generator = validation_datagen.flow_from_directory(path + "/validation",target_size = (224,224),shuffle = False,class_mode = 'categorical')


# # VGG16

# In[5]:


conv_base = VGG16(include_top = False,weights = 'imagenet')


# In[6]:


for layer in conv_base.layers:
    layer.trainable = False
X = conv_base.output
X = keras.layers.GlobalAveragePooling2D()(X)
X = keras.layers.Dense(128,activation = 'relu')(X)
output_1 = keras.layers.Dense(10,activation = 'softmax')(X)
model = keras.Model(inputs = conv_base.input,outputs = output_1)


# In[7]:


model.summary()


# In[8]:


model.compile(loss = 'categorical_crossentropy',optimizer = 'Adam', metrics = ['accuracy'])


# In[9]:


history = model.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)


# In[10]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 20))

plt.subplot(4,4, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Function')

plt.subplot(4, 4, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')


# In[11]:


model.evaluate_generator(train_generator,steps = len(validation_generator),verbose = 1)


# In[12]:


score = model.evaluate_generator(validation_generator)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])


# In[13]:


batch_size = 10
Y_pred1 = model.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred1 = np.argmax(Y_pred1, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm1 = confusion_matrix(validation_generator.classes, y_pred1)
print(cm1)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred1))


# In[14]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = model.to_json()
with open('vgg16_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
model.save_weights('model_vgg16.hdf5', overwrite=True)


# # Resnet50

# In[15]:


import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input
from keras import Model, layers
from keras.models import load_model, model_from_json
from keras.optimizers import *


# In[16]:


conv_base1 = ResNet50(
    include_top=True,
    weights='imagenet')

for layer in conv_base1.layers:
    layer.trainable = True


# In[17]:


from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D,Dropout

x1 = conv_base1.output

x1 = Flatten()(x1)
x1 = Dense(units=512, activation='relu')(x1)
x1 = Dropout(0.2)(x1)
x1 = Dense(units=512, activation='relu')(x1)
x1 = Dropout(0.2)(x1)
output1  = Dense(units=10, activation='softmax')(x1)
model1 = Model(conv_base1.input, output1)


model1.summary()


# In[18]:


loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model1.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])


# In[19]:


from tensorflow.keras.callbacks import ReduceLROnPlateau

lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)



 
import datetime
now = datetime.datetime.now
t1 = now()
transfer_learning_history1 = model1.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)
print('Training time: %s' % (now() - t1))


# In[20]:


# evaluate the performance the new model and report the results
score1 = model1.evaluate_generator(validation_generator)
print("Test Score:", score1[0])
print("Test Accuracy:", score1[1])


# In[21]:


import matplotlib.pyplot as plt
plt.plot(transfer_learning_history1.history['accuracy'])
plt.plot(transfer_learning_history1.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(transfer_learning_history1.history['loss'])
plt.plot(transfer_learning_history1.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[22]:


batch_size = 10
Y_pred2 = model1.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred2 = np.argmax(Y_pred2, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm2 = confusion_matrix(validation_generator.classes, y_pred2)
print(cm2)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred2))


# In[23]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = model1.to_json()
with open('resnet50_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
model1.save_weights('model_resnet50.hdf5', overwrite=True)


# # Resnet152V2

# In[24]:


import os
import tensorflow as tf #tf 2.0.0
import numpy as np


# In[25]:


ResNet_model = tf.keras.applications.ResNet152V2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))


# In[26]:


from tensorflow.keras import Model 
from tensorflow.keras.layers import Conv2D, Dense, MaxPooling2D, Dropout, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential

# The last 15 layers fine tune
for layer in ResNet_model.layers[:-15]:
    layer.trainable = False

x2 = ResNet_model.output
x2 = GlobalAveragePooling2D()(x2)
x2 = Flatten()(x2)
x2 = Dense(units=512, activation='relu')(x2)
x2 = Dropout(0.3)(x2)
x2 = Dense(units=512, activation='relu')(x2)
x2 = Dropout(0.3)(x2)
output2  = Dense(units=10, activation='softmax')(x2)
model2 = Model(ResNet_model.input, output2)


model2.summary()


# In[27]:


loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model2.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])


# In[28]:


from tensorflow.keras.callbacks import ReduceLROnPlateau

lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)



 
import datetime
now = datetime.datetime.now
t2 = now()
transfer_learning_history = model2.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)
print('Training time: %s' % (now() - t2))


# In[29]:


plt.plot(transfer_learning_history.history['accuracy'])
plt.plot(transfer_learning_history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(transfer_learning_history.history['loss'])
plt.plot(transfer_learning_history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[30]:


model2.evaluate(validation_generator)


# In[31]:


pred2=model2.predict(validation_generator)
predicted_class_indices=np.argmax(pred2,axis=1)
print("Pred",pred2)


# In[32]:


print("pci:",predicted_class_indices)


# In[33]:


batch_size = 10
Y_pred3 = model2.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred3 = np.argmax(Y_pred3, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm3 = confusion_matrix(validation_generator.classes, y_pred3)
print(cm3)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred3))


# In[34]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = model2.to_json()
with open('resnet152_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
model2.save_weights('model_resnet152.hdf5', overwrite=True)


# # VGG19 

# In[35]:


from keras.applications import VGG19
from keras.models import load_model,model_from_json
from keras.applications.vgg19 import preprocess_input


# In[36]:


conv_base3 = VGG19(include_top = False,weights = 'imagenet')


# In[37]:


for layer in conv_base3.layers:
    layer.trainable = False  
    
X3 = conv_base3.output
X3 = keras.layers.GlobalAveragePooling2D()(X3)
X3 = keras.layers.Dense(128,activation = 'relu')(X3)
output_3 = keras.layers.Dense(10,activation = 'softmax')(X3)
model3 = keras.Model(inputs = conv_base3.input,outputs = output_3)


# In[38]:


model3.summary()


# In[39]:


model3.compile(loss = 'categorical_crossentropy',optimizer = 'Adam', metrics = ['accuracy'])


# In[40]:


history3 = model3.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)


# In[41]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 20))

plt.subplot(4, 4, 1)
plt.plot(history3.history['loss'], label='Training Loss')
plt.plot(history3.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Function')

plt.subplot(4, 4, 2)
plt.plot(history3.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')


# In[42]:


model3.evaluate_generator(train_generator,steps = len(validation_generator),verbose = 1)


# In[43]:


score3 = model3.evaluate_generator(validation_generator)
print("Test Score:", score3[0])
print("Test Accuracy:", score3[1])


# In[44]:


batch_size = 10
Y_pred4 = model3.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred4 = np.argmax(Y_pred4, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm4 = confusion_matrix(validation_generator.classes, y_pred4)
print(cm4)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred4))


# In[45]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = model3.to_json()
with open('vgg19_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
model3.save_weights('model_vgg19.hdf5', overwrite=True)


# # Inception V3

# In[46]:


from tensorflow.keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3  import preprocess_input


# In[47]:


conv_base4 = InceptionV3(include_top = False,weights = 'imagenet')


# In[48]:


for layer in conv_base4.layers:
    layer.trainable = False
    
X4 = conv_base4.output
X4 = keras.layers.GlobalAveragePooling2D()(X4)
X4 = keras.layers.Dense(128,activation = 'relu')(X4)
output_4 = keras.layers.Dense(10,activation = 'softmax')(X4)
model4 = keras.Model(inputs = conv_base4.input,outputs = output_4)


# In[49]:


model4.summary()


# In[50]:


model4.compile(loss = 'categorical_crossentropy',optimizer = 'Adam', metrics = ['accuracy'])


# In[51]:


history4 = model4.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)


# In[52]:


import matplotlib.pyplot as plt
plt.figure(figsize=(20, 20))

plt.subplot(4, 4, 1)
plt.plot(history4.history['loss'], label='Training Loss')
plt.plot(history4.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss Function')

plt.subplot(4, 4, 2)
plt.plot(history4.history['accuracy'], label='Training Accuracy')
plt.plot(history4.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')


# In[53]:


model4.evaluate_generator(train_generator,steps = len(validation_generator),verbose = 1)


# In[54]:


score4 = model4.evaluate_generator(validation_generator)
print("Test Score:", score4[0])
print("Test Accuracy:", score4[1])


# In[55]:


batch_size = 10
Y_pred5 = model4.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred5 = np.argmax(Y_pred5, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm5 = confusion_matrix(validation_generator.classes, y_pred5)
print(cm5)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred5))


# In[56]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = model4.to_json()
with open('inceptionv3_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
model4.save_weights('model_inceptionv3.hdf5', overwrite=True)


# In[57]:


# Efficient Net


# In[58]:


from tensorflow.keras.applications import EfficientNetB7


# In[59]:


conv_base5 = EfficientNetB7(include_top = False,weights = 'imagenet')


# In[60]:


for layer in conv_base5.layers:
    layer.trainable = False
X5 = conv_base5.output
X5 = GlobalAveragePooling2D()(X5)
X5 = Flatten()(X5)
X5 = Dense(units=128, activation='relu')(X5)
X5 = Dropout(0.2)(X5)
X5 = Dense(units=128, activation='relu')(X5)
X5 = Dropout(0.2)(X5)
output5  = Dense(units=10, activation='softmax')(X5)
model5 = Model(conv_base5.input, output5)


model5.summary()


# In[61]:


loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model5.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])


# In[62]:


from tensorflow.keras.callbacks import ReduceLROnPlateau

lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)



 
import datetime
now = datetime.datetime.now
t5 = now()
history5 = model5.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)
print('Training time: %s' % (now() - t5))


# In[63]:


import matplotlib.pyplot as plt
plt.plot(history5.history['accuracy'])
plt.plot(history5.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history5.history['loss'])
plt.plot(history5.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[64]:


model5.evaluate_generator(train_generator,steps = len(validation_generator),verbose = 1)


# In[65]:


score5 = model5.evaluate_generator(validation_generator)
print("Test Score:", score5[0])
print("Test Accuracy:", score5[1])


# In[66]:


batch_size = 10
Y_pred6 = model5.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred6 = np.argmax(Y_pred6, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm6 = confusion_matrix(validation_generator.classes, y_pred6)
print(cm6)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred6))


# In[67]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = model5.to_json()
with open('efficientnet_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
model5.save_weights('model_efficientnet.hdf5', overwrite=True)


# # Mobile Net

# In[68]:


from tensorflow.keras.applications import MobileNetV2


# In[69]:


conv_base6 = MobileNetV2(include_top = False,weights = 'imagenet')


# In[70]:


for layer in conv_base6.layers:
    layer.trainable = False
X6 = conv_base6.output
X6 = GlobalAveragePooling2D()(X6)
X6 = Flatten()(X6)
X6 = Dense(units=128, activation='relu')(X6)
X6 = Dropout(0.2)(X6)
X6 = Dense(units=128, activation='relu')(X6)
X6 = Dropout(0.2)(X6)
output6  = Dense(units=10, activation='softmax')(X6)
model6 = Model(conv_base6.input, output6)


model6.summary()


# In[71]:


loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model6.compile(optimizer=optimizer, loss=loss, metrics= ['accuracy'])


# In[72]:


from tensorflow.keras.callbacks import ReduceLROnPlateau

lrr = ReduceLROnPlateau(monitor='val_accuracy', 
                        patience=3, 
                        verbose=1, 
                        factor=0.4, 
                        min_lr=0.0001)



 
import datetime
now = datetime.datetime.now
t6 = now()
history6 = model6.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)
print('Training time: %s' % (now() - t6))


# In[73]:


model6.evaluate_generator(train_generator,steps = len(validation_generator),verbose = 1)


# In[74]:


score6 = model6.evaluate_generator(validation_generator)
print("Test Score:", score6[0])
print("Test Accuracy:", score6[1])


# In[75]:


print(history6.history.keys())


# In[76]:


plt.plot(history6.history['accuracy'])
plt.plot(history6.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history6.history['loss'])
plt.plot(history6.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[77]:


batch_size = 10
Y_pred7 = model6.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred7 = np.argmax(Y_pred7, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm7 = confusion_matrix(validation_generator.classes, y_pred7)
print(cm7)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred7))


# In[78]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = model6.to_json()
with open('mobilenet_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
model6.save_weights('model_mobilenet.hdf5', overwrite=True)


# # Alexnet 

# In[79]:


import numpy as np
from keras import layers
from keras.layers import Input, Dense, Activation,BatchNormalization, Flatten, Conv2D, MaxPooling2D
from keras.models import Model
import keras.backend as K
K.set_image_data_format('channels_last')


# In[80]:


def AlexNet(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(5,(11,11),strides = 4,name="conv0")(X_input)
    X = BatchNormalization(axis = 3 , name = "bn0")(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max0')(X)
    
    X = Conv2D(10,(5,5),padding = 'same' , name = 'conv1')(X)
    X = BatchNormalization(axis = 3 ,name='bn1')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max1')(X)
    
    X = Conv2D(15, (3,3) , padding = 'same' , name='conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(20, (3,3) , padding = 'same' , name='conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(25, (3,3) , padding = 'same' , name='conv4')(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max2')(X)
    
    X = Flatten()(X)
    
    X = Dense(40, activation = 'relu', name = "fc0")(X)
    
    X = Dense(40, activation = 'relu', name = 'fc1')(X) 
    
    X = Dense(10,activation='softmax',name = 'fc2')(X)
    
    model7 = Model(inputs = X_input, outputs = X, name='AlexNet')
    return model7


# In[81]:


alex = AlexNet(train_generator[0][0].shape[1:])


# In[82]:


alex.summary()


# In[83]:


alex.compile(optimizer = 'SGD' , loss = 'categorical_crossentropy' , metrics=['accuracy'])


# In[84]:


history7 = alex.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)


# In[85]:


import matplotlib.pyplot as plt
plt.plot(history7.history['accuracy'])
plt.plot(history7.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history7.history['loss'])
plt.plot(history7.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[86]:


alex.evaluate_generator(train_generator,steps = len(validation_generator),verbose = 1)


# In[87]:


score7 = alex.evaluate_generator(validation_generator)
print("Test Score:", score7[0])
print("Test Accuracy:", score7[1])


# In[88]:


batch_size = 10
Y_pred8 = alex.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred8 = np.argmax(Y_pred8, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm8 = confusion_matrix(validation_generator.classes, y_pred8)
print(cm8)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred8))


# In[89]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = alex.to_json()
with open('alex_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
alex.save_weights('model_alex.hdf5', overwrite=True)


# # Google Net

# In[90]:


def GoogleNet(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(5,(11,11),strides = 4,name="conv0")(X_input)
    X = BatchNormalization(axis = 3 , name = "bn0")(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max0')(X)
    
    X = Conv2D(10,(5,5),padding = 'same' , name = 'conv1')(X)
    X = BatchNormalization(axis = 3 ,name='bn1')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max1')(X)
    
    X = Conv2D(15, (3,3) , padding = 'same' , name='conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(20, (3,3) , padding = 'same' , name='conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(25, (3,3) , padding = 'same' , name='conv4')(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max2')(X)
    
    X = Flatten()(X)
    
    X = Dense(40, activation = 'relu', name = "fc0")(X)
    
    X = Dense(40, activation = 'relu', name = 'fc1')(X) 
    
    X = Dense(10,activation='softmax',name = 'fc2')(X)
    
    model8 = Model(inputs = X_input, outputs = X, name='GoogleNet')
    return model8


# In[91]:


google = GoogleNet(train_generator[0][0].shape[1:])


# In[92]:


google.summary()


# In[93]:


google.compile(optimizer = 'SGD' , loss = 'categorical_crossentropy' , metrics=['accuracy'])


# In[94]:


history8 = google.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)


# In[95]:


import matplotlib.pyplot as plt
plt.plot(history8.history['accuracy'])
plt.plot(history8.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history8.history['loss'])
plt.plot(history8.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[96]:


google.evaluate_generator(train_generator,steps = len(validation_generator),verbose = 1)


# In[97]:


score8 = google.evaluate_generator(validation_generator)
print("Test Score:", score8[0])
print("Test Accuracy:", score8[1])


# In[98]:


batch_size = 10
Y_pred9 = google.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred9 = np.argmax(Y_pred9, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm9 = confusion_matrix(validation_generator.classes, y_pred9)
print(cm9)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred9))


# In[99]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = google.to_json()
with open('googlenet_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
google.save_weights('model_googlenet.hdf5', overwrite=True)


# # Caffe Net

# In[100]:


def CaffeNet(input_shape):
    
    X_input = Input(input_shape)
    
    X = Conv2D(5,(11,11),strides = 4,name="conv0")(X_input)
    X = BatchNormalization(axis = 3 , name = "bn0")(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max0')(X)
    
    X = Conv2D(10,(5,5),padding = 'same' , name = 'conv1')(X)
    X = BatchNormalization(axis = 3 ,name='bn1')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max1')(X)
    
    X = Conv2D(15, (3,3) , padding = 'same' , name='conv2')(X)
    X = BatchNormalization(axis = 3, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(20, (3,3) , padding = 'same' , name='conv3')(X)
    X = BatchNormalization(axis = 3, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    X = Conv2D(25, (3,3) , padding = 'same' , name='conv4')(X)
    X = BatchNormalization(axis = 3, name = 'bn4')(X)
    X = Activation('relu')(X)
    
    X = MaxPooling2D((3,3),strides = 2,name = 'max2')(X)
    
    X = Flatten()(X)
    
    X = Dense(40, activation = 'relu', name = "fc0")(X)
    
    X = Dense(40, activation = 'relu', name = 'fc1')(X) 
    
    X = Dense(10,activation='softmax',name = 'fc2')(X)
    
    model9 = Model(inputs = X_input, outputs = X, name='CaffeNet')
    return model9


# In[101]:


caffe = CaffeNet(train_generator[0][0].shape[1:])


# In[102]:


caffe.summary()


# In[103]:


caffe.compile(optimizer = 'SGD' , loss = 'categorical_crossentropy' , metrics=['accuracy'])


# In[104]:


history9 = caffe.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)


# In[105]:


import matplotlib.pyplot as plt
plt.plot(history9.history['accuracy'])
plt.plot(history9.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history9.history['loss'])
plt.plot(history9.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[106]:


caffe.evaluate_generator(train_generator,steps = len(validation_generator),verbose = 1)


# In[107]:


score9 = caffe.evaluate_generator(validation_generator)
print("Test Score:", score9[0])
print("Test Accuracy:", score9[1])


# In[108]:


batch_size = 10
Y_pred10 = caffe.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred10 = np.argmax(Y_pred10, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm10 = confusion_matrix(validation_generator.classes, y_pred10)
print(cm10)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred10))


# In[109]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = caffe.to_json()
with open('caffenet_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
google.save_weights('model_caffenet.hdf5', overwrite=True)


# # Dense Net

# In[110]:


from keras.applications import DenseNet121
from keras.models import load_model,model_from_json


# In[111]:


conv_base10 = DenseNet121(include_top = False,weights = 'imagenet')


# In[112]:


for layer in conv_base10.layers:
    layer.trainable = False
X10 = conv_base10.output
X10 = GlobalAveragePooling2D()(X10)
X10 = Flatten()(X10)
X10 = Dense(units=128, activation='relu')(X10)
X10 = Dropout(0.2)(X10)
X10 = Dense(units=128, activation='relu')(X10)
X10 = Dropout(0.2)(X10)
output10  = Dense(units=10, activation='softmax')(X10)
model10 = Model(conv_base10.input, output10)


model10.summary()


# In[113]:


model10.compile(loss = 'categorical_crossentropy',optimizer = 'Adam', metrics = ['accuracy'])


# In[114]:


history10 = model10.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)


# In[115]:


import matplotlib.pyplot as plt
plt.plot(history10.history['accuracy'])
plt.plot(history10.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history10.history['loss'])
plt.plot(history10.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[116]:


model10.evaluate_generator(train_generator,steps = len(validation_generator),verbose = 1)


# In[117]:


score10 = model10.evaluate_generator(validation_generator)
print("Test Score:", score10[0])
print("Test Accuracy:", score10[1])


# In[118]:


batch_size = 10
Y_pred11 = model10.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred11 = np.argmax(Y_pred11, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm11 = confusion_matrix(validation_generator.classes, y_pred11)
print(cm11)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred11))


# In[119]:


import pylab as pl
pl.matshow(cm11)
pl.title("Confusion Matrix of DenseNet121 model")
pl.show()


# In[120]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = model10.to_json()
with open('densenet121_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
model10.save_weights('model_densenet121.hdf5', overwrite=True)


# # Xception 

# In[121]:


from keras.applications import Xception
from keras.models import load_model,model_from_json


# In[122]:


conv_base11 = Xception(include_top = False,weights = 'imagenet')


# In[123]:


for layer in conv_base11.layers:
    layer.trainable = False  
    
X11 = conv_base11.output
X11 = keras.layers.GlobalAveragePooling2D()(X11)
X11 = keras.layers.Dense(128,activation = 'relu')(X11)
output_11 = keras.layers.Dense(10,activation = 'softmax')(X11)
model11 = keras.Model(inputs = conv_base11.input,outputs = output_11)


# In[124]:


model11.summary()


# In[125]:


model11.compile(loss = 'categorical_crossentropy',optimizer = 'Adam', metrics = ['accuracy'])


# In[126]:


history11 = model11.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)


# In[127]:


import matplotlib.pyplot as plt
plt.plot(history11.history['accuracy'])
plt.plot(history11.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history11.history['loss'])
plt.plot(history11.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[128]:


model11.evaluate_generator(train_generator,steps = len(validation_generator),verbose = 1)


# In[129]:


score11 = model11.evaluate_generator(validation_generator)
print("Test Score:", score11[0])
print("Test Accuracy:", score11[1])


# In[130]:


batch_size = 10
Y_pred12 = model11.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred12 = np.argmax(Y_pred12, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm12 = confusion_matrix(validation_generator.classes, y_pred12)
print(cm12)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred12))


# In[131]:


import pylab as pl
pl.matshow(cm12)
pl.title("Confusion Matrix of Xception model")
pl.show()


# In[132]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = model11.to_json()
with open('xception_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
model11.save_weights('model_xception.hdf5', overwrite=True)


# # NAS Net Mobile

# In[133]:


from keras.applications import NASNetMobile
from keras.models import load_model,model_from_json


# In[134]:


conv_base12 = NASNetMobile(include_top = False,weights = 'imagenet')


# In[135]:


for layer in conv_base12.layers:
    layer.trainable = False  
    
X12 = conv_base12.output
X12 = keras.layers.GlobalAveragePooling2D()(X12)
X12 = keras.layers.Dense(128,activation = 'relu')(X12)
output_12 = keras.layers.Dense(10,activation = 'softmax')(X12)
model12 = keras.Model(inputs = conv_base12.input,outputs = output_12)


# In[136]:


model12.summary()


# In[137]:


model12.compile(loss = 'categorical_crossentropy',optimizer = 'Adam', metrics = ['accuracy'])


# In[138]:


history12 = model12.fit_generator(train_generator,steps_per_epoch = 45,epochs = 5,verbose = 1, validation_data = validation_generator, validation_steps = 10)


# In[139]:


import matplotlib.pyplot as plt
plt.plot(history12.history['accuracy'])
plt.plot(history12.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history12.history['loss'])
plt.plot(history12.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[140]:


model12.evaluate_generator(train_generator,steps = len(validation_generator),verbose = 1)


# In[141]:


score12 = model12.evaluate_generator(validation_generator)
print("Test Score:", score12[0])
print("Test Accuracy:", score12[1])


# In[142]:


batch_size = 10
Y_pred13 = model12.predict_generator(validation_generator, 1400 // batch_size+1)
y_pred13 = np.argmax(Y_pred13, axis=1)

from sklearn.metrics import confusion_matrix,classification_report
print('Confusion Matrix')
cm13 = confusion_matrix(validation_generator.classes, y_pred13)
print(cm13)
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred13))


# In[143]:


import pylab as pl
pl.matshow(cm13)
pl.title("Confusion Matrix of NAS Net model")
pl.show()


# In[144]:


# save it as a h5 file
import tensorflow as tf
from tensorflow.keras.models import load_model

# save model architecture
model_json = model12.to_json()
with open('nasnet_model.json', 'w',encoding='utf8') as json_file:
    json_file.write(model_json)

# save model's learned weights
model12.save_weights('model_nasnet.hdf5', overwrite=True)


# In[145]:


train_generator.class_indices

