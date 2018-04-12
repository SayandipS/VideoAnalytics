from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D , Cropping2D
from keras.layers import Activation, Dropout, Flatten, Dense,Lambda
from keras.preprocessing.image import ImageDataGenerator
import keras.utils as kutils
from sklearn.utils import shuffle
import numpy as np
import glob
import os
import imageio as ios 
import string as str  


 
#Define Model
model = Sequential()

#Preprocess Layer
#model.add(Cropping2D(cropping=( (72,22),(0,0) ), input_shape=(480,704,3) ))
#model.add(Lambda(lambda x:x / 127.5 - 1.))

#Convolution Layer 1
model.add(Conv2D(32,5,5, subsample=(2,2), dim_ordering='tf' , input_shape=(480,704,3)))
model.add(Activation('relu'))
model.add(Lambda(lambda x:x / 127.5 - 1.))

#Convolution Layer 2
model.add(Conv2D(24,5,5, subsample =(2,2), border_mode = 'valid'))
model.add(Activation('relu'))

#Convolution Layer 3
model.add(Conv2D(36,5,5, subsample =(2,2), border_mode = 'valid'))
model.add(Activation('relu'))

#Convolution Layer 4
model.add(Conv2D(48,3,3, subsample =(1,1), border_mode = 'valid'))
model.add(Activation('relu'))

#Convolution Layer 5
model.add(Conv2D(64,3,3, subsample =(1,1), border_mode = 'valid'))
model.add(Activation('relu'))

#Flatten
model.add(Flatten())

#Dense Layer 1
model.add(Dense(100, activation='relu'))
model.add(Dropout(.5))

#Dense Layer 2
model.add(Dense(50, activation='relu'))
model.add(Dropout(.4))

#Dense Layer 3
model.add(Dense(10, activation='relu'))

#Output Layer
model.add(Dense(4,activation='softmax'))

# Display model summary
print("Model summary")
print(model.summary())
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy']) 



def generator():  
 y_list=[]  
 b=np.empty((0,480,704,3))
 i=0  
 os.chdir("/home/sayandip/img/all_by_name")
 while 1:
  for file in glob.glob("*vid*.jpg"):
     x = ios.imread(file)
     a=np.expand_dims(x,axis=0)
     b=np.append(b,a,axis=0)
     i+=1
     #print("Loop:",i,",",file)
     if "customer_interaction" in file :
        y_list.append(0)
     elif "delivering_coffee" in file :
         y_list.append(1)
     elif "not_in_frame" in file :
        y_list.append(2)  
     elif "working_pos_vid" in file :
        y_list.append(3)
     else :          
        y_list.append(2)
     #print("for",i, "y_list",y_list)       
     train_X=b
#print(shape(train_X))
     train_y=np.array(y_list)
#print("Type of train_y ", type(train_y))
#print(train_y)
     ret_gen=(train_X,train_y)
     yield (ret_gen)
#print(ret_gen)
#train_y=np.array([0,0,0,1,1,1,2,2,2,3,3,3])
#train_y=kutils.to_categorical(train_y, nb_classes=4)



training_generator = generator()

#Split input data into 80% train and 20% validation 
#train_data, validation_data = train_test_split(raw_data, test_size=0.2)
      
#model.fit_generator(training_generator, samples_per_epoch=42 , nb_epoch=10, verbose=2)
history=model.fit_generator(training_generator,samples_per_epoch=40, nb_epoch=200)
model.save("/home/sayandip/model_sbx.h5")

 
