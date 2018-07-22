# Import packages
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
print("Matplotlib backend selected to allow plotting on AWS")
import os
import sklearn
from sklearn.model_selection import train_test_split
from keras.regularizers import l2, activity_l2
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Dropout, Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras import optimizers
from random import shuffle

#Load samples
samples = []
with open('./edata/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		samples.append(line)
samples = samples[1:]
#train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# Load lists of images and angles
images, angles = [], []
for row in samples:
	#print(row[0])
	center_name = row[0].split('/')[-1]
	left_name = row[1].split('/')[-1]
	right_name = row[2].split('/')[-1]
	#print(center_name)
	if 'center_2016' in center_name:
		center_image = cv2.imread('./data/IMG/' + center_name)
		center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
		#print(center_image)
	elif 'center_2017' in center_name:
		center_image = cv2.imread('./edata/IMG/' + center_name)
		center_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
		#print(center_image)

	center_angle = float(row[3])
	images.append(center_image)
	angles.append(center_angle)
	#we also flip the central image and steering
	# images.append(cv2.flip(center_image, 1))
	# angles.append(-1 * center_angle)

	if 'left_2016' in left_name:
		left_image = cv2.imread('./data/IMG/' + left_name)
		left_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
		#print(left_image)
	elif 'left_2017' in left_name:
		left_image = cv2.imread('./edata/IMG/' + left_name)
		left_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
		#print(left_image)
	left_angle = center_angle + 0.1
	images.append(left_image)
	angles.append(left_angle)
	
	if 'right_2016' in right_name:
		right_image = cv2.imread('./data/IMG/' + right_name)
		right_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
		#print(right_image)
	elif 'right_2017' in right_name:
		right_image = cv2.imread('./edata/IMG/' + right_name)
		right_image = cv2.cvtColor(center_image, cv2.COLOR_BGR2RGB)
		#print(right_image)
	right_angle = center_angle - 0.1
	images.append(right_image)
	angles.append(right_angle)

#Specify batch size and create generator to load data. Actually unnecessary as ended up using small number of images        	
batch_size=64
def generator(images, angles, batch_size):
	num_samples = len(samples) * 4
	while 1:
		shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			
			batch_images = images[offset:offset+batch_size]
			batch_angles = angles[offset:offset+batch_size]

			X = np.array(batch_images)
			y = np.array(batch_angles)
			X = np.resize(X,(batch_size,160,320,3))
			y = np.resize(y, (batch_size,1))
			#print(X[0,:,:,:])
			#print("Batch {}: X= {}, y= {}".format(offset//batch_size,X,y))

			yield sklearn.utils.shuffle(X, y)

#Create training and validation set
X_train, X_val, y_train, y_val = train_test_split(images, angles, test_size=0.2, random_state = 42)
train_generator = generator(X_train, y_train, batch_size)
validation_generator = generator(X_val, y_val, batch_size)
print("Training and validation generators prepared")

#Use Keras API to create CNN that maps images to steering angles
model = Sequential()
model.add(Lambda(lambda x: x/255.0 - 0.5, input_shape = (160,320,3) )) #normalise and mean centre  #160x320x3
model.add(Cropping2D(cropping=((70,20),(40,40)))) #crop reduncant pixels  #90x240x3                
model.add(Convolution2D(64,5,5,activation='relu')) 
model.add(MaxPooling2D())
model.add(Convolution2D(16,5,5,activation='relu')) 
model.add(MaxPooling2D())
model.add(Convolution2D(4,5,5,activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Flatten()) #change input shape to match result of cropping
model.add(Dense(100))
model.add(Dense(16))
model.add(Dense(1))
model.summary()

#previous attempt
# model.add(Convolution2D(24,5,5, subsample=(2,2), activation='elu')) 
# model.add(Dropout(0.1))
# model.add(Convolution2D(36,5,5, subsample=(2,2), activation='elu'))
# model.add(Dropout(0.1))
# model.add(Convolution2D(48,5,5, subsample=(2,2), activation='elu'))
# model.add(Dropout(0.1))
# model.add(Convolution2D(64,3,3, activation='elu'))
# model.add(Dropout(0.1))
# model.add(Convolution2D(64,3,3,activation='elu'))
# model.add(Dropout(0.1))
# model.add(Flatten()) #change input shape to match result of cropping
# model.add(Dense(100))
# model.add(Dense(50))
# model.add(Dense(10))
# model.add(Dense(1))



# Specify the loss and optimization technique to be used for training
model.compile(loss='mae', optimizer='adam')
# Specify callbacks to keep best model (lowest val_loss). Patience > epochs so no early stopping
callbacks = [EarlyStopping(monitor='val_loss', patience=100, verbose=1),
   	         ModelCheckpoint(filepath = 'load_model.h5', monitor='val_loss', save_best_only=True, verbose=1)]
# Load data onto graph and train the model
print("Training model...")
r=model.fit_generator(train_generator, samples_per_epoch = len(samples) * 4, validation_data = validation_generator, nb_val_samples = len(X_val), nb_epoch = 25, callbacks = callbacks)
print("Model saved to disc")

#Plot the training and validation loss (note we select a matplotlib backend at start of file so the image is correctly created on AWS)
#acc = r.history['acc']
#val_acc = r.history['val_acc']
loss = r.history['loss']
val_loss = r.history['val_loss']
epochs = range(len(loss))
plt.plot(epochs, loss, 'b', label = 'Training loss')
plt.plot(epochs, val_loss, 'r', label = 'Validatoin loss')
plt.title('Training and validation loss')
plt.legend()
plt.savefig('./training.png')
plt.show()
print("------------ALL DONE------------")
