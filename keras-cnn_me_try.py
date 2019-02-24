import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib.image as mplimg
from matplotlib.pyplot import imshow
#label encoder will give the an array of all the
#classes value present
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras import layers
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D
from keras.layers import AveragePooling2D, MaxPooling2D, Dropout
from keras.models import Model
import keras.backend as K
from keras.models import Sequential
import warnings
warnings.simplefilter("ignore", category=DeprecationWarning)

os.listdir("../input/")
train_df=pd.read_csv("../input/train.csv")
train_df.head()

def prepareImages(data,m,dataset):
	X_train=np.zeros((m,100,100,3))
	count=0
	for fig in data['Image']:
		img=Image.load_img("../input/"+dataset+"/"+fig,target_size=(100,100,3))
		x=image.img_to_array(img)
		x=preprocess_input(x)
		X_train[count]=x
		if count%500==0:
			print("processed"+str(count)+"images")
		count=count+1
	return X_train

def prepare_labels(y):
	values=np.array(y)
	label_encoder=LabelEncoder()
	integer_encoded=label_encoder.fit_transform(values)
	onehot_encoder=OneHotEncoder(sparse=False)
	integer_encoded=integer_encoded.reshape(len(integer_encoded),1)
	onehot_encoded=onehot_encoder.fit_transform(integer_encoded)
	y=onehot_encoded
	return y,label_encoder

#label encoder will give 0,1,2...values
#onehot will give in terms of 1's and 0's

#onehot better than label coz label will
#misunderstand the problem as 2>1>0

X=prepareImages(train_df,train_df.shape[0],"train")
X=X/255

y,label_encoder=prepare_labels(train_df['Id'])

model = Sequential()

model.add(Conv2D(32, (7, 7), strides = (1, 1), name = 'conv0', input_shape = (100, 100, 3)))

model.add(BatchNormalization(axis = 3, name = 'bn0'))
model.add(Activation('relu'))

model.add(MaxPooling2D((2, 2), name='max_pool'))
model.add(Conv2D(64, (3, 3), strides = (1,1), name="conv1"))
model.add(Activation('relu'))
model.add(AveragePooling2D((3, 3), name='avg_pool'))

model.add(Flatten())
model.add(Dense(500, activation="relu", name='rl'))
model.add(Dropout(0.8))
model.add(Dense(y.shape[1], activation='softmax', name='sm'))

model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
model.summary()

history = model.fit(X, y, epochs=100, batch_size=100, verbose=1)
gc.collect()

plt.plot(history.history['acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.show()

print("training ends")


