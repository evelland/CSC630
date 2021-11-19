import numpy as np
from processing import *
from sklearn.model_selection import KFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # disables gpu training on available devices, system ram held priority over processing time

def define_model(): # layer definitions
	model = Sequential() # layers are sequential, each tensor is passed through all layers in order
	model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(100, 100, 1))) # breaks bitmap image into (3,3) kernels with 64 filters, images of shape (100,100)
	model.add(MaxPooling2D((2, 2))) # pools and converts kernels from (3,3) to (2,2)
	model.add(Flatten()) # flattens data to 1 dimension
	model.add(Dense(5000, activation='relu', kernel_initializer='he_uniform')) # densely-connected layer with 5000 nodes
	model.add(Dense(4037, activation='softmax')) # response layer, using softmax: determining a single output value
	opt = SGD(learning_rate=0.0001, momentum=0.9) # optimization learning parameters, standard learning rate and momentum values - best fit for this set based on my testing
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

def evaluate_model(dataX, dataY, n_folds=5): # builds model
	kfold = KFold(n_folds, shuffle=True, random_state=1) # divides data into 5 shuffled subsets for training
	for train_ix, test_ix in kfold.split(dataX): # iterates over each k-fold validation split
		model = define_model() # layer initialization
		trainX, trainY = dataX[train_ix], dataY[train_ix] # training data for split
		testX, testY = dataX[test_ix], dataY[test_ix] # testing data for split
		model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=1) # fits model for 10 epochs, low batch size of 32 to increase fit tightness on provided data
		_, acc = model.evaluate(testX, testY, verbose=1) # returns accuracy, verbose = 1 displays running processes
		print('> %.3f' % (acc * 100.0)) # prints per fold completion (5 total)

def run_test_harness(): # container for model processes
	trainX, trainY, testX, testY = collect_data() # pre-processing
	trainX = np.array(trainX) # converts lists to numpy arrays
	trainY = np.array(trainY)
	testX = np.array(testX)
	testY = np.array(testY)
	trainX = trainX.reshape(-1, 100, 100, 1) # divides traing data by all possible divisions (-1) that fit (100,100) with a color dimension of 1
	testX = testX.reshape(-1, 100, 100, 1)
	trainY = to_categorical(trainY) # converts response data to one-hot vectors
	testY = to_categorical(testY)
	print("training!!") # status update
	evaluate_model(trainX, trainY) # builds model and returns accuracy

run_test_harness()

#def evaluate_model(trainX, trainY, testX, testY):
#	model = define_model()
#	model.fit(trainX, trainY, epochs=10, batch_size=64, validation_data=(testX, testY), verbose=1)
#	_, acc = model.evaluate(testX, testY, verbose=1)
#	print('> %.3f' % (acc * 100.0))