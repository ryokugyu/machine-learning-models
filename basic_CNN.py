'''
Implementing a simple Convolution Neural Network aka CNN
Adapated from this article: https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5
Dataset: MNIST
'''
import tensorflow as tf
from keras.dataset import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten

# downloading and splitting the dataset into test and train network
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#checking the image shape
print(X_train[0].shape)

#reshaping data to fit model
X_train = X_train.reshape(60000,28,28,1)
X_test = X_test.reshape(10000,28,28,1)

# one-hot encode target column
#This means that a column will be created for each output category and a binary
#variable is inputted for each category. For example, we saw that the first
#image in the dataset is a 5. This means that the sixth number in our array
#will have a 1 and the rest of the array will be filled with 0.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Check what we get from the to_categorical function
print(y_train[0])

#Building the model layer by layer
model = Sequential()

#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

#Compiling the model takes three parameters: optimizer, loss and metrics.
model.compile(optimizer='adam', loss='categorical_crossentropy')

model.fit(X_train, y_train, validation_data=(X_test, y_test),epochs=3)

#predict first 4 images in the test set
model.predict(X_test[:4])

