import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.image import load_img
import visualkeras
from PIL import Image
import tensorflow_datasets

print(tf.__version__)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def draw(n):
    plt.imshow(n,cmap=plt.cm.binary)
    plt.show()

mnist = tf.keras.datasets.mnist

(x_train,y_train) , (x_test,y_test) = mnist.load_data()
 
x_train = tf.keras.utils.normalize(x_train,axis=1)
x_test = tf.keras.utils.normalize(x_test,axis=1)

draw(x_train[0])

model = tf.keras.models.Sequential()
 
model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
#reshape

model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128,activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10,activation=tf.nn.softmax))

visualkeras.layered_view(model, legend=True)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
model.fit(x_train,y_train,epochs=10)


model.summary()

val_loss,val_acc = model.evaluate(x_test,y_test)
print("loss-> ",val_loss,"\nacc-> ",val_acc)

predictions=model.predict([x_test])
print('label -> ',y_test[2])
print('prediction -> ',np.argmax(predictions[2]))
 
draw(x_test[2])

#img = Image.open("6_test.png") #open image from file
#img = load_img("6_test_resized.png") #load image from fil

img = Image.open("9_test.png") #open image from file
img = img.resize((28, 28)) #resize image to 28x28
img = img.convert('L')
img = np.array(img) #load image into numpy array

img = np.invert(img).ravel()

img = img.reshape(28, 28) 
print(img.shape)
plt.imshow(img,cmap=plt.cm.binary) #display numpy array image


img = img.reshape(1, 28, 28)

predictions=model.predict([img])
print('prediction -> ',np.argmax(predictions))