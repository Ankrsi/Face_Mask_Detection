import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

img_size=224
epochs=20
read_X=open("X.pickle","rb")
X=pickle.load(read_X)
read_Y=open("Y.pickle","rb")
Y=pickle.load(read_Y)

baseModel = keras.applications.MobileNetV2(weights="imagenet", include_top=False,input_tensor=layers.Input(shape=(224, 224, 3)))
headModel = baseModel.output
headModel = layers.AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = layers.Flatten(name="flatten")(headModel)
headModel = layers.Dense(128, activation="relu")(headModel)
headModel = layers.Dropout(0.5)(headModel)
headModel = layers.Dense(2, activation="softmax")(headModel)
model = keras.Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
 	layer.trainable = False
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
model.fit(X,Y,epochs=epochs,batch_size=32)
model.save("mask_model.h5")
