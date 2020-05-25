#!/usr/bin/env python
# coding: utf-8

from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from tensorflow.keras.activations import relu
from keras.utils.np_utils import to_categorical



(x_train, y_train), (x_test, y_test) = mnist.load_data()


inp_layer = Input(shape=784)
x =  Dense(64,activation=relu)(inp_layer)
out = Dense(10,activation='softmax')(x)
model = Model(inputs=[inp_layer],outputs=[out])


X_train = x_train.reshape(-1,28*28)
X_test = x_test.reshape(-1,28*28)


y_train_cat = to_categorical(y_train)
y_test_cat = to_categorical(y_test)


model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])


history = model.fit(x=X_train,y=y_train_cat,epochs=2)


results = model.evaluate(x=X_test,y=y_test_cat)


p = "{}".format(results[1])


with open('/results/data.txt', 'w') as f:
    f.write(p)


model_json = model.to_json()


with open("/model/architecture/model_num.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("/model/architecture/model_num.h5")


model.save('/model/model_weight/model_mnist_{}.h5'.format(p))






