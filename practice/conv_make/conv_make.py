import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator

# 랜덤시드 고정시키기
np.random.seed(3)

train_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('/Users/exem/workspaces/conv_make/handwriting_shape/train', target_size=(24,24),batch_size=3,class_mode='categorical')

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory('/Users/exem/workspaces/conv_make/handwriting_shape/test',target_size=(24,24),batch_size=3,class_mode='categorical')

model = Sequential()
model.add(Conv2D(23, kernel_size=(3,3),activation='relu',input_shape=(24,24,3)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(Flatten())
model.add(Dense(128,activation='relu'))
model.add(Dense(3,activation='softmax'))

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

#%matplotlib inline

SVG(model_to_dot(model, show_shapes=True).create(prog='dot',format='svg'))

model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

model.fit_generator(train_generator,steps_per_epoch=15,epochs=50,validation_data=test_generator,validation_steps=5)

print("--Evaluate--")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))


print("--Predict--")
output = model.predict_generator(test_generator,steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
print(test_generator.class_indices)
print(output)