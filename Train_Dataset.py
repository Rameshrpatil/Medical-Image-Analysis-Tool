# import required packages
import cv2
import sys
from PIL import Image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

# Initialize image data generator with rescaling
train_data_gen = ImageDataGenerator(rescale=1./255)
validation_data_gen = ImageDataGenerator(rescale=1./255)

# Preprocess all train images
train_generator = train_data_gen.flow_from_directory(
        'Dataset/train',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')
# Preprocess all Test images
validation_generator = validation_data_gen.flow_from_directory(
        'Dataset/val',
        target_size=(48, 48),
        batch_size=64,
        color_mode="grayscale",
        class_mode='categorical')

# create model structure
Medical_Model = Sequential()

Medical_Model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
Medical_Model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
Medical_Model.add(MaxPooling2D(pool_size=(2, 2)))
Medical_Model.add(Dropout(0.25))

Medical_Model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
Medical_Model.add(MaxPooling2D(pool_size=(2, 2)))
Medical_Model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
Medical_Model.add(MaxPooling2D(pool_size=(2, 2)))
Medical_Model.add(Dropout(0.25))

Medical_Model.add(Flatten())
Medical_Model.add(Dense(1024, activation='relu'))
Medical_Model.add(Dropout(0.5))
Medical_Model.add(Dense(2, activation='softmax'))

cv2.ocl.setUseOpenCL(False)

Medical_Model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])

# Train the neural network/model
Medical_Model_info = Medical_Model.fit_generator(
        train_generator,
        steps_per_epoch= 4400// 64,
        epochs=50,
        validation_data=validation_generator,
        validation_steps=240 // 64)

# save model structure in jason file
model_json = Medical_Model.to_json()
with open("Medical_Model.json", "w") as json_file:
    json_file.write(model_json)

# save trained model weight in .h5 file
Medical_Model.save_weights('Medical_Model.h5')