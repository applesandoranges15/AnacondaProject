import tensorflow as tf
print(tf.__version__)
conda create -n tensorflow_env tensorflow
conda activate tensorflow_env


from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras import layers, models
from tensorflow.keras.models import Model
import zipfile
import os
import shutil

# Extract the dataset



bread='C:\Users\zoyaa\Downloads\archive.zip\training\Bread'
dairy='C:\Users\zoyaa\Downloads\archive.zip\training\Dairy product'
local_zip = 'C:\\Users\\zoyaa\\Downloads\\archive.zip'
base_dir = 'C:\\Users\\zoyaa\\Downloads\\archive'

with zipfile.ZipFile(local_zip, 'r') as zip_ref:
    zip_ref.extractall(base_dir)

# Define directories
train_dir = os.path.join(base_dir, 'training')

# Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # Use 20% of the data for validation
)

# Create data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224),
    batch_size=20,
    class_mode='categorical',
    subset='validation'
)

# Load the VGG16 model
base_model = VGG16(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Freeze the layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom layers on top of the base model
x = layers.Flatten()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(train_generator.num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=x)

# Compile the model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['acc'])

# Train the model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    epochs=10
)
