import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the data augmentation parameters
datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Apply data augmentation to the training data
train_generator = datagen.flow_from_directory(
    'path/to/train/directory',
    target_size=(224, 224),
    batch_size=32,
    class_mode='binary'
)