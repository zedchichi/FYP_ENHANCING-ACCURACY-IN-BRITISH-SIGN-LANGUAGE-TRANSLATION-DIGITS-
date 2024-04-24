import os

import mediapipe as mp
from tensorflow.keras.applications import MobileNet, VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from math import ceil

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


def retrain_mobilenet(model, train_dir, val_dir, save_path):
    print("Retraining MobileNet...")

    #Determine dataset size
    num_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
    num_val_samples = sum([len(files) for r, d, files in os.walk(val_dir)])

    batch_size = 32 if num_train_samples >32 else num_train_samples
    step_per_epoch = ceil(num_train_samples / batch_size)
    validation_steps = ceil(num_val_samples / batch_size)

    epochs = 10 if num_train_samples < 500 else 5

    # MobileNet model already loaded with custom top layers and weights
    # make all layers trainable for fine-tuning
    for layer in model.layers:
        layer.trainable = True

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Setup data generators
    train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

    val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    # Fit the model
    model.fit(
        train_generator,
        steps_per_epoch=step_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps)

    # Save the retrained model
    model.save(save_path)
    print("MobileNet Retrained and saved at {}".format(save_path))

def retrain_vgg16(model, train_dir, val_dir, save_path):
    print("Retraining VGG16...")

    num_train_samples = sum([len(files) for r, d, files in os.walk(train_dir)])
    num_val_samples = sum([len(files) for r, d, files in os.walk(val_dir)])

    batch_size = 32 if num_train_samples > 32 else num_train_samples
    steps_per_epoch = ceil(num_train_samples / batch_size)
    validation_steps = ceil(num_val_samples / batch_size)
    epochs = 10 if num_train_samples < 500 else 5

    retrain_layer = 70

    model.trainable = True

    # Making all layers trainable for fine-tuning
    for layer in model.layers[:retrain_layer]:
        layer.trainable = False

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Setup data generators
    train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
                                       rotation_range=50,
                                       width_shift_range=0.1,
                                       height_shift_range=0.1,
                                       shear_range=0.1,
                                       zoom_range=[0.8, 1.2],
                                       horizontal_flip=True,
                                       fill_mode='nearest',
                                       brightness_range=[0.8, 1.2],
                                       channel_shift_range=30.0)

    val_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical')

    # Fit the model
    model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps)

    # Save the retrained model
    model.save(save_path)
    print("VGG16 Retrained and saved at {}".format(save_path))
