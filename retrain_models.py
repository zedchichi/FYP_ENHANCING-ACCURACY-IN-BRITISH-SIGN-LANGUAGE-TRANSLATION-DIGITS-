import mediapipe as mp
from tensorflow.keras.applications import MobileNet, VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)


def retrain_mobilenet(model, train_dir, val_dir, save_path):
    print("Retraining MobileNet...")

    # Assume 'model' is the MobileNet model already loaded with custom top layers and weights
    # Now let's make all layers trainable for fine-tuning
    for layer in model.layers:
        layer.trainable = True

    # Compile the model with a small learning rate
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
        steps_per_epoch=10,  # adjust based on the size of your dataset
        epochs=5,            # can be changed based on how much fine-tuning is needed
        validation_data=validation_generator,
        validation_steps=5)  # adjust based on the size of your validation set

    # Save the retrained model
    model.save(save_path)
    print("MobileNet Retrained and saved at {}".format(save_path))

def retrain_vgg16(model, train_dir, val_dir, save_path):
    print("Retraining VGG16...")

    # Making all layers trainable for fine-tuning
    for layer in model.layers:
        layer.trainable = True

    # Compile the model with a small learning rate
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Setup data generators
    train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True,
                                       fill_mode='nearest')

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
        steps_per_epoch=10,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=5)

    # Save the retrained model
    model.save(save_path)
    print("VGG16 Retrained and saved at {}".format(save_path))
