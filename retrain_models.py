from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet import preprocess_input as mobilenet_preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_preprocess_input
from tensorflow.keras.models import load_model
import os

def retrain_model(model_path, data_dir, preprocess_input_func, model_save_path):
    try:
        model = load_model(model_path)
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input_func)

        generator = datagen.flow_from_directory(
            data_dir,
            target_size=(224, 224),
            batch_size=10,  # Reduced batch size to ensure it fits in memory
            class_mode='categorical'
        )

        # Train the model
        model.fit(generator, epochs=5)
        # Save the updated model
        model.save(model_save_path)
        return "Training succeeded"
    except Exception as e:
        print(f"An error occurred: {e}")
        return str(e)

def main():
    # Define paths
    mobilenet_data_dir = r'C:\Users\anazi\FYP\classified\mobilenet'
    vgg_data_dir = r'C:\Users\anazi\FYP\classified\vgg'
    mobilenet_model_path = r'C:\Users\anazi\FYP\app\BSL_MobileNet_HD_build_mobilenet_hyper4.h5'
    vgg_model_path = r'C:\Users\anazi\FYP\app\BSL_VGG16_Cus_FT_HD_Best_Model3.h5'
    updated_mobilenet_path = r'C:\Users\anazi\FYP\app\updated_mobilenet_model.h5'
    updated_vgg_path = r'C:\Users\anazi\FYP\app\updated_vgg_model.h5'

    print(retrain_model(mobilenet_model_path, mobilenet_data_dir, mobilenet_preprocess_input, updated_mobilenet_path))
    print(retrain_model(vgg_model_path, vgg_data_dir, vgg_preprocess_input, updated_vgg_path))

if __name__ == "__main__":
    main()