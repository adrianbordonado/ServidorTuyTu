import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array, save_img
import os

# Configuración de parámetros
IMG_HEIGHT = 1024
IMG_WIDTH = 1024

# Cargar el modelo entrenado
model = tf.keras.models.load_model('unet_model.h5')

def segment_image(image_path, output_path):
    # Cargar y preprocesar la imagen
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    img_array = img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Añadir dimensión de batch

    # Realizar la predicción
    prediction = model.predict(img_array)
    predicted_mask = np.argmax(prediction, axis=-1)  # Obtener la clase con mayor probabilidad
    predicted_mask = np.squeeze(predicted_mask)  # Eliminar la dimensión de batch

    # Convertir la máscara a una imagen
    segmented_image = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
    for class_index in range(segmented_image.shape[2]):
        segmented_image[predicted_mask == class_index] = [255, 0, 0]  # Color rojo para la clase 0, puedes cambiarlo según tus clases

    # Guardar la imagen segmentada
    save_img(output_path, segmented_image)
    print(f"Imagen segmentada guardada en: {output_path}")

# Ruta de la imagen a segmentar
input_image_path = 'image.jpg'  # Cambia esto a la ruta de tu imagen
output_image_path = 'segmented_image.jpg'  # Cambia esto a la ruta donde deseas guardar la imagen segmentada

# Llamar a la función para segmentar la imagen
segment_image(input_image_path, output_image_path)
