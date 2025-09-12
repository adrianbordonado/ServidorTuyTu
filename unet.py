import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

# Configuración de parámetros
IMG_HEIGHT = 1024
IMG_WIDTH = 1024
BATCH_SIZE = 16
EPOCHS = 5

# Leer el archivo Excel con las clases y códigos de color
def load_classes_from_excel(excel_path):
    classes_dict = {}
    xls = pd.ExcelFile(excel_path)
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name)
        for index, row in df.iterrows():
            class_name = row['name']  # Cambia 'Class' por el nombre de la columna correspondiente
            color_code = row['color']   # Cambia 'Color' por el nombre de la columna correspondiente
            classes_dict[class_name] = color_code
    return classes_dict

def load_data(base_path, classes_dict):
    images = []
    masks = []
    discarded_count = 0  # Contador de imágenes descartadas
    
    # Recorrer todas las subcarpetas en el directorio base
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            img_folder = os.path.join(folder_path, 'img')
            mask_folder = os.path.join(folder_path, 'masks')
            print(f"Cargando carpeta: {folder_path}")  # Imprimir carpeta actual
            
            if not os.path.exists(img_folder) or not os.path.exists(mask_folder):
                print(f"Carpeta de imágenes o máscaras no encontrada en: {folder_path}")
                continue
            
            for filename in os.listdir(img_folder):
                if filename.endswith('.png'):  # Asegúrate de que se busquen archivos .png
                    img_path = os.path.join(img_folder, filename)
                    img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
                    img = img_to_array(img) / 255.0  # Normalizar
                    print(f"Cargada imagen: {img_path}")  # Imprimir imagen cargada

                    # Usar el mismo nombre de archivo para la máscara
                    mask_path = os.path.join(mask_folder, filename)  # Sin sufijo _mask
                    if os.path.exists(mask_path):
                        mask = load_img(mask_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='grayscale')
                        mask = img_to_array(mask) / 255.0  # Normalizar

                        # Convertir la máscara a clases usando los códigos de color
                        mask_classes = np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.uint8)
                        for class_name, color_code in classes_dict.items():
                            # Usar solo el canal 0 de la máscara en escala de grises
                            mask_classes[mask[:, :, 0] == color_code[0]] = list(classes_dict.keys()).index(class_name)

                        masks.append(mask_classes)  # Almacenar la máscara de clases
                        images.append(img)  # Agregar la imagen solo si la máscara se carga correctamente
                        print(f"Cargada máscara: {mask_path}")  # Imprimir máscara cargada
                    else:
                        print(f"No se encontró la máscara: {mask_path}")  # Imprimir si no se encuentra la máscara
                        discarded_count += 1  # Incrementar el contador de descartes

    print(f"Total de imágenes descartadas: {discarded_count}")  # Imprimir el total de imágenes descartadas
    
    # Asegurarse de que se devuelvan las matrices, incluso si están vacías
    return np.array(images), np.array(masks)

# Cargar clases desde el archivo Excel
classes_dict = load_classes_from_excel('trainingdatapro/human-segmentation-dataset/versions/7/Human Segmentation 7 Types.xlsx')  # Cambia la ruta al archivo Excel

# Cargar datos
images, masks = load_data('trainingdatapro/human-segmentation-dataset/versions/7/', classes_dict)  # Cambia 'dataset/' por la ruta a tu carpeta base

# Verificar si hay máscaras cargadas
if masks.size == 0:
    raise ValueError("No se han cargado máscaras. Verifica la ruta y el formato de las imágenes y máscaras.")

# Convertir máscaras a formato categórico
masks = tf.keras.utils.to_categorical(masks, num_classes=len(classes_dict))

# Verificar el tamaño de las máscaras
print(f"Tamaño de las máscaras después de la conversión: {masks.shape}")

# Verificar el número de muestras
if images.shape[0] < 2:  # Necesitamos al menos 2 muestras para dividir
    print("No hay suficientes imágenes para dividir en conjuntos de entrenamiento y validación.")
    X_train, y_train = images, masks  # Usar todo como entrenamiento
    X_val, y_val = None, None  # No hay conjunto de validación
else:
    # Dividir en conjunto de entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Definición de la U-Net
def unet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # Encoder
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)

    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c5)

    # Decoder
    u6 = layers.UpSampling2D((2, 2))(c5)
    u6 = layers.concatenate([u6, c4])
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    c6 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c6)

    u7 = layers.UpSampling2D((2, 2))(c6)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(len(classes_dict), (1, 1), activation='softmax')(c9)

    model = models.Model(inputs=[inputs], outputs=[outputs])
    return model

# Crear y compilar el modelo
model = unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3), num_classes=len(classes_dict))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
if X_val is not None:
    model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, epochs=EPOCHS)
else:
    model.fit(X_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Exportar el modelo entrenado
model.save('unet_model.h5')



