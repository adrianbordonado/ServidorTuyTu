import cv2
import numpy as np
import glob, os, time

def aplicar_ruido_universal(img, intensidad=1200, usar_permutacion=True, usar_variacion_espacial=True):
    h, w, c = img.shape
    
    # --- 1. Semilla dinámica ---
    np.random.seed(int(time.time()) % 10000)
    
    # --- 2. Ruido base ---
    ruido = np.random.randint(-intensidad, intensidad+1, img.shape, dtype=np.int16)
    
    # --- 3. Variación espacial ---
    if usar_variacion_espacial:
        x = np.linspace(0, np.pi*4, w)
        y = np.linspace(0, np.pi*4, h)
        X, Y = np.meshgrid(x, y)
        mascara = (np.sin(X) + np.cos(Y)) / 2  # [-1,1]
        ruido = (ruido * mascara[..., None]).astype(np.int16)
    
    # --- 4. Aplicar ruido ---
    img_ruidosa = np.clip(img.astype(np.int16) + ruido, 0, 255).astype(np.uint8)
    
    # --- 5. Permutación de canales ---
    if usar_permutacion:
        img_ruidosa = img_ruidosa[..., np.random.permutation(c)]
    
    return img_ruidosa


# Directorios
input_dir = "img_limpia"
output_dir = "img_contaminada"
os.makedirs(output_dir, exist_ok=True)

for path in glob.glob(os.path.join(input_dir, "*.jpg")):
    img = cv2.imread(path)
    img_ruidosa = aplicar_ruido_universal(img, intensidad=10, usar_permutacion=False)

    nombre = os.path.basename(path)
    cv2.imwrite(os.path.join(output_dir, nombre), img_ruidosa)

print("Listo ✅ | imágenes guardadas en", output_dir)
