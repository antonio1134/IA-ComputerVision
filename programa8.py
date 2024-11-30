import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import os

# Desactivar optimización OneDNN (opcional para sistemas específicos)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Función para preprocesar las imágenes
def preprocess_image(image, target_size=(64, 64)):
    if isinstance(image, str):  # Si es una ruta, carga la imagen
        image = cv2.imread(image)
        if image is None:
            raise FileNotFoundError(f"No se pudo leer la imagen en la ruta: {image}")
    # Procesar la imagen
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    return image / 255.0  # Normalizar entre 0 y 1

# Función para cargar imágenes desde carpetas
def load_images_from_folders(folder_paths, labels, target_size=(64, 64)):
    images = []
    image_labels = []
    for folder, label in zip(folder_paths, labels):
        for filename in os.listdir(folder):
            image_path = os.path.join(folder, filename)
            if filename.endswith(('.png', '.jpg', '.jpeg')):  # Extensiones permitidas
                try:
                    img = preprocess_image(image_path, target_size)
                    images.append(img)
                    image_labels.append(label)
                except Exception as e:
                    print(f"Error cargando la imagen {image_path}: {e}")
    return np.array(images), np.array(image_labels)

# Rutas de carpetas con imágenes (modifica con tus carpetas)
folder_paths = [
    '/home/antonio/9oSemestre/IA/u4/triangulos',
    '/home/antonio/9oSemestre/IA/u4/cuadrados',
    '/home/antonio/9oSemestre/IA/u4/circulos',
]
labels = [0, 1, 2]  # Etiquetas: 0-Triángulo, 1-Cuadrado, 2-Círculo

# Cargar las imágenes y etiquetas
X, y = load_images_from_folders(folder_paths, labels)
y = to_categorical(y, num_classes=3)  # Codificación one-hot

# Dividir los datos en entrenamiento y validación
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo
model = Sequential([
    Flatten(input_shape=(64, 64, 3)),
    Dense(512, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(3, activation='softmax')  # Tres clases: triángulo, cuadrado, círculo
])

# Compilar el modelo
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Entrenar el modelo
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=40)

# Guardar el modelo
model.save('shape_detector_model.h5')

# Función para predecir la forma
def predict_shape(image, model, target_size=(64, 64)):
    img = preprocess_image(image, target_size)
    img = np.expand_dims(img, axis=0)  # Añadir dimensión para el batch
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    return class_idx

# Cargar el modelo
model = tf.keras.models.load_model('shape_detector_model.h5')

# Procesar la imagen para detección de contornos y usar el modelo
image = cv2.imread('/home/antonio/9oSemestre/IA/u4/figuras3.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 50, 200)
canny = cv2.dilate(canny, None, iterations=2)
canny = cv2.erode(canny, None, iterations=1)
cnts, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    if w > 10 and h > 10:  # Filtrar contornos muy pequeños
        roi = image[y:y+h, x:x+w]  # Región de interés
        try:
            class_idx = predict_shape(roi, model)
            label = ["Triangulo", "Cuadrado", "Circulo"][class_idx]
            # Dibujar la etiqueta en el centro del contorno
            text_x = x + w // 2 - 10
            text_y = y + h // 2
            cv2.putText(image, label, (text_x, text_y), 1, 1, (0, 255, 0), 2)
        except Exception as e:
            print(f"Error procesando la ROI: {e}")
        cv2.drawContours(image, [c], 0, (0, 255, 0), 2)

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
