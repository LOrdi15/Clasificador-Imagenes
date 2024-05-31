from flask import Flask, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import io

app = Flask(__name__)

# Carga el modelo entrenado
model = load_model('Modelo_de_Frutas.h5')

# Crea un generador de imágenes para obtener el mapeo de las clases
datagen = ImageDataGenerator(rescale=1./255)
train_generator = datagen.flow_from_directory(
    'frutas',  # especifica la ruta a tus imágenes de frutas
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical')  # usa 'categorical' para múltiples clases

# Crea un diccionario que mapea los índices de las clases a sus nombres
class_indices = train_generator.class_indices
index_to_class = {v: k for k, v in class_indices.items()}

@app.route('/predict', methods=['POST'])
def predict():
    # Obtiene la imagen del request
    img_data = request.files['file'].read()
    img = image.load_img(io.BytesIO(img_data), target_size=(150, 150))

    # Preprocesa la imagen para la entrada del modelo
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.

    # Usa el modelo para predecir la clase de la imagen
    prediction = model.predict(img)

    # La predicción será un array con la probabilidad de cada clase
    # La clase predicha será el índice con la mayor probabilidad
    predicted_class_index = np.argmax(prediction)

    # Usa el diccionario para obtener el nombre de la clase
    predicted_class_name = index_to_class[predicted_class_index]

    # Devuelve el nombre de la clase predicha
    return {'prediction': predicted_class_name}

if __name__ == '__main__':
    app.run(debug=True)
