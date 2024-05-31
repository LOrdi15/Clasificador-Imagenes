import tkinter as tk
from tkinter import filedialog, ttk
from PIL import ImageTk, Image
import requests
import io

def predict_image():
    # Abre un cuadro de diálogo para seleccionar la imagen
    file_path = filedialog.askopenfilename()
    
    # Carga y muestra la imagen en la GUI
    img = Image.open(file_path)
    img = img.resize((150, 150), Image.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)
    label_image.config(image=img_tk)
    label_image.image = img_tk
    
    # Abre el archivo de imagen en modo binario
    with open(file_path, 'rb') as f:
        # Prepara los datos para la solicitud POST
        files = {'file': f.read()}
        
        # Hace una solicitud POST a la API Flask
        response = requests.post('http://localhost:5000/predict', files=files)
        
        # Obtiene la predicción de la respuesta
        prediction = response.json()['prediction']

    # Muestra la predicción en la GUI
    label_prediction.config(text="Predicción: " + prediction)

# Crea la GUI
root = tk.Tk()
root.title("Clasificador de imágenes de frutas")
root.geometry("300x400")

# Crea un marco para los widgets
frame = ttk.Frame(root, padding="10")
frame.pack(fill='both', expand=True)

# Crea un botón para seleccionar la imagen
button = ttk.Button(frame, text="Seleccionar imagen", command=predict_image)
button.pack(pady=10)

# Crea una etiqueta para mostrar la imagen
label_image = ttk.Label(frame)
label_image.pack(pady=10)

# Crea una etiqueta para mostrar la predicción
label_prediction = ttk.Label(frame, text="Predicción: ")
label_prediction.pack(pady=10)

# Ejecuta la GUI
root.mainloop()
