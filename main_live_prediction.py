from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2

# Caricamento del modello addestrato
model = load_model("keras_model.h5", compile=False)

# Caricamento delle etichette da file labels.txt
class_names = open("labels.txt", "r").readlines()

# Sorgente video da analizzare e predirre
video_path = input("Inserisci il percorso contente il file video da analizzare: ")

# Apertura video tramite OpenCV
video = cv2.VideoCapture(video_path)

# Inizializzazione dell'array delle predizioni
predictions = []

# Lettura ed elaborazione dei frame del video in input
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Conversione del frame in un'immagine PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Pre-elaborazione dell'immagine
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Adattamento della forma di input del modello
    data = np.expand_dims(normalized_image_array, axis=0)

    # Effettua la predizione e aggiungi il risultato all'array delle predizioni
    prediction = model.predict(data)
    predictions.append(prediction)

    # Ottieni l'indice della classe prevista e il punteggio di confidenza
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]

    # Stampa a video la previsione per il frame corrente
    print("Frame:", len(predictions))
    print("Classe:", class_name[2:])
    print("Punteggio di Confidenza:", confidence_score)
    print("")

# Calcola la media delle predizioni per tutti i frame
average_prediction = np.mean(predictions, axis=0)

# Ottieni l'indice della classe prevista e il punteggio di confidenza dalla previsione media
index = np.argmax(average_prediction)
class_name = class_names[index]
confidence_score = average_prediction[0][index]

# Stampa a video la previsione complessiva e il punteggio di confidenza per il video
print("Classe (Media):", class_name[2:])
print("Punteggio di Confidenza (Media):", confidence_score)

# Rilascia l'oggetto di acquisizione video e chiude tutte le finestre aperte
video.release()
cv2.destroyAllWindows()
