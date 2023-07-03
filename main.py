from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np
import cv2

# Progetto Sistemi ad Agenti sul riconoscimento delle iterazioni "Stretta di mano,
# Batti cinque, Abbraccio,Bacio, Nessuna delle precedenti"


# Disattivata la notazione scientifica poiché superflua per il nostro caso
np.set_printoptions(suppress=True)

# Caricamento del modello addestrato tramite TM
model = load_model("keras_model.h5", compile=False)

# Caricamento delle etichetta da file labels.txt
class_names = open("labels.txt", "r").readlines()

# Sorgente video da analizzare e predirre
video_path = input("Inserisci il percorso del file video da analizzare: ")

#Apertura video tramite OpenCV
video = cv2.VideoCapture(video_path)

# Inizializzazione dell'array delle predizioni
total_predictions = []

# Lettura ed elaborazione delle clip del video in input da analizzare
while True:
    ret, frame = video.read()
    if not ret:
        break

    # Il video viene convertito in immagini (frame)
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Pre-elabora il frame in esame
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Adatta la forma di input del modello per renderlo compatibile per l'analisi
    data = np.expand_dims(normalized_image_array, axis=0)

    # Effettua predizione e successivamente l'aggiunge alla lista
    prediction = model.predict(data)
    total_predictions.append(prediction)

# Se ci dovessere essere meno di 10 frame nel video, prende tutti i frame
if len(total_predictions) < 10:
    top_predictions = total_predictions
else:
    # Altrimenti trova gli indici dei primi 10 frame con la maggiore confidenza
    top_indices = np.argsort([p[0][np.argmax(p)] for p in total_predictions])[-10:]
    # Seleziona i frame corrispondenti agli indici delle prima 10 posizioni sopracitate
    top_predictions = [total_predictions[i] for i in top_indices]

# Calcola la media delle predizioni
average_prediction = np.mean(top_predictions, axis=0)

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
