import os
import cv2


def convert_to_frames(video_path, output_dir):
    # Ottieni il nome del file senza l'estensione
    file_name = os.path.splitext(os.path.basename(video_path))[0]

    # Apri il video
    video = cv2.VideoCapture(video_path)
    success, frame = video.read()
    count = 0

    # Leggi e salva tutti i frame del video
    while success:
        # Crea il nome del file per il frame corrente
        frame_file_name = f"{file_name}_{count}.jpg"

        # Crea il percorso di output per il frame corrente
        frame_path = os.path.join(output_dir, frame_file_name)

        # Salva il frame come immagine
        cv2.imwrite(frame_path, frame)

        # Leggi il frame successivo
        success, frame = video.read()
        count += 1

    # Rilascia la risorsa del video
    video.release()


def main():
    # Cartella contenente i video
    input_dir = "tv_human_interactions_videos/Test_esterni/Highfive"

    # Cartella di output per i frame
    output_dir = "tv_human_interactions_videos/Test_esterni/Highfive"
    os.makedirs(output_dir, exist_ok=True)

    # Leggi tutti i file nella cartella di input
    file_list = os.listdir(input_dir)

    # Elabora ogni file nella cartella di input
    for file_name in file_list:
        # Crea il percorso completo del file video
        video_path = os.path.join(input_dir, file_name)

        # Verifica se il file è un video
        if os.path.isfile(video_path) and file_name.endswith((".avi", ".mp4", ".mkv")):
            # Converti il video in frame e salvali nella cartella di output
            convert_to_frames(video_path, output_dir)
            print(f"Video {file_name} convertito in frame nella cartella {output_dir}")


if __name__ == "__main__":
    main()
