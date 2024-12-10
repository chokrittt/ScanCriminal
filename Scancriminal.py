import os
import cv2
import dlib
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk


# Fonction pour charger les visages connus
def load_known_faces(known_faces_dir):
    known_faces = []
    known_infos = []  # Liste des informations des individus

    face_detector = dlib.get_frontal_face_detector()
    face_recognition_model = dlib.face_recognition_model_v1("C:/Users/21692/Downloads/dlib_face_recognition_resnet_model_v1.dat")
    shape_predictor = dlib.shape_predictor("C:/Users/21692/Downloads/shape_predictor_68_face_landmarks.dat")

    for filename in os.listdir(known_faces_dir):
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_path = os.path.join(known_faces_dir, filename)
            image = cv2.imread(image_path)

            if image is None:
                print(f"Erreur : L'image {filename} n'a pas pu être lue.")
                continue

            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_detector(gray)

            for face in faces:
                shape = shape_predictor(image, face)
                face_descriptor = np.array(face_recognition_model.compute_face_descriptor(image, shape))

                known_faces.append(face_descriptor)

                # Ajouter des informations pour chaque visage
                base_name = os.path.splitext(filename)[0]
                info = {
                    "Nom": base_name.split("_")[0],
                    "Prenom": base_name.split("_")[1] if "_" in base_name else "Inconnu",
                    "Nationalite": "Tunisienne",
                    "B3": "Vide",
                }
                known_infos.append(info)

    return known_faces, known_infos


# Fonction pour effectuer la reconnaissance faciale
def recognize_face(image, known_faces, known_infos):
    face_detector = dlib.get_frontal_face_detector()
    face_recognition_model = dlib.face_recognition_model_v1("C:/Users/21692/Downloads/dlib_face_recognition_resnet_model_v1.dat")
    shape_predictor = dlib.shape_predictor("C:/Users/21692/Downloads/shape_predictor_68_face_landmarks.dat")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray)

    for face in faces:
        shape = shape_predictor(image, face)
        face_descriptor = np.array(face_recognition_model.compute_face_descriptor(image, shape))

        distances = [np.linalg.norm(face_descriptor - known_face) for known_face in known_faces]
        min_distance_index = np.argmin(distances)
        info = known_infos[min_distance_index]

        # Dessiner un rectangle autour du visage
        (x, y, w, h) = (face.left(), face.top(), face.width(), face.height())
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Afficher les informations sous forme structurée
        overlay = image.copy()
        bar_height = 20 + 20 * len(info)  # Ajuster en fonction du nombre de lignes
        cv2.rectangle(overlay, (x, y - bar_height), (x + w, y), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

        y_text = y - bar_height + 15
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(image, text, (x + 5, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_text += 15

    return image


# Classe pour l'interface graphique
class FaceRecognitionApp:
    def __init__(self, root, known_faces_dir):
        self.root = root
        self.root.title("Reconnaissance Faciale")
        self.root.geometry("800x600")
        self.root.configure(bg="#333")  # Couleur de fond noir

        self.known_faces_dir = known_faces_dir
        self.known_faces, self.known_infos = load_known_faces(self.known_faces_dir)

        self.canvas = tk.Canvas(self.root, width=640, height=480)
        self.canvas.pack(pady=20)

        self.video_source = 0  # Caméra par défaut
        self.vid = None
        self.running = False

        # Boutons
        self.start_button = tk.Button(self.root, text="Démarrer", bg="black", fg="white", font=("Arial", 12, "bold"), command=self.start_video)
        self.start_button.pack(side=tk.LEFT, padx=20)

        self.stop_button = tk.Button(self.root, text="Arrêter", bg="black", fg="white", font=("Arial", 12, "bold"), command=self.stop_video, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=20)

        self.quit_button = tk.Button(self.root, text="Quitter", bg="black", fg="white", font=("Arial", 12, "bold"), command=self.quit_app)
        self.quit_button.pack(side=tk.LEFT, padx=20)

    def start_video(self):
        if not self.running:
            self.running = True
            self.vid = cv2.VideoCapture(self.video_source)
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            self.update_frame()

    def stop_video(self):
        if self.running:
            self.running = False
            if self.vid:
                self.vid.release()
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

    def update_frame(self):
        if self.running:
            ret, frame = self.vid.read()
            if ret:
                frame = recognize_face(frame, self.known_faces, self.known_infos)

                # Convertir l'image de OpenCV à Tkinter
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame_rgb)
                img_tk = ImageTk.PhotoImage(img)

                self.canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
                self.canvas.img_tk = img_tk  # Conserver une référence pour éviter le garbage collection

            self.root.after(10, self.update_frame)

    def quit_app(self):
        self.stop_video()
        self.root.quit()
        self.root.destroy()


if __name__ == "__main__":
    faces_directory = "C:/Users/21692/Pictures/projet/faces"

    if not os.path.exists(faces_directory):
        print(f"Erreur : Le répertoire des visages {faces_directory} est introuvable.")
    else:
        root = tk.Tk()
        app = FaceRecognitionApp(root, faces_directory)
        root.mainloop()
