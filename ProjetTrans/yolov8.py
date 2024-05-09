import numpy as np
import cv2
from ultralytics import YOLO
import tkinter as tk


def SMART_tracker():

    # Charger un modèle plus léger de YOLOv8
    model = YOLO('yolov8n')

    # Initialiser la webcam avec une résolution inférieure
    cap = cv2.VideoCapture("video.mp4")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 60)

    if not cap.isOpened():
        print("Erreur: Impossible d'accéder à la webcam.")
        exit()

    # Couleurs uniques pour les indices de classe désirés
    # person, backpack, handbag, suitcase, bottle, chair
    desired_classes = [0, 24, 26, 28, 39, 56, 67, 19]
    colors = {cls: [int(x) for x in np.random.uniform(0, 255, 3)] for cls in desired_classes}

    cv2.namedWindow('System for Metro in Real-Time', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('System for Metro in Real-Time', 1280, 720)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Erreur: Impossible de capturer une image de la webcam.")
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = model(image_rgb)

        # Dictionnaire pour compter les objets détectés
        class_counts = {cls: 0 for cls in desired_classes}

        if isinstance(results, list) and hasattr(results[0], 'boxes'):
            bboxes = results[0].boxes.xyxy.numpy()
            confs = results[0].boxes.conf.numpy()
            classes = results[0].boxes.cls.numpy()

            seen = set()
            class_names_replacement = {0: "Passager", 24: "Sac", 26: "Sac", 28: "Valise", 39: "Bouteille",
                                       56: "Siege", 19: "Vache", 67: "Phone"}

            # Dans la boucle, utilisez ce dictionnaire pour afficher les nouveaux noms
            for bbox, conf, cls in zip(bboxes, confs, classes):
                if int(cls) in desired_classes:
                    class_counts[int(cls)] += 1
                    key = (cls, tuple(bbox))
                    if key not in seen:
                        seen.add(key)
                        color = colors[int(cls)]
                    else:
                        color = [int(x) for x in np.random.uniform(0, 255, 3)]

                    x1, y1, x2, y2 = map(int, bbox)
                    # Utiliser le dictionnaire pour obtenir le nom remplaçant
                    label_name = class_names_replacement.get(int(cls), "Non classé")
                    label = f"{label_name}: {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Afficher les comptages dans la console
        print("Nombre d'objets détectés par classe:")
        for cls in class_counts:
            print(f"{results[0].names[cls]}: {class_counts[cls]}")

        cv2.imshow('System for Metro in Real-Time', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def SMART_tracker_image():
    # Charger un modèle plus léger de YOLOv8
    model = YOLO('yolov8n')

    # Charger l'image depuis le chemin spécifié
    frame = cv2.imread("assets/chaises.jpg")
    if frame is None:
        print("Erreur: Impossible de charger l'image.")
        return

    # Redimensionner pour la fenêtre d'affichage si nécessaire
    frame = cv2.resize(frame, (1280, 720))

    # Couleurs uniques pour les indices de classe désirés
    desired_classes = [0, 24, 26, 28, 39, 56]
    colors = {cls: [int(x) for x in np.random.uniform(0, 255, 3)] for cls in desired_classes}

    cv2.namedWindow('SMART Tracker Image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('SMART Tracker Image', 1280, 720)

    # Convertir l'image BGR en RGB pour YOLO
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(image_rgb)

    # Dictionnaire pour compter les objets détectés
    class_counts = {cls: 0 for cls in desired_classes}

    if isinstance(results, list) and hasattr(results[0], 'boxes'):
        bboxes = results[0].boxes.xyxy.numpy()
        confs = results[0].boxes.conf.numpy()
        classes = results[0].boxes.cls.numpy()

        seen = set()
        for bbox, conf, cls in zip(bboxes, confs, classes):
            if int(cls) in desired_classes:
                class_counts[int(cls)] += 1  # Mise à jour du comptage
                key = (cls, tuple(bbox))
                if key not in seen:
                    seen.add(key)
                    color = colors[int(cls)]
                else:
                    color = [int(x) for x in np.random.uniform(0, 255, 3)]

                x1, y1, x2, y2 = map(int, bbox)
                label = f"{results[0].names[int(cls)]}: {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Afficher les comptages dans la console
    print("Nombre d'objets détectés par classe:")
    for cls in class_counts:
        print(f"{results[0].names[cls]}: {class_counts[cls]}")

    cv2.imshow('SMART Tracker Image', frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def choose_metro():
    fenetre2 = tk.Toplevel()
    fenetre2.title("Choisir le métro")
    fenetre2.geometry("1800x800")
    fenetre2.resizable(False, False)

    # Charger les images et garder une référence dans l'instance de la fenêtre pour éviter la collecte des ordures
    fenetre2.background_img = tk.PhotoImage(file="assets/background.png")
    fenetre2.train_img = tk.PhotoImage(file="assets/train.png")
    fenetre2.train_hover_img = tk.PhotoImage(file="assets/train_hover.png")

    canvas = tk.Canvas(fenetre2, width=1800, height=800, bg='white')
    canvas.pack(fill="both", expand=True)
    canvas.create_image(0, 0, image=fenetre2.background_img, anchor="nw")

    # Fonction pour gérer le clic sur un wagon
    def on_wagon_click(wagon_number):
        print(f"Wagon {wagon_number} clicked")

    # Créer et positionner les labels des wagons sur le canvas
    for i in range(5):
        x_position = (i * 360) + 180
        train_image = canvas.create_image(x_position, 400, image=fenetre2.train_img, anchor="center")

        # Associer les événements de survol pour chaque image avec un lambda correctement bindé
        canvas.tag_bind(train_image, "<Enter>", lambda event, img=fenetre2.train_hover_img, item=train_image: canvas.itemconfig(item, image=img))
        canvas.tag_bind(train_image, "<Leave>", lambda event, img=fenetre2.train_img, item=train_image: canvas.itemconfig(item, image=img))

        # Associer l'événement de clic pour imprimer le numéro du wagon
        canvas.tag_bind(train_image, "<Button-1>", lambda event, wagon_number=i+1: on_wagon_click(wagon_number))

# Créer une instance de la fenêtre principale
fenetre = tk.Tk()
fenetre.title("SMART Tracker")
fenetre.geometry("300x200")

# Créer des boutons pour les différentes fonctions
bouton = tk.Button(fenetre, text="Webcam", command=lambda: print("Webcam started"))  # Simuler la fonction
bouton2 = tk.Button(fenetre, text="Image", command=lambda: print("Image processing started"))  # Simuler la fonction
bouton3 = tk.Button(fenetre, text="Choisir le métro", command=choose_metro)

# Placer les boutons dans la fenêtre principale
bouton.pack(expand=True)
bouton2.pack(expand=True)
bouton3.pack(expand=True)

# Lancer la boucle principale de Tkinter
fenetre.mainloop()