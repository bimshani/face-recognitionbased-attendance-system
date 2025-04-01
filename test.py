import cv2
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine
import h5py
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading

# Configuration
MODEL_FOLDER = 'vgg_model'
FEATURE_MODEL_PATH = os.path.join(MODEL_FOLDER, 'vggface_features.h5')
KNOWN_FACES_DIR = 'faces'
ATTENDANCE_FILE = 'attendance.csv'
SIMILARITY_THRESHOLD = 0.6  # Cosine similarity threshold (lower is more similar)

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

class FaceRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition Attendance System")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Load models and known faces
        self.load_models_and_faces()

        # Create UI Components
        self.create_ui()

    def load_models_and_faces(self):
        # Create attendance file if it doesn't exist
        if not os.path.exists(ATTENDANCE_FILE):
            with open(ATTENDANCE_FILE, "w") as f:
                f.write("Name,Time\n")

        # Load the VGG feature extraction model
        try:
            self.vgg_feature_model = load_model(FEATURE_MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Model Loading Error", f"Error loading model: {str(e)}")
            exit(1)

        # Load known faces
        self.known_faces = {}
        self.attended_persons = set()

        # Add a progress bar for loading faces
        self.progress_window = tk.Toplevel(self.root)
        self.progress_window.title("Loading Known Faces")
        self.progress_window.geometry("400x150")
        self.progress_bar = ttk.Progressbar(self.progress_window, orient="horizontal", length=300, mode="determinate")
        self.progress_bar.pack(pady=20)
        self.progress_label = tk.Label(self.progress_window, text="Processing known faces...")
        self.progress_label.pack(pady=10)

        # Start loading faces in a separate thread
        threading.Thread(target=self.load_known_faces, daemon=True).start()

    def load_known_faces(self):
        # Face cascade for detection
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            messagebox.showerror("Classifier Error", "Could not load face cascade classifier")
            return

        total_images = sum([len(files) for r, d, files in os.walk(KNOWN_FACES_DIR) if any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in files)])
        processed_images = 0

        for person_name in os.listdir(KNOWN_FACES_DIR):
            person_path = os.path.join(KNOWN_FACES_DIR, person_name)
            if os.path.isdir(person_path):
                self.known_faces[person_name] = []
                
                for img_name in os.listdir(person_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_path, img_name)
                        try:
                            img = cv2.imread(img_path)
                            if img is not None:
                                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                                faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                                
                                if len(faces) > 0:
                                    x, y, w, h = faces[0]
                                    face_img = img[y:y+h, x:x+w]
                                    embedding = self.extract_vgg_features(face_img)
                                    self.known_faces[person_name].append(embedding)
                            
                            processed_images += 1
                            progress = int((processed_images / total_images) * 100)
                            self.progress_bar['value'] = progress
                            self.progress_label.config(text=f"Processing: {person_name} - {processed_images}/{total_images}")
                        
                        except Exception as e:
                            print(f"Error processing {img_path}: {str(e)}")

        # Close progress window
        self.progress_window.destroy()
        messagebox.showinfo("Faces Loaded", f"Loaded embeddings for {len(self.known_faces)} persons")

    def create_ui(self):
        # Main frame
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # Left side: Camera Feed
        left_frame = tk.Frame(main_frame, bg='white', bd=5, relief=tk.RAISED)
        left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10)

        self.camera_label = tk.Label(left_frame, text="Camera Feed", bg='black', fg='white')
        self.camera_label.pack(fill=tk.BOTH, expand=True)

        # Right side: Attendance and System Info
        right_frame = tk.Frame(main_frame, bg='#e0e0e0', bd=5, relief=tk.RAISED)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        # System Status
        status_frame = tk.LabelFrame(right_frame, text="System Status", bg='#e0e0e0')
        status_frame.pack(fill=tk.X, padx=10, pady=10)

        self.time_label = tk.Label(status_frame, text="", font=('Arial', 12), bg='#e0e0e0')
        self.time_label.pack(pady=5)

        # Attendance Log
        log_frame = tk.LabelFrame(right_frame, text="Attendance Log", bg='#e0e0e0')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.attendance_log = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=40, height=15)
        self.attendance_log.pack(padx=10, pady=10)

        # Control Buttons
        button_frame = tk.Frame(right_frame, bg='#e0e0e0')
        button_frame.pack(fill=tk.X, padx=10, pady=10)

        self.start_button = tk.Button(button_frame, text="Start Attendance", command=self.start_attendance, bg='green', fg='white')
        self.start_button.pack(side=tk.LEFT, padx=5)

        stop_button = tk.Button(button_frame, text="Stop", command=self.stop_attendance, bg='red', fg='white')
        stop_button.pack(side=tk.RIGHT, padx=5)

        # Start time update
        self.update_time()

        # Camera and recognition variables
        self.camera_active = False
        self.cap = None

    def start_attendance(self):
        if not self.camera_active:
            self.camera_active = True
            self.start_button.config(state=tk.DISABLED)
            threading.Thread(target=self.start_camera, daemon=True).start()

    def stop_attendance(self):
        self.camera_active = False
        if self.cap:
            self.cap.release()
        self.start_button.config(state=tk.NORMAL)
        self.camera_label.config(image='')

    def start_camera(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", "Could not open webcam")
            self.camera_active = False
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        frame_count = 0
        while self.camera_active:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_count += 1
            display_frame = frame.copy()  # Keep the original color frame

            # Use grayscale ONLY for face detection
            if frame_count % 5 == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )

                for (x, y, w, h) in faces:
                    face_img = frame[y:y+h, x:x+w]
                    try:
                        face_embedding = self.extract_vgg_features(face_img)
                        name = self.recognize_face(face_embedding)

                        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                        cv2.putText(display_frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        if name != "Unknown":
                            self.mark_attendance(name)
                    except Exception as e:
                        print(f"Recognition error: {e}")

            # Convert frame for Tkinter (maintain color)
            cv2image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            img = tk.PhotoImage(data=cv2.imencode('.png', cv2image)[1].tobytes())
            self.camera_label.config(image=img)
            self.camera_label.image = img

            self.root.update()

        if self.cap:
            self.cap.release()

    def recognize_face(self, face_embedding):
        name = "Unknown"
        min_distance = SIMILARITY_THRESHOLD

        for person_name, embeddings in self.known_faces.items():
            for stored_embedding in embeddings:
                distance = cosine(face_embedding, stored_embedding)
                if distance < min_distance:
                    name = person_name
                    min_distance = distance

        return name

    def mark_attendance(self, name):
        if name not in self.attended_persons:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(ATTENDANCE_FILE, "a") as f:
                f.write(f"{name},{current_time}\n")
            self.attended_persons.add(name)
            
            # Update attendance log
            self.attendance_log.insert(tk.END, f"{name} - {current_time}\n")
            self.attendance_log.see(tk.END)

    def update_time(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.time_label.config(text=current_time)
        self.root.after(1000, self.update_time)

    def preprocess_image(self, img):
        resized = cv2.resize(img, (224, 224))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb_img / 255.0
        preprocessed = np.expand_dims(normalized, axis=0)
        return preprocessed

    def extract_vgg_features(self, face_img):
        preprocessed = self.preprocess_image(face_img)
        features = self.vgg_feature_model.predict(preprocessed, verbose=0)
        return features[0]

    def run(self):
        self.root.mainloop()

def main():
    root = tk.Tk()
    app = FaceRecognitionApp(root)
    app.run()

if __name__ == "__main__":
    main()