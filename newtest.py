import cv2
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine
import h5py
import streamlit as st
import pandas as pd
import base64
import re
import mediapipe as mp
import time
from ready import fine_tune_model

# Configuration
MODEL_FOLDER = 'vgg_model'
FEATURE_MODEL_PATH = os.path.join(MODEL_FOLDER, 'vggface_features.h5')
FINE_TUNED_PATH = os.path.join(MODEL_FOLDER, 'vggface_finetuned.h5')
KNOWN_FACES_DIR = 'faces'
ATTENDANCE_FILE = 'attendance.csv'
REGISTRATION_FILE = 'registered_users.csv'
EMBEDDINGS_FILE = 'face_embeddings.h5'
SIMILARITY_THRESHOLD = 0.4
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

class FaceRecognitionSystem:
    def __init__(self):
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        
        self.load_models_and_faces()
        self.initialize_thresholds()
        
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.6)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6
        )
        self.alignment_available = True
        self.attended_persons = set()

    def initialize_thresholds(self):
        self.person_thresholds = {}
        for person_name, embeddings in self.known_faces.items():
            if len(embeddings) < 2:
                self.person_thresholds[person_name] = SIMILARITY_THRESHOLD
                continue
            distances = [cosine(embeddings[i], embeddings[j]) 
                        for i in range(len(embeddings)) 
                        for j in range(i+1, len(embeddings))]
            if distances:
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                self.person_thresholds[person_name] = min(mean_dist + 2 * std_dist, 0.5)
            else:
                self.person_thresholds[person_name] = SIMILARITY_THRESHOLD

    def load_models_and_faces(self):
        for file_path in [ATTENDANCE_FILE, REGISTRATION_FILE]:
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    f.write("Name,Time\n" if file_path == ATTENDANCE_FILE else 
                           "Name,Email,Department,Registration Date\n")

        try:
            self.vgg_feature_model = load_model(FEATURE_MODEL_PATH)
        except Exception as e:
            raise Exception(f"Failed to load feature model: {e}")

        num_classes = len([d for d in os.listdir(KNOWN_FACES_DIR) 
                         if os.path.isdir(os.path.join(KNOWN_FACES_DIR, d))]) if os.path.exists(KNOWN_FACES_DIR) else 1
        try:
            self.fine_tuned_model = fine_tune_model(num_classes + 1)  # +1 for unknown class
        except Exception as e:
            raise Exception(f"Failed to initialize fine-tuned model: {e}")

        self.known_faces = {}
        self.known_user_info = {}
        if os.path.exists(EMBEDDINGS_FILE):
            try:
                with h5py.File(EMBEDDINGS_FILE, 'r') as hf:
                    for person_name in hf.keys():
                        self.known_faces[person_name] = list(hf[person_name]['embeddings'][:])
                        self.known_user_info[person_name] = {
                            'email': hf[person_name].attrs.get('email', 'No email'),
                            'department': hf[person_name].attrs.get('department', 'No department'),
                            'registration_date': hf[person_name].attrs.get('registration_date', 'Unknown')
                        }
            except Exception as e:
                print(f"Error loading embeddings: {e}, recomputing...")
                self.compute_face_embeddings()
        else:
            self.compute_face_embeddings()

    def compute_face_embeddings(self):
        with h5py.File(EMBEDDINGS_FILE, 'w') as hf:
            df = pd.read_csv(REGISTRATION_FILE) if os.path.exists(REGISTRATION_FILE) else pd.DataFrame()
            for person_name in os.listdir(KNOWN_FACES_DIR):
                person_path = os.path.join(KNOWN_FACES_DIR, person_name)
                if os.path.isdir(person_path):
                    user_row = df[df['Name'] == person_name]
                    grp = hf.create_group(person_name)
                    grp.attrs['email'] = user_row['Email'].values[0] if not user_row.empty else 'No email'
                    grp.attrs['department'] = user_row['Department'].values[0] if not user_row.empty else 'No department'
                    grp.attrs['registration_date'] = user_row['Registration Date'].values[0] if not user_row.empty else 'Unknown'

                    embeddings_list = []
                    for img_name in os.listdir(person_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(person_path, img_name)
                            img = cv2.imread(img_path)
                            if img is None:
                                continue
                            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            face_results = self.detect_and_align_faces(img)
                            if face_results:
                                embedding = self.extract_vgg_features(face_results[0]['face'])
                                embeddings_list.append(embedding)
                    if embeddings_list:
                        grp.create_dataset('embeddings', data=embeddings_list)
                        self.known_faces[person_name] = embeddings_list

        self.initialize_thresholds()

    def align_face(self, face_img, landmarks):
        h, w = face_img.shape[:2]
        left_eye = np.array([landmarks.landmark[33].x * w, landmarks.landmark[33].y * h])
        right_eye = np.array([landmarks.landmark[263].x * w, landmarks.landmark[263].y * h])
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        eye_dist = np.sqrt(dx**2 + dy**2)
        desired_dist = w * 0.3
        scale = desired_dist / eye_dist if eye_dist > 0 else 1.0
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, angle, scale)
        return cv2.warpAffine(face_img, M, (w, h))

    def recognize_face_ensemble(self, face_img):
        samples = [face_img] + self.augment_face_image(face_img)
        votes = {}
        confidences = {}
        
        for sample in samples:
            embedding = self.extract_vgg_features(sample)
            name, distance = self.compare_embeddings(embedding)
            votes[name] = votes.get(name, 0) + 1
            confidences[name] = min(confidences.get(name, 1.0), distance)
        
        winner = max(votes.items(), key=lambda x: x[1])[0] if votes else "Unknown"
        return winner, confidences.get(winner, 1.0) if winner != "Unknown" else 1.0

    def augment_face_image(self, img):
        augmented = []
        for angle in [-5, 5]:
            M = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), angle, 1.0)
            augmented.append(cv2.warpAffine(img, M, (img.shape[1], img.shape[0])))
        for alpha in [0.9, 1.1]:
            augmented.append(cv2.convertScaleAbs(img, alpha=alpha))
        return augmented

    def compare_embeddings(self, face_embedding):
        name, min_distance = "Unknown", 1.0
        for person_name, embeddings in self.known_faces.items():
            threshold = self.person_thresholds.get(person_name, SIMILARITY_THRESHOLD)
            distances = [cosine(face_embedding, emb) for emb in embeddings]
            curr_min = min(distances)
            if curr_min < threshold and curr_min < min_distance:
                name, min_distance = person_name, curr_min
        return name, min_distance

    def preprocess_image(self, img):
        resized = cv2.resize(img, (224, 224))
        rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB) if resized.shape[2] == 3 else resized
        normalized = rgb_img.astype(np.float32) / 255.0
        mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        return np.expand_dims(normalized, axis=0)

    def extract_vgg_features(self, face_img):
        preprocessed = self.preprocess_image(face_img)
        return self.vgg_feature_model.predict(preprocessed, verbose=0)[0]

    def detect_and_align_faces(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = []
        detections = self.face_detector.process(rgb_frame)
        if detections.detections:
            for detection in detections.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x1, y1 = int(bbox.xmin * w), int(bbox.ymin * h)
                box_w, box_h = int(bbox.width * w), int(bbox.height * h)
                margin = int(max(box_w, box_h) * 0.2)
                x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
                x2, y2 = min(w, x1 + box_w + 2*margin), min(h, y1 + box_h + 2*margin)
                
                face_img = frame[y1:y2, x1:x2]
                landmarks = self.face_mesh.process(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                if landmarks.multi_face_landmarks:
                    aligned_face = self.align_face(face_img, landmarks.multi_face_landmarks[0])
                    results.append({'face': aligned_face, 'box': (x1, y1, x2-x1, y2-y1)})
                else:
                    results.append({'face': face_img, 'box': (x1, y1, x2-x1, y2-y1)})
        return results

    def mark_attendance(self, name):
        if name not in self.attended_persons:
            current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with open(ATTENDANCE_FILE, "a") as f:
                f.write(f"{name},{current_time}\n")
            self.attended_persons.add(name)
            return current_time
        return None

    def register_user(self, name, email, department):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        safe_name = re.sub(r'[^\w\-_\. ]', '_', name.strip())
        with open(REGISTRATION_FILE, "a") as f:
            f.write(f"{safe_name},{email},{department},{current_time}\n")
        return current_time, safe_name

def main():
    st.set_page_config(page_title="Face Recognition Attendance System", layout="wide")
    try:
        face_system = FaceRecognitionSystem()
    except Exception as e:
        st.error(f"Failed to initialize FaceRecognitionSystem: {e}")
        return

    st.sidebar.title("Navigation")
    menu = st.sidebar.radio("Menu", ["Attendance", "Register User", "View Attendance", "View Registered Users"])

    if menu == "Attendance":
        st.title("Attendance Marking")
        if st.button("Start Camera"):
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Failed to open camera")
                return
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            frame_placeholder = st.empty()
            last_recognized = {}

            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                display_frame = frame.copy()
                face_results = face_system.detect_and_align_faces(frame)
                
                for face_result in face_results:
                    face_img = face_result['face']
                    x, y, w, h = face_result['box']
                    name, confidence = face_system.recognize_face_ensemble(face_img)
                    accuracy = int((1 - confidence) * 100) if confidence <= 1 else 0
                    
                    if name != "Unknown":
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(display_frame, f"{name} ({accuracy}%)", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        if name not in last_recognized or (datetime.datetime.now() - last_recognized[name]).seconds > 300:
                            attendance_time = face_system.mark_attendance(name)
                            if attendance_time:
                                st.success(f"Attendance marked for {name} at {attendance_time}")
                            last_recognized[name] = datetime.datetime.now()
                    else:
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(display_frame, "Unknown", 
                                  (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), channels="RGB")
            
            cap.release()

    elif menu == "Register User":
        st.title("Register New User")
        with st.form("registration_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            department = st.text_input("Department")
            samples = st.slider("Number of samples", 1, 10, 5)
            submitted = st.form_submit_button("Start Capture")
        
        if submitted and name and email and department:
            reg_time, safe_name = face_system.register_user(name, email, department)
            user_dir = os.path.join(KNOWN_FACES_DIR, safe_name)
            os.makedirs(user_dir, exist_ok=True)
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Failed to open camera")
                return
            frame_placeholder = st.empty()
            samples_taken = 0
            
            while samples_taken < samples:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break
                face_results = face_system.detect_and_align_faces(frame)
                display_frame = frame.copy()
                
                if face_results:
                    x, y, w, h = face_results[0]['box']
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Sample {samples_taken+1}/{samples}", 
                              (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    frame_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                    
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    cv2.imwrite(os.path.join(user_dir, f"face_{timestamp}.jpg"), face_results[0]['face'])
                    samples_taken += 1
                    time.sleep(0.5)
            
            cap.release()
            face_system.compute_face_embeddings()
            st.success(f"Registered {name} with {samples_taken} samples!")

    elif menu == "View Attendance":
        st.title("Attendance Records")
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            if not df.empty:
                df['Time'] = pd.to_datetime(df['Time'])
                start_date = st.date_input("From", df['Time'].min().date())
                end_date = st.date_input("To", df['Time'].max().date())
                mask = (df['Time'].dt.date >= start_date) & (df['Time'].dt.date <= end_date)
                st.dataframe(df.loc[mask])
            else:
                st.info("No attendance records found")
        except Exception as e:
            st.error(f"Error reading attendance file: {e}")

    elif menu == "View Registered Users":
        st.title("Registered Users")
        try:
            df = pd.read_csv(REGISTRATION_FILE)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Error reading registration file: {e}")

if __name__ == "__main__":
    main()