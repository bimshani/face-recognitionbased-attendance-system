import cv2
import os
import datetime
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from scipy.spatial.distance import cosine
import h5py
import threading
import streamlit as st
import pandas as pd
import base64
import re
import mediapipe as mp
import time

# Configuration
MODEL_FOLDER = 'vgg_model'
FEATURE_MODEL_PATH = os.path.join(MODEL_FOLDER, 'vggface_features.h5')
KNOWN_FACES_DIR = 'faces'
ATTENDANCE_FILE = 'attendance.csv'
REGISTRATION_FILE = 'registered_users.csv'
EMBEDDINGS_FILE = 'face_embeddings.h5'
SIMILARITY_THRESHOLD = 0.45  # Decreased threshold for stricter matching

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

class FaceRecognitionSystem:
    def __init__(self):
        os.makedirs(MODEL_FOLDER, exist_ok=True)
        # Load models and known faces
        self.load_models_and_faces()
        
        # Initialize person-specific thresholds based on data
        self.initialize_thresholds()
        
        # Initialize MediaPipe Face Detection and Landmarks
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_detector = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.alignment_available = True  # MediaPipe supports alignment
            
        # Track faces with confidence over time for stability
        self.face_tracking = {}
        self.attended_persons = set()

    def initialize_thresholds(self):
        """Initialize person-specific thresholds based on intra-class variations"""
        self.person_thresholds = {}
        
        for person_name, embeddings in self.known_faces.items():
            if len(embeddings) < 2:
                self.person_thresholds[person_name] = SIMILARITY_THRESHOLD
                continue
                
            distances = []
            for i in range(len(embeddings)):
                for j in range(i+1, len(embeddings)):
                    dist = cosine(embeddings[i], embeddings[j])
                    distances.append(dist)
            
            if distances:
                mean_dist = np.mean(distances)
                std_dist = np.std(distances)
                threshold = min(mean_dist + 2 * std_dist, 0.5)
                self.person_thresholds[person_name] = threshold
            else:
                self.person_thresholds[person_name] = SIMILARITY_THRESHOLD
                
        st.info(f"Dynamic thresholds initialized for {len(self.person_thresholds)} persons")

    def load_models_and_faces(self):
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        
        for file_path in [ATTENDANCE_FILE, REGISTRATION_FILE]:
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    if file_path == ATTENDANCE_FILE:
                        f.write("Name,Time\n")
                    else:
                        f.write("Name,Email,Department,Registration Date\n")

        try:
            self.vgg_feature_model = load_model(FEATURE_MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None  # This causes the method to exit early

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
                st.success(f"Loaded pre-computed embeddings for {len(self.known_faces)} persons")
            except Exception as e:
                st.error(f"Error loading embeddings: {str(e)}")
                self.compute_face_embeddings()
        else:
            self.compute_face_embeddings()

    def compute_face_embeddings(self):
        progress_bar = st.progress(0)
        status_text = st.empty()

        total_images = 0
        for person_name in os.listdir(KNOWN_FACES_DIR):
            person_path = os.path.join(KNOWN_FACES_DIR, person_name)
            if os.path.isdir(person_path):
                total_images += sum(1 for img_name in os.listdir(person_path) 
                                 if img_name.lower().endswith(('.png', '.jpg', '.jpeg')))
        
        processed_images = 0
        
        with h5py.File(EMBEDDINGS_FILE, 'w') as hf:
            try:
                df = pd.read_csv(REGISTRATION_FILE)
            except Exception as e:
                st.error(f"Error reading registration file: {e}")
                df = pd.DataFrame(columns=['Name', 'Email', 'Department', 'Registration Date'])

            for person_name in os.listdir(KNOWN_FACES_DIR):
                person_path = os.path.join(KNOWN_FACES_DIR, person_name)
                if os.path.isdir(person_path):
                    user_row = df[df['Name'] == person_name]
                    email = user_row['Email'].values[0] if not user_row.empty else 'No email'
                    department = user_row['Department'].values[0] if not user_row.empty else 'No department'
                    registration_date = user_row['Registration Date'].values[0] if not user_row.empty else 'Unknown'

                    grp = hf.create_group(person_name)
                    grp.attrs['email'] = email
                    grp.attrs['department'] = department
                    grp.attrs['registration_date'] = registration_date

                    embeddings_list = []
                    
                    for img_name in os.listdir(person_path):
                        if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                            img_path = os.path.join(person_path, img_name)
                            try:
                                img = cv2.imread(img_path)
                                if img is None:
                                    continue
                                
                                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                face_results = self.detect_and_align_faces(img)
                                
                                if face_results:
                                    face_img = face_results[0]['face']
                                    embedding = self.extract_vgg_features(face_img)
                                    embeddings_list.append(embedding)
                                    
                                    augmented_faces = self.augment_face_image(face_img)
                                    for aug_face in augmented_faces:
                                        aug_embedding = self.extract_vgg_features(aug_face)
                                        embeddings_list.append(aug_embedding)
                                
                                processed_images += 1
                                progress = int((processed_images / total_images) * 100)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing: {person_name} - {processed_images}/{total_images}")
                            
                            except Exception as e:
                                st.error(f"Error processing {img_path}: {str(e)}")
                    
                    if embeddings_list:
                        grp.create_dataset('embeddings', data=embeddings_list)
                        self.known_faces[person_name] = embeddings_list
                        self.known_user_info[person_name] = {
                            'email': email,
                            'department': department,
                            'registration_date': registration_date
                        }

        progress_bar.empty()
        status_text.empty()
        st.success(f"Computed and saved embeddings for {len(self.known_faces)} persons")

    def align_face(self, face_img, landmarks=None):
        """Align face using MediaPipe landmarks"""
        try:
            h, w = face_img.shape[:2]
            if landmarks:
                # MediaPipe provides normalized coordinates (0-1), convert to pixel values
                left_eye_idx = 33  # Left eye center
                right_eye_idx = 263  # Right eye center
                left_eye = np.array([landmarks.landmark[left_eye_idx].x * w, landmarks.landmark[left_eye_idx].y * h])
                right_eye = np.array([landmarks.landmark[right_eye_idx].x * w, landmarks.landmark[right_eye_idx].y * h])

                # Calculate angle and scale
                dy = right_eye[1] - left_eye[1]
                dx = right_eye[0] - left_eye[0]
                angle = np.degrees(np.arctan2(dy, dx))
                eye_dist = np.sqrt(dx**2 + dy**2)
                desired_dist = w * 0.3
                scale = desired_dist / eye_dist if eye_dist > 0 else 1.0

                # Rotate and scale
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, scale)
                aligned_face = cv2.warpAffine(face_img, M, (w, h))
                return aligned_face
        except Exception as e:
            print(f"Face alignment error: {e}")
        return face_img

    def augment_face_image(self, img):
        """Generate augmented versions of a face image"""
        augmented_images = []
        
        for angle in [-3, 3]:
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(img, M, (w, h))
            augmented_images.append(rotated)
        
        for alpha in [0.85, 1.15]:
            adjusted = cv2.convertScaleAbs(img, alpha=alpha, beta=0)
            augmented_images.append(adjusted)
        
        return augmented_images

    def recognize_face_ensemble(self, face_img, min_votes=3):
        samples = [face_img]
        h, w = face_img.shape[:2]
        
        shifts = [(5, 0), (-5, 0), (0, 5), (0, -5)]
        for dx, dy in shifts:
            if dx > 0:
                shifted = face_img[:, dx:]
            elif dx < 0:
                shifted = face_img[:, :w+dx]
            elif dy > 0:
                shifted = face_img[dy:, :]
            elif dy < 0:
                shifted = face_img[:h+dy, :]
                
            if shifted.size > 0:
                shifted = cv2.resize(shifted, (w, h))
                samples.append(shifted)
        
        votes = {}
        confidences = {}
        
        for sample in samples:
            face_embedding = self.extract_vgg_features(sample)
            name, distance = self.compare_embeddings(face_embedding)
            
            votes[name] = votes.get(name, 0) + 1
            if name not in confidences or distance < confidences[name]:
                confidences[name] = distance
        
        max_votes = 0
        winner = "Unknown"
        
        for name, vote_count in votes.items():
            if vote_count > max_votes:
                max_votes = vote_count
                winner = name
        
        if max_votes >= min_votes and winner != "Unknown":
            return winner, confidences.get(winner, 1.0)
        return "Unknown", 1.0

    def compare_embeddings(self, face_embedding):
        name = "Unknown"
        global_min_distance = 1.0
        
        for person_name, embeddings in self.known_faces.items():
            threshold = self.person_thresholds.get(person_name, SIMILARITY_THRESHOLD)
            min_distance = 1.0
            for stored_embedding in embeddings:
                cosine_dist = cosine(face_embedding, stored_embedding)
                if cosine_dist < min_distance:
                    min_distance = cosine_dist
            
            if min_distance < threshold and min_distance < global_min_distance:
                name = person_name
                global_min_distance = min_distance
        
        return name, global_min_distance

    def recognize_face(self, face_img):
        name, confidence = self.recognize_face_ensemble(face_img)
        
        if name == "Unknown" and self.alignment_available:
            try:
                rgb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
                landmark_results = self.face_mesh.process(rgb_img)
                if landmark_results.multi_face_landmarks:
                    aligned_face = self.align_face(face_img, landmark_results.multi_face_landmarks[0])
                    name, confidence = self.recognize_face_ensemble(aligned_face)
            except Exception as e:
                pass
        
        return name, confidence

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

    def preprocess_image(self, img):
        if img.shape[0] != 224 or img.shape[1] != 224:
            resized = cv2.resize(img, (224, 224))
        else:
            resized = img.copy()
        
        if resized.shape[2] == 3 and resized[0,0,0] > resized[0,0,2]:
            rgb_img = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = resized
        
        normalized = rgb_img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized = (normalized - mean) / std
        preprocessed = np.expand_dims(normalized, axis=0)
        return preprocessed

    def extract_vgg_features(self, face_img):
        preprocessed = self.preprocess_image(face_img)
        features = self.vgg_feature_model.predict(preprocessed, verbose=0)
        return features[0]

    def detect_and_align_faces(self, frame):
        """Detect and align faces using MediaPipe"""
        results = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detection_results = self.face_detector.process(rgb_frame)
        if detection_results.detections:
            for detection in detection_results.detections:
                bbox = detection.location_data.relative_bounding_box
                h, w = frame.shape[:2]
                x1 = int(bbox.xmin * w)
                y1 = int(bbox.ymin * h)
                box_w = int(bbox.width * w)
                box_h = int(bbox.height * h)
                
                margin_x = int(box_w * 0.1)
                margin_y = int(box_h * 0.1)
                x1 = max(0, x1 - margin_x)
                y1 = max(0, y1 - margin_y)
                x2 = min(w, x1 + box_w + 2 * margin_x)
                y2 = min(h, y1 + box_h + 2 * margin_y)
                
                face_img = frame[y1:y2, x1:x2]
                
                landmark_results = self.face_mesh.process(rgb_frame[y1:y2, x1:x2])
                if landmark_results.multi_face_landmarks:
                    aligned_face = self.align_face(face_img, landmark_results.multi_face_landmarks[0])
                    results.append({
                        'face': aligned_face if aligned_face is not None else face_img,
                        'box': (x1, y1, x2-x1, y2-y1)
                    })
                else:
                    results.append({
                        'face': face_img,
                        'box': (x1, y1, x2-x1, y2-y1)
                    })
        
        if not results:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            for (x, y, w, h) in faces:
                margin_x = int(w * 0.1)
                margin_y = int(h * 0.1)
                x1 = max(0, x - margin_x)
                y1 = max(0, y - margin_y)
                x2 = min(frame.shape[1], x + w + margin_x)
                y2 = min(frame.shape[0], y + h + margin_y)
                
                face_img = frame[y1:y2, x1:x2]
                results.append({
                    'face': face_img,
                    'box': (x1, y1, x2-x1, y2-y1)
                })
        
        return results

def main():
    global SIMILARITY_THRESHOLD

    st.set_page_config(page_title="Face Recognition Attendance System", 
                       page_icon=":camera:", 
                       layout="wide", 
                       initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    .reportview-container {
        background: #1E1E1E;
        color: white;
    }
    .sidebar .sidebar-content {
        background: #2C2C2C;
    }
    .stTextInput>div>div>input {
        color: white;
        background-color: #3C3C3C;
    }
    .stButton>button {
        color: white;
        background-color: #4CAF50;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
    """, unsafe_allow_html=True)

    face_system = FaceRecognitionSystem()

    st.sidebar.title("Face Recognition Attendance")
    menu = st.sidebar.radio("Navigation", 
        ["Attendance", "Register User", "View Attendance", "View Registered Users", "System Settings"])

    if menu == "Attendance":
        st.title("Attendance Marking")
    
    col1, col2 = st.columns(2)
    with col1:
        start_camera = st.button("Start Camera", key="start_camera_btn_unique")
    with col2:
        stop_camera = st.button("Stop Camera", key="stop_camera_btn_unique")
    
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False

    if start_camera:
        st.session_state.camera_running = True
    
    if stop_camera:
        st.session_state.camera_running = False

    with st.sidebar.expander("Recognition Settings"):
        confidence_threshold = st.slider("Recognition Confidence", 0.0, 1.0, SIMILARITY_THRESHOLD, 0.01)
        min_detections = st.slider("Minimum consecutive detections", 1, 10, 3)
        
    if st.session_state.camera_running:
        frame_placeholder = st.empty()
        status_placeholder = st.empty()
        detected_faces = st.empty()

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        
        frame_count = 0
        face_trackers = {}
        last_recognized = {}
        recognized_names = []  # Initialize here, outside the if block

        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break

            frame_count += 1
            display_frame = frame.copy()

            if frame_count % 3 == 0:
                face_results = face_system.detect_and_align_faces(frame)
                
                recognized_names = []  # Reset on detection frames
                for i, face_result in enumerate(face_results):
                    face_img = face_result['face']
                    x, y, w, h = face_result['box']
                    
                    try:
                        name, confidence = face_system.recognize_face(face_img)
                        
                        if i not in face_trackers:
                            face_trackers[i] = {'counts': {}, 'total': 0}
                            
                        face_trackers[i]['counts'][name] = face_trackers[i]['counts'].get(name, 0) + 1
                        face_trackers[i]['total'] += 1
                        
                        most_common_name = max(face_trackers[i]['counts'].items(), key=lambda x: x[1])[0]
                        detection_count = face_trackers[i]['counts'][most_common_name]
                        detection_ratio = detection_count / face_trackers[i]['total']
                        
                        final_name = most_common_name if detection_count >= min_detections and detection_ratio > 0.6 else "Unknown"
                        color = (0, 255, 0) if final_name != "Unknown" else (0, 0, 255)
                        
                        conf_pct = int((1 - confidence) * 100) if confidence <= 1 else 0
                        
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                        if final_name != "Unknown":
                            attendance_time = None
                            if final_name not in last_recognized or datetime.datetime.now() - last_recognized[final_name] > datetime.timedelta(minutes=5):
                                attendance_time = face_system.mark_attendance(final_name)
                                last_recognized[final_name] = datetime.datetime.now()
                            
                            cv2.putText(display_frame, f"{final_name} ({conf_pct}%)", 
                                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                            
                            if final_name not in recognized_names:
                                recognized_names.append(final_name)
                                
                            if attendance_time:
                                st.success(f"Attendance marked for {final_name} at {attendance_time}")
                        else:
                            cv2.putText(display_frame, "Unknown", 
                                        (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                                            
                    except Exception as e:
                        print(f"Error in recognition: {str(e)}")
            
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(display_frame, channels="RGB")
            
            if recognized_names:
                status_placeholder.success(f"Recognized: {', '.join(recognized_names)}")
            else:
                status_placeholder.info("No known faces detected")
        
        cap.release()
    
    elif menu == "Register User":
        st.title("Register New User")
        
        with st.form("registration_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            department = st.text_input("Department")
            face_samples = st.slider("Number of face samples to capture", 1, 10, 5)
            
            submitted = st.form_submit_button("Register")
        
        if submitted and name and email and department:
            reg_time, safe_name = face_system.register_user(name, email, department)
            user_dir = os.path.join(KNOWN_FACES_DIR, safe_name)
            os.makedirs(user_dir, exist_ok=True)
            
            st.success(f"User {name} registered successfully!")
            st.info("Now let's capture face samples. Please look at the camera.")
            
            st.subheader("Face Capture")
            capture_btn = st.button("Start Capture")
            
            if capture_btn:
                cap = cv2.VideoCapture(0)
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                
                frame_placeholder = st.empty()
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                samples_taken = 0
                while samples_taken < face_samples:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Failed to capture frame")
                        break
                        
                    face_results = face_system.detect_and_align_faces(frame)
                    
                    display_frame = frame.copy()
                    for face_result in face_results:
                        x, y, w, h = face_result['box']
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    frame_placeholder.image(display_frame, channels="RGB")
                    
                    if face_results:
                        face_img = face_results[0]['face']
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        filename = os.path.join(user_dir, f"face_{timestamp}.jpg")
                        cv2.imwrite(filename, face_img)
                        
                        samples_taken += 1
                        progress = int((samples_taken / face_samples) * 100)
                        progress_bar.progress(progress)
                        status_text.text(f"Captured {samples_taken}/{face_samples} samples")
                        
                        time.sleep(0.5)
                
                cap.release()
                
                if samples_taken == face_samples:
                    st.success(f"Successfully captured {samples_taken} face samples!")
                    st.info("Updating face recognition database...")
                    face_system.compute_face_embeddings()
                    st.success("Face recognition database updated successfully!")
                else:
                    st.warning(f"Only captured {samples_taken}/{face_samples} samples. Registration may not be optimal.")
    
    elif menu == "View Attendance":
        st.title("Attendance Records")
        
        try:
            df = pd.read_csv(ATTENDANCE_FILE)
            
            st.subheader("Filter by Date")
            if not df.empty:
                df['Time'] = pd.to_datetime(df['Time'])
                
                min_date = df['Time'].min().date()
                max_date = df['Time'].max().date()
                
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input("From", min_date, min_value=min_date, max_value=max_date)
                with col2:
                    end_date = st.date_input("To", max_date, min_value=min_date, max_value=max_date)
                
                mask = (df['Time'].dt.date >= start_date) & (df['Time'].dt.date <= end_date)
                filtered_df = df.loc[mask]
                
                summary = filtered_df.groupby([filtered_df['Time'].dt.date, 'Name']).first().reset_index()
                summary = summary.rename(columns={'Time': 'First Attendance'})
                
                st.subheader("Attendance Summary")
                if not summary.empty:
                    st.dataframe(summary)
                    
                    csv = summary.to_csv(index=False)
                    b64 = base64.b64encode(csv.encode()).decode()
                    href = f'<a href="data:file/csv;base64,{b64}" download="attendance_summary.csv">Download Summary CSV</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    st.subheader("Attendance by Person")
                    person_counts = filtered_df['Name'].value_counts().reset_index()
                    person_counts.columns = ['Name', 'Attendance Count']
                    st.bar_chart(person_counts.set_index('Name'))
                else:
                    st.info("No attendance records found for selected date range.")
            else:
                st.info("No attendance records available.")
                
        except Exception as e:
            st.error(f"Error reading attendance data: {str(e)}")
    
    elif menu == "View Registered Users":
        st.title("Registered Users")
        
        try:
            df = pd.read_csv(REGISTRATION_FILE)
            
            st.subheader("User List")
            if not df.empty:
                st.dataframe(df)
                
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()
                href = f'<a href="data:file/csv;base64,{b64}" download="registered_users.csv">Download User List</a>'
                st.markdown(href, unsafe_allow_html=True)
                
                st.subheader("Users by Department")
                dept_counts = df['Department'].value_counts().reset_index()
                dept_counts.columns = ['Department', 'Count']
                st.bar_chart(dept_counts.set_index('Department'))
            else:
                st.info("No users registered yet.")
                
        except Exception as e:
            st.error(f"Error reading user data: {str(e)}")
    
    elif menu == "System Settings":
        st.title("System Settings")
    
    # Expander for Recognition Settings
    with st.expander("Recognition Settings"):
        new_threshold = st.slider("Global Recognition Threshold", 0.0, 1.0, SIMILARITY_THRESHOLD, 0.01)
        
        if st.button("Update Threshold"):
            SIMILARITY_THRESHOLD = new_threshold
            face_system.initialize_thresholds()
            st.success(f"Recognition threshold updated to {new_threshold}")
    
    # Expander for Database Management (now a sibling, not nested)
    with st.expander("Database Management"):
        if st.button("Rebuild Face Database"):
            with st.spinner("Rebuilding face database..."):
                face_system.compute_face_embeddings()
            st.success("Face database rebuilt successfully!")
    
    # Expander for System Information
    with st.expander("System Information"):
        st.info(f"Number of registered users: {len(face_system.known_faces)}")
        st.info(f"Face alignment available: {face_system.alignment_available}")
        
        st.subheader("Person-specific thresholds")
        threshold_data = [{"Person": p, "Threshold": t} for p, t in face_system.person_thresholds.items()]
        if threshold_data:
            st.dataframe(pd.DataFrame(threshold_data))

if __name__ == "__main__":
    main()