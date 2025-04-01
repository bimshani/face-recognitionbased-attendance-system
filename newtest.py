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

# Configuration
MODEL_FOLDER = 'vgg_model'
FEATURE_MODEL_PATH = os.path.join(MODEL_FOLDER, 'vggface_features.h5')
KNOWN_FACES_DIR = 'faces'
ATTENDANCE_FILE = 'attendance.csv'
REGISTRATION_FILE = 'registered_users.csv'
EMBEDDINGS_FILE = 'face_embeddings.h5'
SIMILARITY_THRESHOLD = 0.6  # Cosine similarity threshold (lower is more similar)

# Camera settings
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

class FaceRecognitionSystem:
    def __init__(self):
        # Load models and known faces
        self.load_models_and_faces()

    def load_models_and_faces(self):
        # Ensure necessary directories exist
        os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
        
        # Create attendance and registration files if they don't exist
        for file_path in [ATTENDANCE_FILE, REGISTRATION_FILE]:
            if not os.path.exists(file_path):
                with open(file_path, "w") as f:
                    if file_path == ATTENDANCE_FILE:
                        f.write("Name,Time\n")
                    else:
                        f.write("Name,Email,Department,Registration Date\n")

        # Load the VGG feature extraction model
        try:
            self.vgg_feature_model = load_model(FEATURE_MODEL_PATH)
            print("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            return None

        # Load known faces and embeddings
        self.known_faces = {}
        self.known_user_info = {}
        self.attended_persons = set()

        # Load embeddings if exists
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
            # Compute embeddings if file doesn't exist
            self.compute_face_embeddings()

    def compute_face_embeddings(self):
        # Progress bar for loading faces
        progress_bar = st.progress(0)
        status_text = st.empty()

        # Load known faces
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        if face_cascade.empty():
            st.error("Could not load face cascade classifier")
            return

        total_images = sum([len(files) for r, d, files in os.walk(KNOWN_FACES_DIR) if any(file.lower().endswith(('.png', '.jpg', '.jpeg')) for file in files)])
        processed_images = 0

        # Prepare to save embeddings
        with h5py.File(EMBEDDINGS_FILE, 'w') as hf:
            # Read registration file to get user details
            try:
                df = pd.read_csv(REGISTRATION_FILE)
            except Exception as e:
                st.error(f"Error reading registration file: {e}")
                return

            for person_name in os.listdir(KNOWN_FACES_DIR):
                person_path = os.path.join(KNOWN_FACES_DIR, person_name)
                if os.path.isdir(person_path):
                    # Get user details from DataFrame
                    user_row = df[df['Name'] == person_name]
                    email = user_row['Email'].values[0] if not user_row.empty else 'No email'
                    department = user_row['Department'].values[0] if not user_row.empty else 'No department'
                    registration_date = user_row['Registration Date'].values[0] if not user_row.empty else 'Unknown'

                    # Create a group for each person
                    grp = hf.create_group(person_name)
                    grp.attrs['email'] = email
                    grp.attrs['department'] = department
                    grp.attrs['registration_date'] = registration_date

                    # Store embeddings
                    embeddings_list = []
                    
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
                                        embeddings_list.append(embedding)
                                
                                processed_images += 1
                                progress = int((processed_images / total_images) * 100)
                                progress_bar.progress(progress)
                                status_text.text(f"Processing: {person_name} - {processed_images}/{total_images}")
                            
                            except Exception as e:
                                st.error(f"Error processing {img_path}: {str(e)}")
                    
                    # Save embeddings for this person
                    grp.create_dataset('embeddings', data=embeddings_list)
                    
                    # Store in memory
                    self.known_faces[person_name] = embeddings_list
                    self.known_user_info[person_name] = {
                        'email': email,
                        'department': department,
                        'registration_date': registration_date
                    }

        progress_bar.empty()
        status_text.empty()
        st.success(f"Computed and saved embeddings for {len(self.known_faces)} persons")

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
            return current_time
        return None

    def register_user(self, name, email, department):
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Sanitize the name for file system
        safe_name = re.sub(r'[^\w\-_\. ]', '_', name.strip())
        
        with open(REGISTRATION_FILE, "a") as f:
            f.write(f"{safe_name},{email},{department},{current_time}\n")
        return current_time, safe_name

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

def main():
    # Set page configuration
    st.set_page_config(page_title="Face Recognition Attendance System", 
                       page_icon=":camera:", 
                       layout="wide", 
                       initial_sidebar_state="expanded")

    # Custom CSS for dark theme
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

    # Initialize the face recognition system
    face_system = FaceRecognitionSystem()

    # Sidebar for navigation
    st.sidebar.title("Face Recognition Attendance")
    menu = st.sidebar.radio("Navigation", 
        ["Attendance", "Register User", "View Attendance", "View Registered Users"])

    if menu == "Attendance":
        st.title("Attendance Marking")
        
        # Camera feed and recognition
        col1, col2 = st.columns(2)
        with col1:
            start_camera = st.button("Start Camera", key="start_camera_btn_unique")
        with col2:
            stop_camera = st.button("Stop Camera", key="stop_camera_btn_unique")
        
        # Session state to manage camera
        if 'camera_running' not in st.session_state:
            st.session_state.camera_running = False

        if start_camera:
            st.session_state.camera_running = True
        
        if stop_camera:
            st.session_state.camera_running = False

        if st.session_state.camera_running:
            # Camera placeholder
            frame_placeholder = st.empty()
            detected_faces = st.empty()

            # Open webcam
            cap = cv2.VideoCapture(0)
            
            # Tracking variables
            frame_count = 0
            last_recognized = {}

            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break

                frame_count += 1
                display_frame = frame.copy()

                # Face detection every 5 frames
                if frame_count % 5 == 0:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

                    recognized_names = []
                    for (x, y, w, h) in faces:
                        face_img = frame[y:y+h, x:x+w]
                        try:
                            face_embedding = face_system.extract_vgg_features(face_img)
                            name = face_system.recognize_face(face_embedding)

                            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                            cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                            cv2.putText(display_frame, name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                            if name != "Unknown":
                                # Mark attendance with throttling
                                current_time = datetime.datetime.now()
                                if (name not in last_recognized or 
                                    (current_time - last_recognized.get(name, datetime.datetime.min)).total_seconds() > 60):
                                    attendance_time = face_system.mark_attendance(name)
                                    if attendance_time:
                                        last_recognized[name] = current_time
                                        recognized_names.append(f"{name} - Attendance Marked")

                        except Exception as e:
                            st.error(f"Recognition error: {e}")

                    # Display recognized faces
                    if recognized_names:
                        detected_faces.success("\n".join(recognized_names))

                # Convert frame for Streamlit
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(frame_rgb, channels="RGB")

                # Add a small delay to reduce CPU usage
                st.empty()

            # Release webcam
            cap.release()

    elif menu == "Register User":
        st.title("User Registration")
        
        # Registration form
        with st.form("registration_form"):
            name = st.text_input("Full Name")
            email = st.text_input("Email")
            department = st.text_input("Department")
            uploaded_files = st.file_uploader("Upload Face Images", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)
            submit_button = st.form_submit_button("Register User")

            if submit_button:
                if name and email and department and uploaded_files:
                    # Save registration
                    registration_time, safe_name = face_system.register_user(name, email, department)
                    
                    # Create user directory if not exists
                    user_dir = os.path.join(KNOWN_FACES_DIR, safe_name)
                    os.makedirs(user_dir, exist_ok=True)
                    
                    # Save uploaded images
                    for uploaded_file in uploaded_files:
                        # Generate a unique filename to prevent overwrites
                        filename = f"{safe_name}_{uploaded_file.name}"
                        file_path = os.path.join(user_dir, filename)
                        
                        # Save the file
                        with open(file_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    # Recompute embeddings to include new user
                    face_system.compute_face_embeddings()
                    
                    st.success(f"User {safe_name} registered successfully at {registration_time}")
                    st.success(f"Uploaded {len(uploaded_files)} face images")
                else:
                    st.error("Please fill all fields and upload at least one image")

    elif menu == "View Attendance":
        st.title("Attendance Records")
        
        try:
            # Read and display attendance
            df = pd.read_csv(ATTENDANCE_FILE)
            st.dataframe(df)
            
            # Download button
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.download_button(
                label="Download Attendance CSV",
                data=csv,
                file_name="attendance_records.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error reading attendance file: {e}")

    elif menu == "View Registered Users":
        st.title("Registered Users")
        
        try:
            # Read and display registered users
            df = pd.read_csv(REGISTRATION_FILE)
            st.dataframe(df)
            
            # Create a data display with more details
            st.subheader("Registered People Details")
            for _, row in df.iterrows():
                with st.expander(f"Details for {row['Name']}"):
                    st.write(f"**Name:** {row['Name']}")
                    st.write(f"**Email:** {row['Email']}")
                    st.write(f"**Department:** {row['Department']}")
                    st.write(f"**Registration Date:** {row['Registration Date']}")
                    
                    # Show face images
                    user_dir = os.path.join(KNOWN_FACES_DIR, row['Name'])
                    if os.path.exists(user_dir):
                        images = [os.path.join(user_dir, img) for img in os.listdir(user_dir) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                        if images:
                            cols = st.columns(len(images))
                            for col, img_path in zip(cols, images):
                                col.image(img_path, caption=f"Face Image - {os.path.basename(img_path)}")
            
            # Download button
            csv = df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            st.download_button(
                label="Download Registered Users CSV",
                data=csv,
                file_name="registered_users.csv",
                mime="text/csv"
            )
        except Exception as e:
            st.error(f"Error reading registration file: {e}")

if __name__ == "__main__":
    main()