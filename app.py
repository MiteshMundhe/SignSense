import streamlit as st
from PIL import Image
import os
import json
from streamlit_lottie import st_lottie
import pickle
import cv2
import mediapipe as mp
import numpy as np

# --- PATHS AND MODEL LOADING ---
# All paths are now correct based on our new folder structure.
MODEL_PATH = 'model/model.p'
ANIMATION_PATH = "assets/Animation/Ani.json"
IMAGE_PATH = "assets/image/sign.png"

# Load the trained model
try:
    with open(MODEL_PATH, 'rb') as f:
        model_dict = pickle.load(f)
    model = model_dict['model']
except FileNotFoundError:
    st.error(f"Error: The model file was not found at {MODEL_PATH}. Please make sure the file exists.")
    st.stop()


# --- MEDIAPIPE INITIALIZATION ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Dictionary to map model predictions to characters
labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 
               12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 
               23: 'X', 24: 'Y', 25: 'Z', 26: '0', 27: '1', 28: '2', 29: '3', 30: '4', 31: '5', 32: '6', 33: '7', 
               34: '8', 35: '9', 36: 'I Love You', 37: 'EAT', 38: 'OK', 39: 'HELP'}


# --- UI SETUP ---
st.set_page_config(page_title="SignSense", layout="wide")

# Function to load Lottie animation
def load_lottiefile(filepath: str):
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Error: The animation file was not found at {ANIMATION_PATH}.")
        return None

# Load UI assets
lottie_coding = load_lottiefile(ANIMATION_PATH)
try:
    img_arg = Image.open(IMAGE_PATH)
except FileNotFoundError:
    st.error(f"Error: The image file was not found at {IMAGE_PATH}.")
    st.stop()

# --- HEADER SECTION ---
with st.container():
    left_column, right_column = st.columns([3, 1])
    with left_column:
        st.title("SignSense")
        st.write("## Real-Time Hand Gesture Recognition") 
    with right_column:
        # Corrected parameter to remove deprecation warning
        st.image(img_arg, use_container_width=True)

# --- INFO SECTION ---
with st.container():
    st.write("---")
    left_column, right_column = st.columns([1, 2])
    with left_column:
        if lottie_coding:
            st_lottie(
                lottie_coding,
                speed=0.25,
                loop=True,
                quality="low",
                height=300,
                width=300,
                key="coding",
            )
    with right_column:
        st.write("") # Vertical spacer
        st.subheader("How you interact matters.")
        st.subheader("Do it together with SignSense.")
        st.subheader("Language is just a medium for us to convey.")
        
st.write("---")

# --- CAMERA CONTROL ---
# Use session state to manage camera status
if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

col1, col2 = st.columns(2)
if col1.button("Open Camera!"):
    st.session_state.run_camera = True

if col2.button("Close Camera!"):
    st.session_state.run_camera = False

# --- REAL-TIME RECOGNITION LOGIC ---
if st.session_state.run_camera:
    st.write("Camera is ON")
    camera_box = st.empty()
    cap = cv2.VideoCapture(0)

    while st.session_state.run_camera:
        data_aux = []
        x_ = []
        y_ = []

        ret, frame = cap.read()
        if not ret:
            st.warning("Could not read frame from camera. Please ensure it is not being used by another application.")
            break

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                # Extract landmark data for prediction
                for landmark in hand_landmarks.landmark:
                    x_.append(landmark.x)
                    y_.append(landmark.y)
                
                min_x = min(x_)
                min_y = min(y_)

                for landmark in hand_landmarks.landmark:
                    data_aux.append(landmark.x - min_x)
                    data_aux.append(landmark.y - min_y)

            # Define bounding box
            x1 = int(min(x_) * W) - 10
            y1 = int(min(y_) * H) - 10
            x2 = int(max(x_) * W) + 10
            y2 = int(max(y_) * H) + 10

            # Make prediction
            prediction = model.predict([np.asarray(data_aux)])
            predicted_character = labels_dict.get(int(prediction[0]), 'Unknown')

            # Draw bounding box and predicted character
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
            cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)

        # Display the frame in the Streamlit app
        frame_to_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        camera_box.image(frame_to_display, channels="RGB")
    
    # Release camera when loop breaks
    cap.release()
    cv2.destroyAllWindows()
    st.write("Camera has been closed.")
