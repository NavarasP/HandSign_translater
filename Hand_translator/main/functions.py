import pandas as pd
import numpy as np
import tensorflow as tf
import cv2
import joblib
import mediapipe as mp
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from keras.initializers import Orthogonal
import pickle
from django.conf import settings
import cv2
import numpy as np
import time

model1_path = settings.MODEL1_PATH
model1_encoder = settings.MODEL1_ENCODER




custom_objects = {'Orthogonal': Orthogonal}
loaded_model = tf.keras.models.load_model(model1_path, custom_objects=custom_objects)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose
mp_face_mesh = mp.solutions.face_mesh


def extract_landmarks_mediapipe(frame):
    with mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5) as hands:
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process hand landmarks
                hands_results = hands.process(frame_rgb)
                left_hand_landmarks, right_hand_landmarks = [],[]
                if hands_results.multi_hand_landmarks:
                    for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                        if handedness.classification[0].label == 'Left':
                            left_hand_landmarks = hand_landmarks
                        elif handedness.classification[0].label == 'Right':
                            right_hand_landmarks = hand_landmarks

                # Process pose landmarks
                pose_results = pose.process(frame_rgb)
                pose_landmarks = pose_results.pose_landmarks


                # Process face landmarks
                face_results = face_mesh.process(frame_rgb)
                face_landmarks = face_results.multi_face_landmarks

                

    return left_hand_landmarks, right_hand_landmarks, pose_landmarks, face_landmarks

max_face_index = 467
max_left_hand_index = 20
max_right_hand_index = 20
max_pose_index = 32

face_columns = [f"face_{i}" for i in range(max_face_index + 1)]
left_hand_columns = [f"left_hand_{i}" for i in range(max_left_hand_index + 1)]
right_hand_columns = [f"right_hand_{i}" for i in range(max_right_hand_index + 1)]
pose_columns = [f"pose_{i}" for i in range(max_pose_index + 1)]

header =      [f"{col}_{coord}" for col in face_columns for coord in ['x', 'y']] + \
              [f"{col}_{coord}" for col in left_hand_columns for coord in ['x', 'y']] + \
              [f"{col}_{coord}" for col in right_hand_columns for coord in ['x', 'y']] + \
              [f"{col}_{coord}" for col in pose_columns for coord in ['x', 'y']] 







def landmarks_to_df(left_hand_landmarks, right_hand_landmarks, pose_landmarks, face_landmarks, header):
    # Initialize dictionaries to store landmark data
    landmarks_data = {}

    # Process face landmarks
    if face_landmarks:
        for i, landmark_list in enumerate(face_landmarks):
            for j, lm in enumerate(landmark_list.landmark):
                landmarks_data[f"face_{j}_x"] = lm.x
                landmarks_data[f"face_{j}_y"] = lm.y
            # Fill missing face landmarks with zeros
            for j in range(len(landmark_list.landmark), max_face_index + 1):
                landmarks_data[f"face_{j}_x"] = 0.0
                landmarks_data[f"face_{j}_y"] = 0.0
    else:
        # Fill all face landmarks with zeros if face_landmarks is None
        for j in range(max_face_index + 1):
            landmarks_data[f"face_{j}_x"] = 0.0
            landmarks_data[f"face_{j}_y"] = 0.0

    # Process left hand landmarks
    if left_hand_landmarks:
        for i, lm in enumerate(left_hand_landmarks.landmark):
            landmarks_data[f"left_hand_{i}_x"] = lm.x
            landmarks_data[f"left_hand_{i}_y"] = lm.y
        # Fill missing left hand landmarks with zeros
        for i in range(len(left_hand_landmarks.landmark), max_left_hand_index + 1):
            landmarks_data[f"left_hand_{i}_x"] = 0.0
            landmarks_data[f"left_hand_{i}_y"] = 0.0
    else:
        # Fill all left hand landmarks with zeros if left_hand_landmarks is None
        for i in range(max_left_hand_index + 1):
            landmarks_data[f"left_hand_{i}_x"] = 0.0
            landmarks_data[f"left_hand_{i}_y"] = 0.0

    # Process right hand landmarks
    if right_hand_landmarks:
        for i, lm in enumerate(right_hand_landmarks.landmark):
            landmarks_data[f"right_hand_{i}_x"] = lm.x
            landmarks_data[f"right_hand_{i}_y"] = lm.y
        # Fill missing right hand landmarks with zeros
        for i in range(len(right_hand_landmarks.landmark), max_right_hand_index + 1):
            landmarks_data[f"right_hand_{i}_x"] = 0.0
            landmarks_data[f"right_hand_{i}_y"] = 0.0
    else:
        # Fill all right hand landmarks with zeros if right_hand_landmarks is None
        for i in range(max_right_hand_index + 1):
            landmarks_data[f"right_hand_{i}_x"] = 0.0
            landmarks_data[f"right_hand_{i}_y"] = 0.0

    # Process pose landmarks
    if pose_landmarks:
        for i, lm in enumerate(pose_landmarks.landmark):
            landmarks_data[f"pose_{i}_x"] = lm.x
            landmarks_data[f"pose_{i}_y"] = lm.y
        # Fill missing pose landmarks with zeros
        for i in range(len(pose_landmarks.landmark), max_pose_index + 1):
            landmarks_data[f"pose_{i}_x"] = 0.0
            landmarks_data[f"pose_{i}_y"] = 0.0
    else:
        # Fill all pose landmarks with zeros if pose_landmarks is None
        for i in range(max_pose_index + 1):
            landmarks_data[f"pose_{i}_x"] = 0.0
            landmarks_data[f"pose_{i}_y"] = 0.0

    # Create DataFrame from extracted landmark data
    df = pd.DataFrame([landmarks_data], columns=header)


    return df

def preprocess_landmarks(df):
    left_hand_columns = [col for col in df.columns if col.startswith('left_hand')]
    right_hand_columns = [col for col in df.columns if col.startswith('right_hand')]
    pose_columns = [col for col in df.columns if col.startswith('pose')]

    # Ensure the data is in the correct shape (number_of_samples, number_of_frames, number_of_features_per_frame)
    def reshape_data(df, columns, num_frames):
        data = df[columns].values
        num_samples = len(df) // num_frames
        data = data.reshape(num_samples, num_frames, len(columns))
        return data


    # Assuming num_frames is known
    num_frames = 1 # This should be the length of the time series

    left_hand_data = reshape_data(df, left_hand_columns, num_frames)
    right_hand_data = reshape_data(df, right_hand_columns, num_frames)
    pose_data = reshape_data(df, pose_columns, num_frames)


    return left_hand_data, right_hand_data, pose_data


with open(model1_encoder, 'rb') as file:
    label_encoder = pickle.load(file)


camera = cv2.VideoCapture(0)



def generate_frames():
    prev_time = time.time()
    last_prediction_time = time.time()
    no_prediction_counter = 0

    while True:
        success, frame = camera.read()
        if not success:
            print('not success')
            break
        else:
            frame = cv2.flip(frame, 1)  # Flip the frame horizontally

            # Extract landmarks using MediaPipe
            left_hand, right_hand, pose, face = extract_landmarks_mediapipe(frame)

            current_time = time.time()
            if current_time - prev_time >= 1:  # Process frame every second
                prev_time = current_time

                if left_hand or right_hand:
                    # Preprocess landmarks
                    df = landmarks_to_df(left_hand, right_hand, pose, face, header)
                    left_hand_input, right_hand_input, pose_input = preprocess_landmarks(df)

                    # Make prediction
                    prediction = loaded_model.predict([left_hand_input, right_hand_input, pose_input])

                    if not np.isnan(prediction).any():
                        predicted_class_index = np.argmax(prediction, axis=1)
                        predicted_class_label = label_encoder.inverse_transform(predicted_class_index)
                        confidence = prediction[0, predicted_class_index][0]

                        if confidence > 0.95:
                            no_prediction_counter = 0
                            last_prediction_time = current_time
                            label = predicted_class_label[0]  # Assuming single prediction per frame
                            
                            print(f"Predicted sign: {label} with confidence {confidence:.2f}")

                            with open('predicted_labels.txt', 'a') as file:
                                file.write(label)

                    else:
                        print("Prediction contains NaN values")

                else:
                    if current_time - last_prediction_time >= 3:
                        no_prediction_counter += 1
                        last_prediction_time = current_time
                        with open('predicted_labels.txt', 'a') as file:
                            file.write(' ')

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


