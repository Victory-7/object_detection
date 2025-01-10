import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

# Calibration constant for distance calculation (adjust based on your setup)
K = 1000

# Load the dataset
dataset_path = "facial_expression_data_new.csv"
data = pd.read_csv(dataset_path)

# Prepare data for training
X = data.iloc[:, :-1].values  # Landmark positions
y = data.iloc[:, -1].values   # Expressions

# Train a KNN classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=False, max_num_faces=1)

# Function to calculate distance based on the size of an object
def calculate_distance(area, K):
    if area > 0:
        return K / np.sqrt(area)
    else:
        return float('inf')

# Function to infer actions based on node movement
def infer_action(landmarks):
    if landmarks:
        left_hand = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
        right_hand = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
        nose = landmarks[mp_pose.PoseLandmark.NOSE.value]

        if left_hand.y < nose.y and right_hand.y < nose.y:
            return "Hands raised"
        elif left_hand.y > nose.y and right_hand.y > nose.y:
            return "Hands lowered"
        else:
            return "Neutral position"
    return "No action detected"

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Capture a frame
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # Convert the image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame for pose estimation
        pose_results = pose.process(rgb_frame)

        # Process the frame for face mesh
        face_results = face_mesh.process(rgb_frame)

        # Annotate the frame with pose landmarks
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate distance based on the nose node
            nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            if nose.visibility > 0.5:  # Only consider visible landmarks
                area = 10000  # Example fixed area; adjust as needed
                distance = calculate_distance(area, K)
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Infer action
            action = infer_action(pose_results.pose_landmarks.landmark)
            cv2.putText(frame, f"Action: {action}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Detect facial expressions
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])

                # Predict expression
                expression = knn.predict([landmarks])[0]
                cv2.putText(frame, f"Expression: {expression}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Pose and Facial Expression Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
