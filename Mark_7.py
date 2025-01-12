import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pyttsx3
import os

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Constants for distance calculation
K = 1000

# Load YOLO model
required_files = ['yolov4-tiny.weights', 'yolov4-tiny.cfg', 'coco.names']
if not all(os.path.exists(file) for file in required_files):
    print("Required YOLO files are missing!")
    exit(1)

net = cv2.dnn.readNet('yolov4-tiny.weights', 'yolov4-tiny.cfg')
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Load class labels for YOLO
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Load facial expression dataset
dataset_path = "facial_expression_data_new.csv"
data = pd.read_csv(dataset_path)
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Initialize MediaPipe Pose and Face Mesh
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=False, max_num_faces=1)


# Function to infer action from pose
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

# Start video capture
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

try:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        height, width, _ = frame.shape

        # Convert frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # MediaPipe Pose and Face Mesh
        pose_results = pose.process(rgb_frame)
        face_results = face_mesh.process(rgb_frame)

        # YOLO Object Detection
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward([net.getLayerNames()[i - 1] for i in net.getUnconnectedOutLayers()])

        boxes, confidences, class_ids = [], [], []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype(int)
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        # Display detected objects and calculate feedback
        detected_objects = []
        if len(indexes) > 0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                detected_objects.append((label, x + w // 2))

        if detected_objects:
            feedback = []
            for label, center_x in detected_objects:
                direction = "left" if center_x < width // 3 else "right" if center_x > 2 * width // 3 else "center"
                feedback.append(f"{label} on your {direction}")
            description = ", ".join(feedback)
            engine.say(description)
            engine.runAndWait()

        # Pose Landmarks and Distance
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = pose_results.pose_landmarks.landmark
            nose = landmarks[mp_pose.PoseLandmark.NOSE]
            if nose.visibility > 0.5:
                distance = distance()
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            action = infer_action(landmarks)
            cv2.putText(frame, f"Action: {action}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Face Mesh and Expression Detection
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                landmarks = []
                for lm in face_landmarks.landmark:
                    landmarks.extend([lm.x, lm.y])
                expression = knn.predict([landmarks])[0]
                cv2.putText(frame, f"Expression: {expression}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Integrated Detection", frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    engine.stop()
