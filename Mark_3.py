import cv2
import mediapipe as mp
import numpy as np

# Calibration constant for distance calculation (adjust based on your setup)
K = 1000

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

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
        results = pose.process(rgb_frame)

        # Annotate the frame with pose landmarks
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate distance based on the nose node
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            if nose.visibility > 0.5:  # Only consider visible landmarks
                # Assume a fixed bounding box area for simplicity
                area = 10000  # Example fixed area; adjust as needed
                distance = calculate_distance(area, K)
                cv2.putText(frame, f"Distance: {distance:.2f} cm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Infer action
            action = infer_action(results.pose_landmarks.landmark)
            cv2.putText(frame, f"Action: {action}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        # Display the frame
        cv2.imshow("Pose Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
