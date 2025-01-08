import cv2
import numpy as np

# Define the object color in HSV (Hue, Saturation, Value)
# Example for detecting a red object
hue_min = 0
hue_max = 10  # Adjust based on your object's color
sat_min = 127
sat_max = 255
val_min = 127
val_max = 255

# Calibration constant for distance calculation (adjust based on your setup)
K = 1000

def calculate_distance(area, K):
    """Calculate distance based on the object area."""
    if area > 0:
        return K / np.sqrt(area)
    else:
        return float('inf')  # Return infinity for invalid areas

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

        # Convert the image to HSV color space
        hsv_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create a binary mask based on the defined HSV range
        lower_bound = np.array([hue_min, sat_min, val_min])
        upper_bound = np.array([hue_max, sat_max, val_max])
        mask = cv2.inRange(hsv_img, lower_bound, upper_bound)

        # Perform morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours of the detected objects
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw bounding boxes and calculate distances
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 0:  # Filter out small areas
                x, y, w, h = cv2.boundingRect(contour)
                centroid_x = int(x + w / 2)
                centroid_y = int(y + h / 2)

                # Calculate distance
                distance = calculate_distance(area, K)

                # Draw the bounding box and centroid
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.circle(frame, (centroid_x, centroid_y), 5, (255, 0, 0), -1)

                # Display object information
                label = f"Object: Ball\nDistance: {distance:.2f} cm"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Show the original frame with annotations
        cv2.imshow("Object Detection", frame)

        # Break the loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
