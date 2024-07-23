import cv2
import numpy as np
from jetracer.nvidia_racecar import NvidiaRacecar

# Initialize JetRacer
car = NvidiaRacecar()
car.throttle_gain = 0.5
car.steering_gain = 0.5

# Initialize USB webcam
cap = cv2.VideoCapture(0)  # If the USB webcam is not detected, you might need to change the index to 1 or higher

if not cap.isOpened():
    print("Cannot open camera")
    exit()

prev_center_x = 320

while True:
    ret, frame = cap.read()

    if not ret:
        print("Cannot receive frame")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Thresholding
    _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY_INV)

    # Region of interest
    roi = binary[binary.shape[0]//2:, :]

    # Find contours
    contours, _ = cv2.findContours(roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            center_x = int(M["m10"] / M["m00"])
        else:
            center_x = prev_center_x

        error = center_x - 320  # Calculate the difference from the image center

        # Simple PID control
        kP = 0.01
        steering = kP * error

        # Set motor speed and direction based on steering
        car.steering = np.clip(steering, -1, 1)
        car.throttle = 0.3

        prev_center_x = center_x

        # Debug output
        print(f"Center: {center_x}, Error: {error}, Steering: {steering}")

        # Visualize contours and center point
        cv2.drawContours(frame, [largest_contour + np.array([0, frame.shape[0]//2])], -1, (0, 255, 0), 2)
        cv2.circle(frame, (center_x, frame.shape[0]//2 + roi.shape[0]//2), 5, (0, 0, 255), -1)

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
