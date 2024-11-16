import cv2
import numpy as np

# Initialize the video capture (0 for webcam, or replace with video file path)
cap = cv2.VideoCapture(0)

# Create the Background Subtractor
fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=True)

# Minimum area threshold to filter small motions
min_area = 800

# Initialize tracker (using centroid tracking)
trackers = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize frame for better performance (optional)
    frame = cv2.resize(frame, (640, 480))

    # Apply Background Subtraction
    fgmask = fgbg.apply(frame)

    # Remove noise using morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections = []

    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue

        # Bounding box for the detected motion
        x, y, w, h = cv2.boundingRect(cnt)
        detections.append((x, y, x + w, y + h))

        # Draw bounding box on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Update and track objects
    for tracker in trackers:
        tracker.update(detections)
        tracker.draw(frame)

    # Display the frame
    cv2.imshow('Motion Detection and Tracking', frame)

    # Quit with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
