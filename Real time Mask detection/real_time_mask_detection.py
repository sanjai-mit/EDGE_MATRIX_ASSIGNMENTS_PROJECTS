import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model

mask_model = load_model('mask_detection_model.keras')

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (128, 128))
    image = np.array(image)
    image = image / 255.0
    image_rshape = np.reshape(image, [1, 128, 128, 3])
    return image_rshape

def predict_mask(face_image):
    preprocessed_image = preprocess_image(face_image)
    prediction = mask_model.predict(preprocessed_image)
    mask_prob = prediction[0][1]
    label = "With mask" if mask_prob > 0.5 else "Without mask"
    return label, mask_prob

cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
        model_selection=1, min_detection_confidence=0.5) as face_detection:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform face detection
        results = face_detection.process(rgb_frame)

        if results.detections:
            for detection in results.detections:
                # Extract the bounding box
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = frame.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
                             int(bboxC.width * iw), int(bboxC.height * ih)
                
                # Extract face region
                face = frame[y:y+h, x:x+w]

                # Predict mask
                label, mask_prob = predict_mask(face)

                # Draw the bounding box and label
                color = (0, 255, 0) if label == "With mask" else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, f"{label} ({mask_prob*100:.2f}%)", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display the result
        cv2.imshow('Mask Detection', frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
