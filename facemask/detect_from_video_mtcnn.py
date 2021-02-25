
import cv2
import sys
from mtcnn import MTCNN
from classification_model import load_classification_model
import numpy as np
import time

start = time.time()
detector = MTCNN()
end = time.time()
print("(time) MTCNN():", end - start)

classification_model = load_classification_model("models/model1_300obs")

video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = detector.detect_faces(image_rgb)

    min_conf = 0.9

    for detection in detections:
        if detection['confidence'] >= min_conf:
            x, y, width, height = detection['box']

            diff = 0
            # squaring image
            if height > width:
                delta = int(round((height - width) / 2))
                y_min = y - diff
                y_max = y + height + diff
                x_min = x - delta - diff
                x_max = x + width + delta + diff
            elif width > height:
                delta = int(round((width - height) / 2))
                y_min = y - delta - diff
                y_max = y + height + delta + diff
                x_min = x - diff
                x_max = x + width + diff

            face = image_rgb[y_min:y_max, x_min:x_max]

            face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
            face = cv2.resize(face, (256, 256))
            face_norm = face / 255.0

            face_expanded = np.expand_dims(face_norm, axis=0)

            y_pred = classification_model.predict(face_expanded)
            y_pred_round = np.round(y_pred)

            if y_pred_round[0][0] == 1:
                color = (255, 0, 0)
                title = " no mask"
            elif y_pred_round[0][1] == 1:
                color = (0, 255, 0)
                title = " mask"
            else:
                color = (255, 165, 0)
                title = " incorrect"

            # Draw a rectangle around the faces
            bounding_box = detection['box']

            cv2.rectangle(frame,
                (bounding_box[0], bounding_box[1]),
                (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
                color, 1)

            cv2.putText(frame, str(round(max(y_pred[0]), 2)) + title, (bounding_box[0], bounding_box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            cv2.rectangle(frame,
                (x_min, y_min),
                  (x_max, y_max),
                  (0,0,0),
                  1)

            # Display the resulting frame
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
