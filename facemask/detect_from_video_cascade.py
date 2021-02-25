
from classification_model import load_classification_model
import numpy as np
import cv2
import sys
import time

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
classification_model = load_classification_model("../models/model1_300")

video_capture = cv2.VideoCapture(0)

while True:

    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = face_cascade.detectMultiScale(gray, 1.3, 5)

    for detection in detections:

        x, y, width, height = detection

        face = frame[y:y+height, x:x+width]
        face = cv2.resize(face, (256, 256))

        face_norm = face / 255.0
        face_expanded = np.expand_dims(face_norm, axis=0)

        y_pred = classification_model.predict(face_expanded)
        y_pred_round = np.round(y_pred)

        if y_pred_round[0][0] == 1:
            color = (0, 0, 255)
            title = " no mask"
        elif y_pred_round[0][1] == 1:
            color = (0, 255, 0)
            title = " mask"
        else:
            color = (0, 165, 255)
            title = " incorrect"

        cv2.rectangle(frame,
            (detection[0], detection[1]),
            (detection[0]+detection[2], detection[1] + detection[3]),
            color, 1)

        cv2.putText(frame, str(round(max(y_pred[0]), 2)) + title, (detection[0], detection[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
