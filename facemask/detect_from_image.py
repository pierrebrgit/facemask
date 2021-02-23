import cv2
from cv2 import rectangle
from mtcnn import MTCNN
import argparse
import numpy as np
from classification_model import load_classification_model
import time

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

start = time.time()
detector = MTCNN()
end = time.time()
print("(time) MTCNN():", end - start)

image = cv2.cvtColor(cv2.imread(args["image"]), cv2.COLOR_BGR2RGB)
start = time.time()
detections = detector.detect_faces(image)
end = time.time()
print("(time) detect_faces(image):", end - start)

start = time.time()
classification_model = load_classification_model("models/model1_300obs")
end = time.time()
print("(time) load_classification_model():", end - start)

img_with_dets = image.copy()
min_conf = 0.9
ind = 0
for detection in detections:
    start = time.time()
    if detection['confidence'] >= min_conf:
        x, y, width, height = detection['box']
        # print("Face found")

        diff = 20
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

        face = img_with_dets[y_min:y_max, x_min:x_max]

        cv2.imwrite(f"tmp_{ind}.jpg", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        ind += 1

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (256, 256))
        face_norm = face / 255.0
        # print(face_norm.shape)
        # face_norm_exp = np.expand_dims(face_norm, axis=0)

        # face = img_to_array(face)
        # face = preprocess_input(face)
        face_expanded = np.expand_dims(face_norm, axis=0)

        start = time.time()
        y_pred = classification_model.predict(face_expanded)
        end = time.time()
        print("(time) predict():", end - start)
        # print("Prediction :", y_pred)
        y_pred_round = np.round(y_pred)
        # print("Prediction(round) :", y_pred_round)
        # print("AP :", round(max(y_pred[0]), 2))

        if y_pred_round[0][0] == 1:
                color = (255, 0, 0)
                title = " no mask"
        elif y_pred_round[0][1] == 1:
            color = (0, 255, 0)
            title = " mask"
        else:
            color = (255, 165, 0)
            title = " incorrect"

        # print("Drawing rect in ", color)

        # dessiner
        bounding_box = detection['box']
        cv2.rectangle(image,
            (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              color,
              1)
        cv2.putText(image, str(round(max(y_pred[0]), 2)) + title, (bounding_box[0], bounding_box[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.rectangle(image,
            (x_min, y_min),
              (x_max, y_max),
              (0,0,0),
              1)
    end = time.time()
    print("(time) Processing time per detection(image):", end - start)

cv2.imwrite("output.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
