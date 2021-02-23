import cv2
from mtcnn import MTCNN
import argparse
import numpy as np
from classification_model import load_classification_model

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

detector = MTCNN()

image = cv2.cvtColor(cv2.imread(args["image"]), cv2.COLOR_BGR2RGB)
detections = detector.detect_faces(image)

classification_model = load_classification_model("models/model1_300obs")

img_with_dets = image.copy()
min_conf = 0.9
ind = 0
for detection in detections:
    if detection['confidence'] >= min_conf:
        x, y, width, height = detection['box']
        print("Face found")

        # corner_top_left = (x, y)
        # corner_bottom_right = (x + height, y + height)
        # height = int(round(height * 1.2, 0))
        new_x = x - int(round(((height - width) / 2), 0))
        face = img_with_dets[y:y + height, new_x:x + height]
        cv2.imwrite(f"tmp_{ind}.jpg", cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
        ind += 1

        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (256, 256))
        face_norm = face / 255.0
        print(face_norm.shape)
        # face_norm_exp = np.expand_dims(face_norm, axis=0)

        # face = img_to_array(face)
        # face = preprocess_input(face)
        face_expanded = np.expand_dims(face_norm, axis=0)

        y_pred = classification_model.predict(face_expanded)
        print("Prediction :", y_pred)
        y_pred_round = np.round(y_pred)
        print("Prediction(round) :", y_pred_round)

        if y_pred_round[0][0] == 1:
            color = (255, 0, 0)
        elif y_pred_round[0][1] == 1:
            color = (0, 255, 0)
        else:
            color = (0, 0, 255)

        print("Drawing rect in ", color)

        # dessiner
        bounding_box = detection['box']
        cv2.rectangle(image,
            (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              color,
              2)
        # cv2.putText(img, str(rectangle[4]),
        #                     (int(rectangle[0]), int(rectangle[1])),
        #                     cv2.FONT_HERSHEY_SIMPLEX,
        #                     0.5, (0, 255, 0))

cv2.imwrite("output.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
