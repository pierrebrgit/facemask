import cv2
from mtcnn import MTCNN
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input image")
args = vars(ap.parse_args())

detector = MTCNN()

image = cv2.cvtColor(cv2.imread(args["image"]), cv2.COLOR_BGR2RGB)
detections = detector.detect_faces(image)

img_with_dets = image.copy()
min_conf = 0.9
for det in detections:
    if det['confidence'] >= min_conf:
        x, y, width, height = det['box']
        print("Face found")
        face = img_with_dets[y:y + height, x:x + width]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face, (224, 224))

        # face = img_to_array(face)
        # face = preprocess_input(face)
        # face = np.expand_dims(face, axis=0)

        # label = model.predict
        # return 0, 1, 2
        label = np.random.randint(0, 2)

        if label == 0:
            color = (0, 255, 0)
        elif label == 1:
            color = (255, 0,0)
        else:
            color = (0, 0, 255)
        print("Drawing rect in ", color)

        # dessiner
        bounding_box = det['box']
        cv2.rectangle(image,
              (bounding_box[0], bounding_box[1]),
              (bounding_box[0]+bounding_box[2], bounding_box[1] + bounding_box[3]),
              color,
              2)

cv2.imwrite("output.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
