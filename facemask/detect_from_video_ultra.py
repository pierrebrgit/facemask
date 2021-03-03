# ultra light

from classification_model import load_classification_model
import numpy as np
import cv2
import sys
import time
import argparse
import tensorflow as tf

parser = argparse.ArgumentParser(description='convert model')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--img_path', default='imgs/test_input.jpg', type=str,
                    help='Image path for inference')
args = parser.parse_args()


def main():
    if args.net_type == 'slim':
        model_path = "export_models/slim/"
    elif args.net_type == 'RFB':
        model_path = "export_models/RFB/"
    else:
        print("The net type is wrong!")
        sys.exit(1)

    model = tf.keras.models.load_model(model_path)

    # face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classification_model = load_classification_model("../models/model1_300")

    video_capture = cv2.VideoCapture(0)


    while True:

        ret, frame = video_capture.read()

        img = frame.copy()
        h, w, _ = img.shape
        img_resize = cv2.resize(img, (320, 240))
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2RGB)
        img_resize = img_resize - 127.0
        img_resize = img_resize / 128.0

        results = model.predict(np.expand_dims(img_resize, axis=0))

        # print("resized img:")
        # print(img_resize)

        for result in results:
            start_x = max(int(result[2] * w), 0)
            start_y = max(int(result[3] * h), 0)
            end_x = int(result[4] * w)
            end_y = int(result[5] * h)

            face = img[start_y:end_y, start_x:end_x, :]

            # print("Debugging face")
            # print(type(face))
            # print(face)

            # print("Box:")
            # print(start_x, end_x, start_y, end_y)

            # print("img_resize.shape:", img_resize.shape)

            # if face == []:
            #     print("Face is []")

            # face = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
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

            cv2.rectangle(img, (start_x, start_y), (end_x, end_y), color, 2)

            cv2.putText(img, str(round(max(y_pred[0]), 2)) + title, (start_x, start_y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # cv2.rectangle(frame,
            #     (x_min, y_min),
            #       (x_max, y_max),
            #       (0,0,0),
            #       1)


        cv2.imshow('Video', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
