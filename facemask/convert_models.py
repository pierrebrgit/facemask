# convert pb model to h5
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
    tf.keras.models.save_model(model, "ultra_h5.h5", save_format="h5")
    # model.save("ultra_model.h5")


if __name__ == '__main__':
    main()
