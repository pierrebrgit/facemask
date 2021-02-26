import cv2
from cv2 import rectangle
from mtcnn import MTCNN
import argparse
import numpy as np
from classification_model import load_classification_model
import time
import glob
import os
import csv


ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True, help="path to folder")
args = vars(ap.parse_args())

# Loading MTCNN
detector = MTCNN()

# Loading custom classification model

# classification_model = load_classification_model("models/model1_300obs")
# classification_model_name='model1_300obs'

# classification_model = load_classification_model("../models/model2_600obs_multimasks")
# classification_model_name='model2_600obs_multimasks'

# classification_model = load_classification_model("../models/model3_600obs_multimasks")
# classification_model_name='model3_600obs_multimasks'

classification_model = load_classification_model("../models/model4_600obs_multimasks")
classification_model_name='model4_600obs_multimasks'

# classification_model = load_classification_model("../models/model5_600obs_multimasks_BandW")
# classification_model_name='model5_600obs_multimasks_BandW'

# classification_model = load_classification_model("../models/model6_600obs_multicolormasks_BandW")
# classification_model_name='model6_600obs_multicolormasks_BandW'

# classification_model = load_classification_model("../models/model7_1500obs_multicolormasks_BandW_largerpics")
# classification_model_name='model7_1500obs_multicolormasks_BandW_largerpics'

# classification_model = load_classification_model("../models/model8_900obs_monomask_BandW_largerpics")
# classification_model_name='model8_900obs_monomask_BandW_largerpics'

# classification_model = load_classification_model("../models/model9_450obs_256_multicolormasks_largerpics")
# classification_model_name='model9_450obs_256_multicolormasks_largerpics'

# classification_model = load_classification_model("../models/model10_1200obs_256_multicolormasks")
# classification_model_name='model10_1200obs_256_multicolormasks'

# classification_model = load_classification_model("../models/model11_1800_128_multicolormasks")
# classification_model_name='model11_1800_128_multicolormasks'

# Loading images from target folder
folder = args["folder"]
sub_folders = ['0-no_mask','1-mask','2-bad_mask']
print(sub_folders)
total_result=0
nb_faces=0
nb_images=0
for sub_folder in sub_folders:
    print(sub_folder)
    image_list = []
    for filename in glob.glob(f'{folder}/{sub_folder}/*.*'):
        im = cv2.imread(filename)
        image_list.append(im)

    print(f"{len(image_list)} images detected in the folder")
    nb_images+=len(image_list)
    sub_folder_result=0
    for image in image_list:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        detections = detector.detect_faces(image)

        img_with_dets = image.copy()
        min_conf = 0.9
        photo_result=0
        for detection in detections:
            start = time.time()
            face_result=0
            if detection['confidence'] >= min_conf:
                x, y, width, height = detection['box']
                
                diff = 20
                # squaring image for model 7, 8, 9 
                # if height > width:
                #     delta = int(round((height - width) / 2))
                #     y_min = y - diff
                #     y_max = y + height + diff
                #     x_min = x - delta - diff
                #     x_max = x + width + delta + diff
                # elif width > height:
                #     delta = int(round((width - height) / 2))
                #     y_min = y - delta - diff
                #     y_max = y + height + delta + diff
                #     x_min = x - diff
                #     x_max = x + width + diff

                # for model 7, 8, 9
                # if x>0 and y>0 and height>0 and width>0 and x_min>0 and x_max>0 and y_min>0 and y_max>0:

                # for model 1, 2, 3, 4, 5, 6 , 10, 11
                if x>0 and y>0 and height>0 and width>0:
                
                    # for model 1, 2, 3, 4, 5, 6, 10, 11
                    face = img_with_dets[y:y+height, x:x+width]

                    #for model 7, 8, 9 only
                    # face = img_with_dets[y_min:y_max, x_min:x_max]

                    #model 1, 9 , 10
                    # face = cv2.resize(face, (256, 256))

                    #model 2, 3, 4, 5, 6, 7, 8, 11
                    face = cv2.resize(face, (128, 128))

                    # for model 5, 6 ,7, 8  (black and white) only
                    # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

                    face_norm = face / 255.0
                    face_expanded = np.expand_dims(face_norm, axis=0)

                    # for model 5 ,6 ,7 ,8 (black and white) only
                    # face_expanded = np.expand_dims(face_expanded, axis=-1)

                    nb_faces+=1
                    y_pred = classification_model.predict(face_expanded)
                    y_pred_round = np.round(y_pred)
                    if sub_folder== '0-no_mask':
                        y_true = [1,0,0]
                        result=y_true[0]-y_pred_round[0][0]
                    if sub_folder== '1-mask':
                        y_true = [0,1,0]
                        result=y_true[1]-y_pred_round[0][1]
                    if sub_folder== '2-bad_mask':
                        y_true = [0,0,1]
                        result=y_true[2]-y_pred_round[0][2]
                    face_result=result
            photo_result+=face_result
        sub_folder_result+=photo_result
    total_result+=sub_folder_result
accuracy_samples = (nb_faces-total_result)/nb_faces

print(f"number of images: {nb_images}")
print(f"number of faces: {nb_faces}")
print(f"number of mistakes: {total_result}")

print(f"Result: {100*accuracy_samples}%")

with open('../models/benchmark.csv', 'a+', newline='') as file:
    writer = csv.writer(file)
    writer.writerow([classification_model_name,f"{100*accuracy_samples}%"])
