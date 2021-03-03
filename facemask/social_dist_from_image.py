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

detector = MTCNN()

image = cv2.cvtColor(cv2.imread(args["image"]), cv2.COLOR_BGR2RGB)

detections = detector.detect_faces(image)

# classification_model = load_classification_model("models/model1_300obs")
# classification_model = load_classification_model("../models/model2_600obs_multimasks")
# classification_model = load_classification_model("../models/model3_600obs_multimasks")
classification_model = load_classification_model("../models/model4_600obs_multimasks")

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
        if x>0 and y>0 and height>0 and width>0:
            
            # for model 1, 2, 3, 4, 5, 6 , 10, 11
            face = img_with_dets[y:y+height, x:x+width]

            #for model 7, 8, 9 only
            # face = img_with_dets[y_min:y_max, x_min:x_max]



            # output_path = os.path.join(output_folder_path, f"tmp_{img_ind}.jpg")
            # cv2.imwrite(output_path, cv2.cvtColor(face, cv2.COLOR_RGB2BGR))
            # ind += 1



            #model 1, 9, 10
            # face = cv2.resize(face, (256, 256))

            #model 2, 3, 4, 5, 6, 7, 8, 11
            face = cv2.resize(face, (128, 128))

            # for model 5, 6 ,7, 8  (black and white) only
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

            face_norm = face / 255.0
            
            # print(face_norm.shape)
            # face_norm_exp = np.expand_dims(face_norm, axis=0)

            # face = img_to_array(face)
            # face = preprocess_input(face)

            face_expanded = np.expand_dims(face_norm, axis=0)

            #for model 5, 6 , 7, 8 (black and white) only
            # face_expanded = np.expand_dims(face_expanded, axis=-1)

            print(face_expanded.shape)
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


# cv2.imwrite("output.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

## distanciation sociale

focal = 1.18 #px
mean_dim_eyes_to_mouth = 8.874  #cm  // mean between women and men
mean_dim_face = 24.65  #cm  // mean between women and men
people_coord = []
social_dist_array = np.zeros((len(detections),len(detections)))
counter = 0
color_wrong = (255,0,0)
pixels_face = detections[0]['box'][3]
ratio_pix_cm= (0.8*mean_dim_face)/pixels_face
print(ratio_pix_cm)
for detection in detections:
    if detection['confidence'] >= min_conf:
        pixels_face_temp = detection['box'][3]
        ratio_pix_cm_temp= (0.8*mean_dim_face)/pixels_face_temp
        # pixels_eyes_to_mouth= 0.5*(detection['keypoints']['mouth_left'][1]+detection['keypoints']['mouth_right'][1])-0.5*(detection['keypoints']['left_eye'][1]+detection['keypoints']['right_eye'][1])
        # ratio_pix_cm = pixels_eyes_to_mouth/mean_dim_eyes_to_mouth
        Z_center_cm = focal*ratio_pix_cm_temp
        X_center_cm = detection['keypoints']['nose'][0]*ratio_pix_cm
        Y_center_cm = detection['keypoints']['nose'][1]*ratio_pix_cm
        people_coord.append([X_center_cm, Y_center_cm, Z_center_cm])
        counter+=1

offset = 2*int(detections[0]['box'][2])
for person1 in range(counter):
    for person2 in range(counter):
        if person2 != person1:
            social_dist_array[person1][person2] = ((people_coord[person1][0]-people_coord[person2][0])**2+(people_coord[person1][1]-people_coord[person2][1])**2+(people_coord[person1][2]-people_coord[person2][2])**2)**0.5
print(people_coord)


for person1 in range(counter):
    for person2 in range(counter):
        if person2 != person1:
            x0_1=(detections[person1]['keypoints']['nose'][0]+detections[person2]['keypoints']['nose'][0])/2
            x1=detections[person1]['keypoints']['nose'][0]
            y1=detections[person1]['keypoints']['nose'][1]
            x2=detections[person2]['keypoints']['nose'][0]
            y2=detections[person2]['keypoints']['nose'][1]        
            if social_dist_array[person1][person2] <= 100 and social_dist_array[person1][person2] != 0:
                cv2.putText(image, f"{round(social_dist_array[person1][person2]/100,2)}m", (int(x0_1-offset/8),int((y1+y2)/2)+1.2*offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_wrong, 1)
                cv2.line(image,(x1,y1+offset),(x2,y2+offset), (200,0,0), 1)
                cv2.line(image,(x1,int(y1+0.90*offset)), (x1,int(y1+1.10*offset)), (200,0,0), 1)
                cv2.line(image,(x2,int(y2+0.90*offset)), (x2,int(y2+1.10*offset)), (200,0,0), 1)
                offset=int(1.4*offset)
                social_dist_array[person2][person1]=9999
print(social_dist_array)



cv2.imwrite("../images/output.jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))



        





