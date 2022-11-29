import cv2
import numpy as np
import os
import mediapipe as mp
from utils import mediapipe_detection
from utils import draw_landmarks
from utils import pose_array
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

    # Path for exported data, numpy arrays
data_path = os.path.join('Np_Data')
# mimics that we try to detect
mimics = np.array(['hello&bye','u_need_help','I_need_help','how_are_you','good','morning','afternoon','night','name'])
# 15 videos for each mimic
no_sequences = 15
# Videos are going to be 30 frames in Length
sequence_length = 30
start_folder=0
for mimic in mimics: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(data_path, mimic, str(sequence)))
        except:
            pass

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    # NEW LOOP
    # Loop through mimics
    for mimic in mimics:
        # Loop through sequences aka videos
        for sequence in range(start_folder, start_folder+no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)
                draw_landmarks(image, results)
                
                # NEW Apply wait logic
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(mimic, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(500)
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(mimic, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                
                # NEW Export keypoints
                keypoints = pose_array(results)
                npy_path = os.path.join(data_path, mimic, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
    cap.release()
    cv2.destroyAllWindows()

    label_map = {label:num for num, label in enumerate(mimics)}