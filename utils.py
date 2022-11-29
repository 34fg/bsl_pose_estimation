import cv2
import numpy as np
import time
import mediapipe as mp
import time


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities



def mediapipe_detection (image, model):
    image = cv2.cvtColor (image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False # Image is no longer writeable
    results= model.process (image) # Make prediction
    image.flags.writeable = True # Image is now writeable
    image = cv2.cvtColor (image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_landmarks (image, results):
    mp_drawing.draw_landmarks (image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
    mp_drawing.draw_landmarks (image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks (image, results.left_hand_landmarks, mp_holistic. HAND_CONNECTIONS)
    mp_drawing.draw_landmarks (image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def pose_array(results):

    if results.face_landmarks:
        face= np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten()
    else:
        face= np.zeros(1404)

    if results.right_hand_landmarks:
        right_hand = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand= np.zeros(21*3)

    if results.left_hand_landmarks:
        left_hand = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand= np.zeros(21*3)  

    if results.pose_landmarks:
        pose = np.array([[res.x, res.y, res.z,res.visibility] for res in results.pose_landmarks.landmark]).flatten()
    else:
        pose= np.zeros(132)
    return np.concatenate([pose, face, left_hand, right_hand])

def prob_viz(res, mimics, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
         x= cv2.putText(output_frame, mimics[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
         x
         del x
    return output_frame