from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from utils import mediapipe_detection
from utils import draw_landmarks
from utils import pose_array
from utils import prob_viz
from keras.callbacks import ModelCheckpoint
import mediapipe as mp
import numpy as np
import cv2
import os


data_path = os.path.join('Np_Data')

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# mimics that we try to detect
mimics = np.array(['hello&bye','u_need_help','I_need_help','how_are_you','good','morning','afternoon','night','name'])
# 15 videos for each mimic
no_sequences = 15
# Videos are going to be 30 frames in Length
sequence_length = 30
start_folder=0
label_map = {label:num for num, label in enumerate(mimics)}

sequences, labels = [], []
for mimic in mimics:
    for sequence in np.array(os.listdir(os.path.join(data_path, mimic))).astype(int):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(data_path, mimic, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[mimic])
X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

def model_x():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(mimics.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

new_model=model_x()

new_model.load_weights('my_model.h5')
new_model.summary()


sequence = []
sentence = []
predictions = []
threshold = 0.5
colors = [(245,117,16), (117,245,16), (16,117,245)]

cap = cv2.VideoCapture(0)
# Set mediapipe model 
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():

        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mediapipe_detection(frame, holistic)
        print(results)
        
        # Draw landmarks
        draw_landmarks(image, results)
        
        # 2. Prediction logic
        keypoints = pose_array(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]
        
        if len(sequence) == 30:
            res = new_model.predict(np.expand_dims(sequence, axis=0))[0]
            print(mimics[np.argmax(res)])
            predictions.append(np.argmax(res))
            
            
        #3. Viz logic
            if np.unique(predictions[-10:])[0]==np.argmax(res): 
                if res[np.argmax(res)] > threshold: 
                    
                    if len(sentence) > 0: 
                        if mimics[np.argmax(res)] != sentence[-1]:
                            sentence.append(mimics[np.argmax(res)])
                    else:
                        sentence.append(mimics[np.argmax(res)])

            if len(sentence) > 5: 
                sentence = sentence[-5:]

            # Viz probabilities
            image = prob_viz(res, mimics, image, colors)
            
        cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3,30), 
                       cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 1, cv2.LINE_AA)
                       
        
        # Show to screen
        cv2.imshow('OpenCV Feed', image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()