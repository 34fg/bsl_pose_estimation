from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
import os

data_path = os.path.join('Np_Data')
# mimics that we try to detect
mimics = np.array(['hello&bye','u_need_help','I_need_help','how_are_you','good','morning','afternoon','night','name'])
# 15 videos for each mimic
no_sequences = 15
# Videos are going to be 30 frames in Length
sequence_length = 30
start_folder=0
label_map = {label:num for num, label in enumerate(mimics)}
checkpoint_path = "training_1/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
cp_callback = ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

def create_model():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
    model.add(LSTM(128, return_sequences=True, activation='relu'))
    model.add(LSTM(64, return_sequences=False, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(mimics.shape[0], activation='softmax'))
    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return model

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


model = create_model()


model.fit(X_train, y_train, epochs=300, callbacks=[cp_callback])
model.save('my_model.h5')


#model.summary()
