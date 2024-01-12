import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def  load_mod():
    return load_model('models/model_right.keras')

model = load_mod()

def pred(r, s, ball_vector, position, player_left, player_right, score_left, score_right):
    global model
    observations = np.array([ball_vector[0], ball_vector[1], position[0], position[1], player_right])
    action = np.argmax(model.predict(np.array([observations])))
    return action

pred(1,1,[-0.73401,-0.764839],[190.611,274.869],1,1,1,0)