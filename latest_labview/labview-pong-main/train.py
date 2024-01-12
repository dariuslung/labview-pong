import gym
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization,Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model


def random_action():
    action_one = np.random.choice([0, 1])
    action_two = np.random.choice([0, 1])
    return(np.array([action_one, action_two]))

def ins(r, s, ball_vector, position, player_x, player_y):
    return(np.array([r, s, ball_vector[0], ball_vector[1], position[0], position[1], player_x, player_y]))

def  load_mod():
    return load_model('models/model.keras')

def save_mod(model):
    model.save('models/model.keras')
    return

# Hyperparameters
learning_rate = 0.001
discount_factor = 0.9
epsilon = 0.2
decay = 0.95

num_actions = 2  
num_features = 5 

# Neural Network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(num_features,)),
    BatchNormalization(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dense(num_actions, activation='linear')
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

# Replay Buffer
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add_experience(self, experience):
        if len(self.buffer) >= self.buffer_size:
            self.buffer.pop(0)
        experience = np.array(experience, dtype=object)
        self.buffer.append(experience)

    def sample_batch(self, batch_size):
        rnd_ind = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        #print(rnd_ind)
        batch = np.array(self.buffer)[rnd_ind]
        return batch
    
    def save_buffer(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.buffer, file)

    def load_buffer(self, filename):
        with open(filename, 'rb') as file:
            self.buffer = pickle.load(file)
    

# Initialize replay buffer
replay_buffer = ReplayBuffer(buffer_size=10000)

class Environment:
    def __init__(self):
        self.current_state = np.zeros(num_features)  # Replace with the actual initialization
        self.done = False

    def reset(self):
        # Reset the environment and return the initial state
        self.current_state = np.zeros(num_features)  # Replace with the actual reset logic
        self.done = False
        return self.current_state

    def step(self, action):
        # Take action in the environment and return the next state, reward, and done
        # Replace with the actual step logic in your environment
        reward = 0  # Replace with the actual reward calculation
        self.current_state = np.zeros(num_features)  # Replace with the actual next state
        self.done = False  # Replace with the actual done flag
        return self.current_state, reward, self.done, {}


#start_model()
#train_model(1,1,[1,1],[1,1],1,1)


counter = 0
curr_obs = np.array([0,0,0,0,0])
curr_action = 0
curr_done = 0
last_sf = 0
last_sr = 0

def reward_cal(score_left, score_right):
    global last_sf, last_sr
    if score_right > last_sr:
        return 1
    elif score_left > last_sf:
        return -2
    else:
        return 0

def train_model(r, s, ball_vector, position, player_left, player_right, score_left, score_right):
    global counter, curr_obs, curr_action, curr_done, last_sf,last_sr, model, epsilon
    counter += 1

    if counter >300:
        model = load_mod()
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')


    observations = np.array([ball_vector[0], ball_vector[1], position[0], position[1], player_right])

    if np.random.rand() < epsilon:
        action = np.random.choice(num_actions)
    else:
        action = np.argmax(model.predict(np.array([observations])))



    reward = reward_cal(score_left, score_right)
    last_sf = score_left
    last_sr = score_right

    if score_left == 10 or score_left == 10:
        done = 1
    else:
        done = 0

    #curr_ball_vector, curr_ball_vector, curr_position_z, curr_position_o, curr_player_x = zip(*curr_obs)

    if counter == 1:
        curr_obs = observations
    
    #replay_buffer =ReplayBuffer(buffer_size=10000)
    #replay_buffer.load_buffer('replay_buffer.pkl')

    replay_buffer.add_experience((curr_obs, curr_action, reward, observations, done))
    replay_buffer.save_buffer('replay_buffer.pkl')
    #print(curr_obs)
    
    if(counter% 300 == 0):
        epsilon *= decay
        if counter == 300:
            save_mod(model)

        model = load_mod()

        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

        #print(np.shape(replay_buffer))
        mini_batch = replay_buffer.sample_batch(200)

        states, actions, rewards, next_states, dones = zip(*mini_batch)

        #print(rewards)

        target_q_values = []
        for i in range(200):
            target = rewards[i]
            if not dones[i]:
                target += discount_factor * np.max(model.predict(np.array([next_states[i]])))
            target_q_values.append(target)

        # Convert target Q-values to NumPy array
        target_q_values = np.array(target_q_values)

        # One-hot encode the actions for training
        actions_one_hot = tf.one_hot(actions, num_actions)

        # Calculate the predicted Q-values for the selected actions
        predicted_q_values = tf.reduce_sum(model.predict(np.array(states)) * actions_one_hot, axis=1)

        # Define the loss function (mean squared error between target and predicted Q-values)
        loss = tf.keras.losses.mean_squared_error(target_q_values, predicted_q_values)

        # Optimize the model
        model.train_on_batch(np.array(states), actions_one_hot * target_q_values[:, None])
        save_mod(model)
    
    #print(action)
    
    curr_obs = observations
    curr_action = action

    return action

#train_model(1,1,[0.716787,0.697292],[602.414,76.8336],1,1,1,0)
#train_model(1,1,[-0.73401,-0.764839],[190.611,274.869],1,1,1,0)
#train_model(1,1,[0.716787,0.697292],[602.414,76.8336],1,1,1,0)
#train_model(1,1,[-0.73401,-0.764839],[190.611,274.869],1,1,1,0)
#train_model(1,1,[0.716787,0.697292],[602.414,76.8336],1,1,1,0)
#train_model(1,1,[-0.73401,-0.764839],[190.611,274.869],1,1,1,0)
#train_model(1,1,[0.716787,0.697292],[602.414,76.8336],1,1,1,0)
#train_model(1,1,[-0.73401,-0.764839],[190.611,274.869],1,1,1,0)
#train_model(1,1,[0.716787,0.697292],[602.414,76.8336],1,1,1,0)
#train_model(1,1,[0.716787,0.697292],[602.414,76.8336],1,1,1,0)
#train_model(1,1,[-0.73401,-0.764839],[190.611,274.869],1,1,1,0)

#replay_buf =ReplayBuffer(buffer_size=10000)
#replay_buf.load_buffer('replay_buffer.pkl')
#print(replay_buf.buffer)
#print(replay_buffer.buffer)