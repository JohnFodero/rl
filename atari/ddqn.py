import argparse
import gym
import numpy as np
from collections import deque
import h5py
import random
from skimage.color import rgb2gray
from skimage.transform import resize
from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten
from keras.optimizers import RMSprop
from PIL import Image

class ExperienceReplay():
    def __init__(self, memory_size=50000, batch_size=32):
        # make memory deque
        self.memory = deque(maxlen=memory_size)
        self.batch_size = batch_size

    def remember(self, state, action, reward, next_state, done):
        # < state, action, reward, next_state >
        # each state consists of the last four maximums of t and t-1 observations
        # add it to memory
        self.memory.append((state, action, reward, next_state, done))
        
    def get_batch(self):
        # get a random batch of memory of length batch_sizr
        return np.array(random.sample(self.memory, self.batch_size))

class AtariAgent():
    def __init__(self, environment_name='Breakout-v0'):
        # load environment
        self.environment_name = environment_name
        self.env = gym.make(self.environment_name)
        self.state_size = 4
        self.state_buffer = None
        self.target_update_interval = 32
        self.gamma = 0.99
        # create ExperienceReplay
        self.memory_size = 50000
        self.batch_size = 32
        self.memory = ExperienceReplay(memory_size=self.memory_size, batch_size=self.batch_size)
        # define parameters
        self.train_episodes = 50000
        self.pre_train_episodes = 5
        self.test_episodes = 10
        self.input_shape = (84, 84, self.state_size)

        self.output_shape = self.env.action_space.n
        self.lr = 0.00025
        # calculate epsilon step
        self.epsilon = 1.0
        self.epsilon_min = 0.1
        self.epsilon_test = 0.05
        self.epsilon_decay_steps = 1000000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.epsilon_decay_steps
        # create main network
        self.model = self._build_model()
        # create target network
        self.target_model = self._build_model()
        self.target_model.set_weights(self.model.get_weights())
        self.save_interval = 1000
        self.save_weights_file = 'ddqn_{}_weights.h5'

        print('Agent Initialized')
        print('Environment: {}     Action Space: {}'.format(self.environment_name, self.output_shape))
        

    def _build_model(self):
        # Recreated DeepMind Model
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=(4,4), input_shape=self.input_shape, activation='relu'))
        model.add(Conv2D(32, (4, 4), strides=(2,2), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.output_shape, activation='linear'))
        model.compile(optimizer=RMSprop(lr=self.lr), loss='mse')
        return model

    def load_env_weights(self):
        weights_file = 'ddqn_{}_weights.h5'.format(self.environment_name)
        print('Using {}'.format(weights_file))
        self.model.load_weights(weights_file)
        self.target_model.set_weights(self.model.get_weights())

    def process_observation(self, observation, previous_observation):
        # take the max of the last observation 
        processed_observation = np.maximum(observation, previous_observation)
        # convert to grayscale
        processed_observation = rgb2gray(processed_observation)
        # resize and multiply by 255 (resize converts to float..)
        processed_observation = resize(processed_observation, (self.input_shape[0], self.input_shape[1])) * 255
        # enforce 'uint8' type and reshape the image
        return np.uint8(np.reshape(processed_observation, (self.input_shape[0], self.input_shape[1], 1)))

    def process_batch(self, observations):
        # /= 255.
        # this saves space when storing >1M observations
        pass

    def act(self, state, force_random=False):
        # get random number and check if less than epsilon
        if force_random:
            action = np.random.randint(0, high=self.output_shape)
        elif np.random.rand() < self.epsilon:
            action = np.random.randint(0, high=self.output_shape)
            #print('chose random ', action)
        else:
            action = np.argmax(self.model.predict(np.expand_dims(state, axis=0)))
            #print('guessed action ', action)
        if self.epsilon > self.epsilon_min and not force_random:
            self.epsilon -= self.epsilon_decay
        return action

    def train(self, show=False):
        # iterate over pre-train episodes to populate memory
        print('--------\npre-training')
        frame = 0
        total_reward = 0
        for episode in range(self.pre_train_episodes):
            print('{} episode, {} frame'.format(episode, frame), end='\r')
            observation = self.env.reset()
            processed_observation = self.process_observation(observation, observation)
            # make a complete state so the first action can be taken
            self.state_buffer = np.moveaxis(np.array([processed_observation[:,:,0] for _ in range(self.state_size)]), 0, -1)
            done = False
            # play a game
            while not done:
                # choose an action
                action = self.act(self.state_buffer, force_random=True)
                previous_observation = observation
                # get new observation from action
                observation, reward, done, _ = self.env.step(action)
                # process observation
                processed_observation = self.process_observation(observation, previous_observation)
                # if there have been state_size frames, remember a new frame
                if frame % self.state_size == 0:
                    next_state_buffer = np.append(self.state_buffer[:,:,1:], processed_observation, axis=2)
                    self.memory.remember(self.state_buffer, action, reward, next_state_buffer, done)
                self.state_buffer = np.append(self.state_buffer[:,:,1:], processed_observation, axis=2)
                total_reward += reward
                frame += 1
                

        print('\nMemory size after pre-training: {}'.format(len(self.memory.memory)))
        print('--------\npre-training complete')

        # train the model with a prepopulated 
        print('\n--------\nStart training')
        frame = 0
        loss = 0
        total_reward = 0
        for episode in range(self.train_episodes):
            observation = self.env.reset()
            processed_observation = self.process_observation(observation, observation)
            # make a complete state to start the game
            self.state_buffer = np.moveaxis(np.array([processed_observation[:,:,0] for _ in range(self.state_size)]), 0, -1)

            done = False
            while not done:
                # choose an action (random or predicted)
                if show:
                    self.env.render()

                action = self.act(self.state_buffer, force_random=False)
                previous_observation = observation
                # get updated frames
                observation, reward, done, _ = self.env.step(action)
                total_reward += reward
                processed_observation = self.process_observation(observation, previous_observation)

                if frame % self.state_size == 0:
                    '''
                    # need to check the order of these images
                    img1 = Image.fromarray(self.state_buffer[:,:,0])
                    img1.show()
                    img1 = Image.fromarray(self.state_buffer[:,:,1])
                    img1.show()
                    img1 = Image.fromarray(self.state_buffer[:,:,2])
                    img1.show()
                    img1 = Image.fromarray(self.state_buffer[:,:,3])
                    img1.show()
                    input()
                    '''


                    #print('{} Training on batch'.format(frame))
                    next_state_buffer = np.append(self.state_buffer[:,:,1:], processed_observation, axis=2)
                    self.memory.remember(self.state_buffer, action, reward, next_state_buffer, done)
                    # train the main model
                    # get batch
                    batch = self.memory.get_batch()
                    # put current and next state frames in their own arrays for simplicity
                    target_batch = np.array([batch[i,3] for i in range(len(batch))])
                    predict_batch = np.array([batch[i,0] for i in range(len(batch))])
                    # preprocess batch
                    
                    '''
                    # visualize batch
                    print(target_batch.shape)
                    for i in range(self.input_shape[2]):
                        temp_img = Image.new('I', (self.input_shape[0] * self.batch_size, self.input_shape[1]))
                        for j in range(self.batch_size):
                            temp_img.paste(Image.fromarray(target_batch[j, :, :, i]), (self.input_shape[0] * j, 0))
                        temp_img.show()
                    input()
                    '''

                    # train on batch

                    targets = self.target_model.predict(target_batch)
                    predicted = self.model.predict(predict_batch)

                    for i, entry in enumerate(batch):
                        temp_state, temp_action, temp_reward, temp_next_state, temp_done = entry
                        if temp_done:
                            predicted[i, temp_action] = temp_reward
                        else:
                            predicted[i, temp_action] = temp_reward + self.gamma * np.amax(targets[i])
                    

                    loss += self.model.train_on_batch(predict_batch, predicted)

                if frame % self.target_update_interval == 0:
                    #print('{} Updating target'.format(frame))
                    # update the weights of the target model
                    self.target_model.set_weights(self.model.get_weights())

                if frame % self.save_interval == 0:
                    print('{} Saving weights'.format(frame))
                    self.model.save_weights(self.save_weights_file.format(self.environment_name))

                self.state_buffer = np.append(self.state_buffer[:,:,1:], processed_observation, axis=2)

                total_reward += reward
                frame += 1
            print('{}:  Loss: {:.5f}   Epsilon: {:.5f} Total Reward: {}'.format(episode, loss, self.epsilon, total_reward))
            loss = 0
            total_reward = 0

        print('Training finished')
            # get action (random or guessed)
            # step the env and get next observation
            # make preprocess observation
            # set previous observation to the current observation
            # add to state deque
            # for every length of state_size
                # get state deque
                # make/get state_deque of t + 1
                # remember action < current_state, action, reward, next_state >
            
            # train model every update_interval on batch using model.train_on_batch
                # get batch
                # preprocess *batch* 
                # train_on_batch using epsilon greedy
            # save the model every save_interval to model_weights_file

        
    def test(self):
        self.load_env_weights()
        self.epsilon = 0.05
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.0
        # reset environment to get first observation
        # do a few no_op steps
        # for test_length
        for episode in range(self.test_episodes):
            #print('{} episode, {} frame'.format(episode, frame), end='\r')
            observation = self.env.reset()
            processed_observation = self.process_observation(observation, observation)
            # make a complete state so the first action can be taken
            self.state_buffer = np.moveaxis(np.array([processed_observation[:,:,0] for _ in range(self.state_size)]), 0, -1)
            done = False
            # play a game
            while not done:
                self.env.render()
                action = self.act(self.state_buffer, force_random=False)
                previous_observation = observation
                # get updated frames
                observation, reward, done, _ = self.env.step(action)
                processed_observation = self.process_observation(observation, previous_observation)
                self.state_buffer = np.append(self.state_buffer[:,:,1:], processed_observation, axis=2)
        
'''
TRAINING
 how to evaluate??
 experience replay (memorize, batch train, ..)
 checkpoints??
 epsilon greedy agent
 skip every n steps
 pre-training steps? fill the memory...

LOG
 save metrics - which ones??
'''
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_weights')
    parser.add_argument('--mode', choices=['train', 'test'], default='test')
    parser.add_argument('--show', action='store_true', default=False)
    args = parser.parse_args()

    # make agent
    agent = AtariAgent()

    if args.load_weights:
        agent.load_env_weights()

    if args.mode == 'train':
        # train agent
        print('Training agent')
        agent.train(args.show)

    elif args.mode == 'test':    
        # test agent
        print('Testing agent')
        agent.test()

if __name__ == '__main__':
    main()
