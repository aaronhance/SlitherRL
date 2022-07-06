# Copyright 2022 Aaron Hance

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
import json
import time
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service as ChromeService
from sqlalchemy import false
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait as WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from PIL import Image
from concurrent.futures import thread
from itertools import count
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
import random
import os
import gc
import cv2


class Env: 
    width, height = 160, 160
    name = 'v3'
    url = 'https://slither.io/'

    x_start = 0
    y_start = 0

    driver = None
    driver_service_args = [
        '--ignore-certificate-errors',
        '--headless',
        '--no-sandbox',
        '--disable-infobars',
        '--disable-session-crashed-bubble',
        '--disable-breakpad',
        '--disable-crash-reporter',
        '--disable-hang-monitor',
        '--disable-prompt-on-repost',
        '--disable-extensions',
        '--disable-features',
    ]

    elm = None
    plot = None

    def __init__(self):
        self.driver_service = ChromeService(
            executable_path=ChromeDriverManager().install(),
            service_args=self.driver_service_args
        )

        self.driver_service.start()
        self.driver = webdriver.Remote(self.driver_service.service_url, webdriver.DesiredCapabilities.CHROME)
        self.driver.set_window_size(640+30, 480 + 160)
        self.driver.set_window_position(0, 0)
        self.driver.get(self.url)

        #wait for page to load
        WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.ID, "playh")))
        self.driver.find_element(By.ID, "nick").send_keys(self.name)

    def __del__(self):
        self.driver_service.stop()
        self.driver.quit()

    def start(self):
        WebDriverWait(self.driver, 10).until(EC.element_to_be_clickable((By.ID, "playh")))
        self.driver.find_element(By.ID, "playh").click()
        time.sleep(0.5)

    def _convert_pos_to_state_pos(self, x, y):
        x = x - self.x_start
        y = y - self.y_start

        x = int( (x / 2700) * (self.width-1) )
        y = int( (y / 2700) * (self.height-1) )

        return x, y

    def get_state(self):
        #get snakes json
        #snakes, food, player = self.driver.execute_script("return [window.snakes, window.foods, window.snake]")
        #set active element to body and click
        t1 = time.perf_counter()

        player, snakes, food = self.driver.execute_script("return [JSON.stringify(window.snake), JSON.stringify(window.snakes), JSON.stringify(window.foods)]")
        player = json.loads(player)
        snakes = json.loads(snakes)
        food = json.loads(food)

        snakes = snakes[:-1]

        #get player position
        player_pos = {
            'x': player['xx'],
            'y': player['yy']
        }

        #translation variables 
        self.x_start = player_pos['x'] - 1350
        self.y_start = player_pos['y'] - 1350

        state = np.zeros((self.width, self.height, 3), dtype=np.float32)

        #add food to state
        for f in food:
            if f != None:
                x, y = self._convert_pos_to_state_pos(f['xx'], f['yy'])

                if x < 0 or y < 0 or x >= self.width or y >= self.height:
                    continue

                state[x, y, 1] = 1

        snake_points = [None] * len(snakes)
        snake_state = np.zeros((self.width, self.height, 3), dtype=np.float32)

        #add snakes to state
        for i, s in enumerate(snakes):
            for p in s['pts']:

                #if dying in p keys and true then skip
                if 'dying' in p and p['dying'] == True:
                    continue

                x, y = self._convert_pos_to_state_pos(p['xx'], p['yy'])

                if x < 0 or y < 0 or x >= self.width or y >= self.height:
                    continue

                #state[x, y, 2] = 0.8

                if snake_points[i] == None:
                    snake_points[i] = []

                snake_points[i].append((x, y))

        #connect the points with lines
        for i in range(len(snake_points)):
            
            if snake_points[i] == None:
                continue

            for j in range(len(snake_points[i])):
                if j == 0:
                    continue

                x1, y1 = snake_points[i][j-1]
                x2, y2 = snake_points[i][j]

                cv2.line(snake_state, (x1, y1), (x2, y2), (0, 0, 0.5), 1)
    
        snake_points = None

        #add player to state
        for p in player['pts']:

            #if dying in p keys and true then skip
            if 'dying' in p and p['dying'] == True:
                continue

            x, y = self._convert_pos_to_state_pos(p['xx'], p['yy'])

            if x < 0 or y < 0 or x >= self.width or y >= self.height:
                continue

            if snake_points == None:
                snake_points = []
                
            snake_points.append((x, y))
            # state[x, y, 0] = 1

        #connect the points with lines
        for j in range(len(snake_points)):
            if j == 0:
                continue

            x1, y1 = snake_points[j-1]
            x2, y2 = snake_points[j]

            cv2.line(snake_state, (x1, y1), (x2, y2), (1, 0, 0), 1)

        #rotate -90
        snake_state = np.rot90(snake_state, 1)

        #flip top and bottom
        snake_state = np.flipud(snake_state)



        state = snake_state + state


        #update plot
        image = state * 255
        image = image.astype(np.uint8)
        #image = Image.fromarray(image)

        #resize x5
        image = cv2.resize(image, (0,0), fx=5, fy=5)

        #rotate clockwise -90 degrees
        image = np.rot90(image, -1)

        #flip horizontally
        image = np.fliplr(image)

        cv2.imshow('image', image)
        cv2.waitKey(1)

        return state

    def get_is_playing(self):
        return self.driver.execute_script("return window.snake.dead_amt") == 0

    def get_score(self):
        try:
            if(self.get_is_playing()):
                try:
                    score = self.elm.text
                except:
                    self.elm = self.driver.find_element(By.XPATH, "//*[contains(text(),'Your length: ')]")
                    #get elm parent
                    self.elm = self.elm.find_element(By.XPATH, '..')
                    #get the last child of elm
                    self.elm = self.elm.find_elements(By.TAG_NAME, 'span')[-1]
                    score = self.elm.text

                #normalize score 0 to 1
                #score = float(score) / 300
                
                return int(score)

            #get score, <b> inside of id="lastscore"
            scoreParent = self.driver.find_element(By.ID, 'lastscore')
            score = scoreParent.find_element(By.TAG_NAME, 'b').text
            #return score
            return int(score)
        except:
            return 0

    def act(self, action):
        #perform action on enviroment keyboard
        actions = ActionChains(self.driver)

        #all keys up
        actions.key_up(Keys.ARROW_LEFT)
        actions.key_up(Keys.ARROW_RIGHT)
        actions.key_up(Keys.ARROW_UP)
        actions.perform()

        #take argmax of action, switch statement
        action = np.argmax(action)

        # #boltzmann exploration
        # pas = np.exp(self.beta * action)/np.sum(np.exp(self.beta * action))
        # action = np.random.choice(range(len(pas)), p=pas)

        #left
        if action == 1:
            actions.key_down(Keys.ARROW_LEFT)
        #right
        elif action == 2:
            actions.key_down(Keys.ARROW_RIGHT)
        #boost
        elif action == 3:
            actions.key_down(Keys.ARROW_UP)
        #left boost
        elif action == 4:
            actions.key_down(Keys.ARROW_LEFT)
            actions.key_down(Keys.ARROW_UP)
        #right boost
        elif action == 5:
            actions.key_down(Keys.ARROW_RIGHT)
            actions.key_down(Keys.ARROW_UP)

        actions.perform()
        
#dqn class
class DQN:
    #learning rate
    lr = 0.001
    #discount rate
    discount_rate = 0.95
    #exploration rate
    epsilon = 0.60
    #epsilon decay
    epsilon_decay = 0.999
    #epsilon min    
    epsilon_min = 0.01
    #batch size
    batch_size = 256
    #memory size
    memory_size = 1500
    #memory
    memory = []
    #model
    model = None
    #target model
    target_model = None
    #update_model_every
    update_model_every = 8
    current_update = 0
    #optimizer
    optimizer = keras.optimizers.Adam(lr=lr)
    #loss function
    loss_func = keras.losses.mean_squared_error
    #state size
    state_size = (160, 160, 3)
    #action size, [none, left, right, boost, left_boost, right_boost]
    action_size = 4


    def __init__(self):
        #build model
        self.model = self.build_model()
        #build target model
        self.target_model = self.build_model()
        #compile model
        self.model.compile(loss="mse", optimizer=keras.optimizers.Adam(lr=self.lr), metrics=['accuracy'])
        #load model if exists
        if os.path.isfile('v3model.h5'):
            self.model.load_weights('v3model.h5')

        self.target_model.set_weights(self.model.get_weights())
        #compile target model
        self.target_model.compile(optimizer=self.optimizer, loss=self.loss_func)
        #set target model weights to model weights
        self.target_model.set_weights(self.model.get_weights())
        #print model summary
        self.model.summary()

    #build model
    def build_model(self):
        #model
        model = keras.Sequential()
        #convolutional layer
        model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=self.state_size))
        #convolutional layer
        model.add(keras.layers.Conv2D(8, (3, 3), activation='relu', input_shape=self.state_size))
        #max pooling layer
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        #flatten layer
        model.add(keras.layers.Flatten())
        #dense layer
        model.add(keras.layers.Dense(48, activation='relu'))
        model.add(keras.layers.Dense(64, activation='selu', kernel_initializer=tf.keras.initializers.lecun_normal(seed=None)))
        model.add(keras.layers.Dense(32, activation='relu'))
        #dense layer
        model.add(keras.layers.Dense(self.action_size, activation='softmax'))
        #return model
        return model

    #get action, returns array of actions[3]
    def get_action(self, state):
        #get action 0, 1, 2
        if np.random.rand() < self.epsilon:
            #return array with 3 random value between 0 and 1
            return np.random.rand(4) 

        actions = self.target_model(np.array([state]))
        return actions[0]

    #remember
    def remember(self, state, action, reward, next_state, done):
        #add to memory
        self.memory.append([state, action, reward, next_state, done])
        #if memory is greater than memory size
        if len(self.memory) > self.memory_size:
            #pop first element
            self.memory.pop(0)

    def set_end_game(self):
        #set last 5 memory entried to -1 for reward
        #if memory is greater than 5
        if len(self.memory) > 3:
            #set last 5 memory entried to -1 for reward
            for i in range(3):
                self.memory[-i-1][2] = -1


    #replay
    def replay(self):
        #if memory is less than batch size
        if len(self.memory) < self.batch_size:
            #return
            return
        #get random batch from memory
        minibatch = random.sample(self.memory, self.batch_size)

        current_states = np.array([i[0] for i in minibatch])
        current_qs_list = self.model.predict(current_states)

        next_states = np.array([i[3] for i in minibatch])
        next_qs_list = self.target_model.predict(next_states)

        X, Y = [], []

        for index, (state, action, reward, next_state, done) in enumerate(minibatch):
            action = np.argmax(action)
            if done:
                #set q value to reward
                new_q = reward
            else:
                #set q value to reward + gamma * max q value
                max_future_q = np.max(next_qs_list[index])
                new_q = reward + self.discount_rate * max_future_q

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            #change state shape from (160, 120) to (160, 120, 1)
            # state = np.expand_dims(state, axis=2)
            # state = [state, minibatch[index][0]]

            #add previous state to another channel of the current state
            X.append(state)
            Y.append(current_qs)
        
        X = np.array(X)
        Y = np.array(Y)
        self.model.fit(x=X, y=Y, batch_size=self.batch_size, verbose=1)

        #set target model weights to model weights
        if self.update_model_every == self.current_update:
            self.target_model.set_weights(self.model.get_weights())
            self.current_update = 0
            #save model weights
            self.model.save_weights('v3model.h5')

        self.current_update += 1

        #decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay        
        
        
    #load model
    def load(self, name):
        #load model
        self.model.load_weights(name)
        #set target model weights to model weights
        self.target_model.set_weights(self.model.get_weights())

    #save model
    def save(self, name):
        #save model
        self.model.save_weights(name)

    #get state
    def get_state(self, frame):
        return frame   

    #play game
    def play(self, frame):
        #get state
        state = (frame)
        #get action
        action = self.get_action(state)
        #return action
        return action
    
    #train
    def train(self, frame):
        #get state
        state =(frame)
        #get action
        action = self.get_action(state)
        #remember
        self.remember(state, action, 0, state, False)
        #replay
        self.replay()
        #return action
        return action

env = Env()
dqn = DQN()

env.start()

previous_score = 0
print_interval = 30
episode_count = 0
count=0

while True:
    t1 = time.perf_counter()

    if env.get_is_playing() is False:

        #set end game
        dqn.set_end_game()
        #replay
        dqn.replay()
        #increment episode count
        episode_count += 1
        #reset previous score
        previous_score = 0
        #reset memory
        #dqn.memory = []
        env.start()
        time.sleep(2)
    else:
        # try:
        count += 1
        #get frame
        frame = env.get_state()
        #play game
        action = dqn.play(frame)
        env.act(action)
        screen_next = env.get_state()
        score = env.get_score()
        reward = score - previous_score
        previous_score = score

        reward = reward / 30
        if reward > 1:
            reward = 1

        if reward < 0:
            reward = reward * 5
        
        if reward < -1:
            reward = -1

        dqn.remember(frame, action, reward, screen_next, False)

        if count % print_interval == 0:
            #call gc
            gc.collect()
            print("episode: {} score: {} reward: {} time: {}".format(episode_count, score, reward, time.perf_counter() - t1))
    # except Exception as e:
        #     print(e)    

