# copyright (c) 2022 Aaron Hance
# slither_data.js is freely distributable under the MIT license.
# http://www.opensource.org/licenses/mit-license.php
    

import numpy as np
import json
import time
from PIL import Image
import random
import os
import gc
import cv2
import pickle

from sqlalchemy import true

class Env: 
    width, height = 160, 160
    x_start = 0
    y_start = 0
    plot = None

    def _convert_pos_to_state_pos(self, x, y):
        x = x - self.x_start
        y = y - self.y_start

        x = int( (x / 2700) * (self.width-1) )
        y = int( (y / 2700) * (self.height-1) )

        return x, y

    def get_state(self, player, snakes, food):
        #get snakes json
        #snakes, food, player = self.driver.execute_script("return [window.snakes, window.foods, window.snake]")
        #set active element to body and click
        t1 = time.perf_counter()

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

     
#get all files in data folder starting with 'slither_data'
def get_files():
    files = []
    for file in os.listdir('data'):
        if file.startswith('slither_data'):
            files.append(file)
    return files

#load file
def load_file(file):
    with open('data/' + file, 'r') as f:
        data = json.load(f)

    for d in data:
        d["screen"] = json.loads(d["screen"])
        d["keys"] = json.loads(d["keys"]) 

    return data

env = Env()
samples = []
files = get_files()

for file in files:
    print(file)
    data = load_file(file)
    if len(data) > 400:
        data = data[:-200]
        samples += data
    gc.collect()

for sample in samples:
    # if int(sample['score']) < 100:
    #     continue

    #convert "keys" (left, up, right) array of string to array of int (0 or 1)
    actions = [0] * 4

    if(sample['keys']['left'] == 'true'):
        actions[1] = 1
    elif(sample['keys']['up'] == 'true'):
        actions[3] = 1
    elif(sample['keys']['right'] == 'true'):
        actions[2] = 1
    else:
        actions[0] = 1
    sample['keys'] = np.array(actions)

    #create state from screen
    snakes = sample['screen'][0]
    food = sample['screen'][1]
    player = sample['screen'][2]

    state = env.get_state(player ,snakes, food)
    sample['screen'] = np.array(state)


print(len(samples))

#save samples pickle
with open('samples.pickle', 'wb') as f:
    pickle.dump(samples, f)