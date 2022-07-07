# copyright (c) 2022 Aaron Hance
# slither_data.js is freely distributable under the MIT license.
# http://www.opensource.org/licenses/mit-license.php
    

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

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)