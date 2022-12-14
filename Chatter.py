import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

from Model import Model

class Chatter:
    def __init__(self):
        with open("intents.json") as file:
            self.data = json.load(file)
    