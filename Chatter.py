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
            
        self.model = Model()
        
        self.model.loadModel("chat_model", "tokenizer.pickle", "label_encoder.pickle")
        
    def chatResponse(self, response: str): 
        print(Fore.GREEN + "ChatBot:" + Style.RESET_ALL , response)
        
    def getUserInput(self):
        print(Fore.LIGHTBLUE_EX + "User: " + Style.RESET_ALL, end="")
        inp = input()
        return inp
    
    def chat(self):
        print(Fore.YELLOW + "Start messaging with the bot (type quit to stop)!" + Style.RESET_ALL)
        
        while True:
            inp = self.getUserInput()
            
            tag = self.model.predict(inp)
            
            for i in self.data['intents']:
                if i['tag'] == tag:
                    self.chatResponse(np.random.choice(i['responses']))
                    
            if inp.lower() == "quit" or inp.lower() == "exit" or tag == "goodbye":
                break