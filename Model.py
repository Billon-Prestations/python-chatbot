import json 
import numpy as np 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
import pickle

class Model:
    vocab_size = 1000
    embedding_dim = 16
    max_len = 20
    oov_token = "<OOV>"
    epochs = 6000
    num_classes = 0
    
    training_sentences = []
    training_labels = []
    labels = []
    responses = []
    
    lbl_encoder = LabelEncoder()
    
    def __init__(self, epochs: int = 6000):
        self.epochs = epochs
        
    def loadModel(self, modelPath, tokenizerPath, labelEncodePath):
        self.model = keras.models.load_model(modelPath)
        
        with open(tokenizerPath, 'rb') as handle:
            self.tokenizer = pickle.load(handle)

        # load label encoder object
        with open(labelEncodePath, 'rb') as enc:
            self.lbl_encoder = pickle.load(enc)
            
    def predict(self, input: str):
        result = self.model.predict(
            keras.preprocessing.sequence.pad_sequences(
                self.tokenizer.texts_to_sequences([input]),
                truncating='post', maxlen=self.max_len
            )
        )
        
        tag = self.lbl_encoder.inverse_transform([np.argmax(result)])
        
        return tag

    def initModel(self):
        self.model = Sequential()
        self.model.add(Embedding(self.vocab_size, self.embedding_dim, input_length=self.max_len))
        self.model.add(GlobalAveragePooling1D())
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(16, activation='relu'))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        
    def loadDataForTrainig(self):
        with open('intents.json') as file:
            data = json.load(file)

        for intent in data['intents']:
            for pattern in intent['patterns']:
                self.training_sentences.append(pattern)
                self.training_labels.append(intent['tag'])
            self.responses.append(intent['responses'])
            
            if intent['tag'] not in self.labels:
                self.labels.append(intent['tag'])
                
        self.num_classes = len(self.labels)
        
        self.lbl_encoder.fit(self.training_labels)
        self.training_labels = self.lbl_encoder.transform(self.training_labels)
        
        self.tokenizer = Tokenizer(num_words=self.vocab_size, oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(self.training_sentences)

        self.sequences = self.tokenizer.texts_to_sequences(self.training_sentences)
        self.padded_sequences = pad_sequences(self.sequences, truncating='post', maxlen=self.max_len)

    def train(self):
        self.loadDataForTrainig()
        
        self.initModel()
                
        self.model.compile(loss='sparse_categorical_crossentropy', 
                    optimizer='adam', metrics=['accuracy'])

        self.model.summary()

        self.history = self.model.fit(self.padded_sequences, np.array(self.training_labels), epochs=self.epochs)

    def save(self, modelPath, tokenizerPath, labelEncodePath):
        # to save the trained model
        self.model.save(modelPath)

        # to save the fitted tokenizer
        with open(tokenizerPath, 'wb') as handle:
            pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # to save the fitted label encoder
        with open(labelEncodePath, 'wb') as ecn_file:
            pickle.dump(self.lbl_encoder, ecn_file, protocol=pickle.HIGHEST_PROTOCOL)