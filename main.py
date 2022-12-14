from Chatter import Chatter
from Model import Model

model = Model(2000)

model.initModel()
model.train()
model.save("chat_model", "tokenizer.pickle", "label_encoder.pickle")

chatter = Chatter()
chatter.chat()
