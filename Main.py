from Agent3x3UCB import Agent3x3UCB
from DefaultAgent import DefaultAgent
from Agent5x5 import Agent5x5
from PlayerInput import PlayerInput
from Player import Player
from Env import Env
import tensorflow as tf
from tensorflow import keras
import time
#create 2 new agent
m1 = keras.models.load_model("5x5_6hidden_tanhrelu_10kbatches_20pBatch.keras")
agent1=Agent5x5(1, .1, m1)
agent2=Player(2)
#call env();
#inputs are board size, win condition, agent1, agent2, and batch size
env=Env(5,4,agent1,agent2, 1, agent1_is_training = False, agent2_is_training = False)
#play batches
start= time.time()
for i in range(1):
    print(i)
    start2= time.time()
    env.playBatch()
    print(time.time()-start2)
print(time.time()-start)
#save the agents
agent1.save("5x5_6hidden_tanhrelu_10kbatches_20pBatch.keras")
