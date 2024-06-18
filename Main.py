from DefaultAgent import DefaultAgent
from Agent5x5 import Agent5x5
from PlayerInput import PlayerInput
from Env import Env
import tensorflow as tf
from tensorflow import keras
import time
#create 2 new agent
#m1 = keras.models.load_model("Agent1.05.keras")
agent1=PlayerInput(1,.1,3)
agent2=PlayerInput(2,.1,3)
#call env();
#inputs are board size, win condition, agent1, agent2, and batch size
env=Env(3,3,agent1,agent2,20, agent1_is_training = True, agent2_is_training = True)
#play batches
start= time.time()
for i in range(1000):
    print(i)
    env.playBatch()
print(time.time()-start)
#save the agents
agent1.save("PlayerInputAgent.keras")

