from MinimaxModel import MinimaxModel
from DefaultAgent import DefaultAgent
from DefaultMinimaxAgent import DefaultMiniMaxAgent
from Player import Player
from Env import Env
import tensorflow as tf
from tensorflow import keras
import time
#create 2 new agent)
#model = keras.models.load_model("3x3model.keras")
agent1 = DefaultMiniMaxAgent(1, 0.1)
agent2 = DefaultMiniMaxAgent(2, 0.2)
#call env();
#inputs are board size, win condition, agent1, agent2, and batch size
env=Env(3,3,agent1,agent2,1,epsilon=0, agent1_is_training = True, agent2_is_training = True)
#play batches
start= time.time()
for i in range(10):
    print(i)
    start2= time.time()
    env.playBatch()
    agent1.evaluate()
    print(time.time()-start2)
print(time.time()-start)
#save the agents
agent1.save("3x3model.keras")
