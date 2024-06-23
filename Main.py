from MinimaxModel import MinimaxModel
from Player import Player
from Env import Env
import tensorflow as tf
from tensorflow import keras
import time
#create 2 new agent)
#model = keras.models.load_model("MiniMax.keras")
agent1 = MinimaxModel(3, "MinMax.csv", 1)
agent2 = Player(2)
#call env();
#inputs are board size, win condition, agent1, agent2, and batch size
env=Env(3,3,agent1,agent2,1,epsilon=0, agent1_is_training = False, agent2_is_training = False)
#play batches
start= time.time()
for i in range(5):
    print(i)
    start2= time.time()
    env.playBatch()
    print(time.time()-start2)
print(time.time()-start)
#save the agents
agent1.save("mORELAYERS.keras")
