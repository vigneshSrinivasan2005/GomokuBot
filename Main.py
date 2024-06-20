from Agent3x3UCB import Agent3x3UCB
from DefaultAgent import DefaultAgent
from Agent5x5 import Agent5x5
from Agent19x19 import Agent19x19
from Env import Env
import tensorflow as tf
from tensorflow import keras
import time
#create 2 new agent
m1 = keras.models.load_model("3x3_7hidden_tanhrelu_10kbatches_20pBatch_UCB.keras")
agent1=DefaultAgent(1,.1, input_model=m1)
agent2=DefaultAgent(2,.1, input_model=m1)
#call env();
#inputs are board size, win condition, agent1, agent2, and batch size
env=Env(3,3,agent1,agent2,20, epsilon=0, agent1_is_training = False, agent2_is_training = False)
#play batches
start= time.time()
for i in range(1):
    print(i)
    start2= time.time()
    env.playBatch()
    print(time.time()-start2)
print(time.time()-start)
#save the agents
agent1.save("3x3_7hidden_tanhrelu_10kbatches_20pBatch_UCB.keras")
agent2.save("./Agent2.05.keras")

