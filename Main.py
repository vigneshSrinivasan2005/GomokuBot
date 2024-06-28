from DefaultMinimaxAgent import DefaultMiniMaxAgent
from DefaultAgent import DefaultAgent
from Env import Env
import tensorflow as tf
import time
#create 2 new agent)
agent1 = DefaultMiniMaxAgent(1, 0.1)
agent2 = DefaultMiniMaxAgent(2,0.1)
#call env();
#inputs are board size, win condition, agent1, agent2, and batch size
env=Env(3,3,agent1,agent2,1,epsilon=0, agent1_is_training = True, agent2_is_training = True)
#play batches
start= time.time()
for i in range(100):
    print(i)
    start2= time.time()
    env.playBatch()
    #print("Error: ",agent1.calculateMeanSquareError())
    print(time.time()-start2)
print(time.time()-start)
#save the agents
#agent1.savePlict("3x3MiniMaxPlict.csv")
agent1.save("DefaultMiniMaxAgent.keras")
