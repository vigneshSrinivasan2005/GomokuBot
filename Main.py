from DefaultAgent import DefaultAgent
from Env import Env
import time
#create 2 new agent
agent1=DefaultAgent(1,.1)
agent2=DefaultAgent(2,.1)
#call env();
#inputs are board size, win condition, agent1, agent2, and batch size
env=Env(3,3,agent1,agent2,20)
#play batches
start= time.time()
for i in range(1):
    #print(i)
    env.playBatch()
#print(time.time()-start)
#save the agents
agent1.save("./Agent1.05.keras")
agent2.save("./Agent2.05.keras")