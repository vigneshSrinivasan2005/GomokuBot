from Agent import Agent
from Env import Env
#create 2 new agent
agent1=Agent(1,.05)
agent2=(2,.05)
#call env();
env=Env(3,3,agent1,agent2,20)
#play batches
for i in range(1):
    env.playBatch()
#save the agents
agent1.saveModel("Agent1.05")
agent2.saveModel("Agent2.05")