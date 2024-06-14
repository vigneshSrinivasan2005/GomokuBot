from DefaultAgent import DefaultAgent
from Env import Env
#create 2 new agent
agent1=DefaultAgent(1,.05)
agent2=DefaultAgent(2,.05)
#call env();
env=Env(3,3,agent1,agent2,20)
#play batches
for i in range(1):
    env.playBatch()
#save the agents
agent1.save("./Agent1.05.keras")
agent2.save("./Agent2.05.keras")