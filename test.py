import sys
import numpy as np
import csv
import pandas as pd
from agents.agent import DDPG
from task import Hover
import statistics
import matplotlib.pyplot as plt
import datetime
import jinja2
import os
import string
np.seterr(divide='raise')

translator=str.maketrans('','',string.punctuation)


def render(tpl_path, context):
    path, filename = os.path.split(tpl_path)
    return jinja2.Environment(
        loader=jinja2.FileSystemLoader(path or './')
    ).get_template(filename).render(context)

num_episodes = 1000
task = Hover()
agent = DDPG(task) 
labels = ['episode', 'rewards_total','final_pose','final_reward']
rewards={}
results=pd.DataFrame(columns=labels)
states={}
actions={}

dateTime=str(datetime.datetime.now())
dateTime=dateTime.translate(translator).replace(' ','_')

startTime=datetime.datetime.now()
best_total_reward = 0
best_episode=0
for i_episode in range(1, num_episodes+1):
    total_reward = 0
    state = agent.reset_episode() # start a new episode
    states[i_episode]=[]
    while True:
        action = agent.act(state)
        states[i_episode].append(state)
        next_state, reward, done = task.step(action)
        total_reward += reward
        agent.step(action, reward, next_state, done, i_episode)
        state = next_state
        if done:
            if total_reward > best_total_reward:
                best_total_reward = total_reward
                best_episode=i_episode
            rows={'episode':str(i_episode),'rewards_total':str(total_reward),'final_pose':'|'.join(str(e) for e in list(state[:3])),'final_reward':str(reward)}
            results=results.append(rows,ignore_index=True)
            rewards[i_episode]=total_reward 
            print("\rEpisode = {:4d}, total reward={:7.3f} , best_reward = {:7.3f}, best_episode={:4d})".format(
                i_episode, total_reward, best_total_reward, best_episode), end="")  # [debug]
            break
    sys.stdout.flush()
endTime=datetime.datetime.now()
trainingTime=(endTime-startTime).total_seconds()
results.to_csv('rewards1.csv', index=False)

_=plt.plot(rewards.keys(),rewards.values())
plt.ylabel('total reward')
plt.xlabel('episode')
fig = plt.gcf()
fig.set_size_inches(10, 8)
fig.savefig('graphs/'+dateTime+'.jpg')

meanReward=statistics.mean(rewards.values())
medianReward=statistics.median(rewards.values())
minReward=min(rewards.values())
maxReward=max(rewards.values())
meanLast10=statistics.mean(list(rewards.values())[-10:])
meanLast100=statistics.mean(list(rewards.values())[-100:])

stringlist = []
agent.actor_local.model.summary(print_fn=lambda x: stringlist.append(x))
actor_model = "<br>\n".join(stringlist)

stringlist = []
agent.critic_local.model.summary(print_fn=lambda x: stringlist.append(x))
critic_model = "<br>\n".join(stringlist)

c={'graph':dateTime,'actorModel':actor_model, 'criticModel':critic_model,'medianReward':medianReward,'meanReward':meanReward, 'maxReward':maxReward, 'minReward':minReward, 'meanLast10':meanLast10, 'meanLast100':meanLast100, 'trainingTime':trainingTime}

report=render('templates/report.template',c)

with open('reports/'+dateTime+".html", "w") as fh:
   fh.write(report)   
   
del c['actorModel']
del c['criticModel']
with open('results.csv', 'a') as f:  # Just use 'w' mode in 3.x
    w = csv.DictWriter(f, c.keys())
    w.writerow(c)
