import gym
import os
import minerl
import logging
import torch
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from importlib import reload
from collections import deque, namedtuple


from matplotlib.pyplot import imshow
import matplotlib.pyplot as pyplot
import numpy as np
from PIL import Image

#%matplotlib inline

import ddpg_agent_sequential
reload(ddpg_agent_sequential)
from ddpg_agent_sequential import Agent_TS


#logging.basicConfig(level=logging.DEBUG)
#pil_img = transforms.ToPILImage()(pov)
#imshow(pil_img)


#writer = SummaryWriter()

sequence_len = 32
sample_len = 1
weight_file_timestamp = os.stat("checkpoint_actor.pth")[8]
agent = Agent_TS(agent_mh_size = 3, agent_inventory_size = 18, \
        world_state_size = [3, 32, 64, 64], action_size=14, \
        random_seed=0, seq_len = sequence_len, actor_chkpt_file="checkpoint_actor.pth")

#action_counts = np.zeros((action_s-1, 10), dtype=int)
action_names = ("attack", "back", "craft", "equip",
                     "forward", "jump", "left", "nearbyCraft", "nearbySmelt", 
                     "place", "right", "sneak", "sprint")
camera_list = deque(maxlen=10000)
camera_action_names = ("pitch", "yaw")

agent_state_list_names = ['damage', 'maxDamage', 'type', 'coal', 'cobblestone', 'crafting_table', 
                    'dirt', 'furnace','iron_axe', 'iron_ingot', 'iron_ore', 'iron_pickaxe', 
                    'log', 'planks', 'stick', 'stone', 'stone_axe', 'stone_pickaxe', 
                    'torch', 'wooden_axe', 'wooden_pickaxe']
agent_state_list = []


pyplot.ion()
#pyplot.show()
#fig_1 = pyplot.figure(num=1)
#fig_1.set_size_inches(12, 6)
#pyplot.ylim(-50,50)
#fig_2 = pyplot.figure(num=2)
#pyplot.ylim(0,10)


env = gym.make("MineRLObtainDiamondDense-v0") 
env.seed(125)
obs_a = env.reset()

action_a = deque(maxlen=sequence_len)
mainhand_a = deque(maxlen=sequence_len)
inventory_a = deque(maxlen=sequence_len)
pov_a = deque(maxlen=sequence_len)


#action_a.append(env.action_space.sample())
for i in range(sequence_len):
    mainhand_a.append(obs_a['equipped_items']['mainhand'])
    inventory_a.append(obs_a['inventory'])
    pov_a.append(obs_a['pov'])

agent_state_s = len(list(obs_a['equipped_items']['mainhand'].values())) + len(list(obs_a['inventory'].values()))
world_state_s = obs_a['pov'].shape
action_s = len(list(env.action_space.sample().values()))



# Iterate through a single epoch gathering sequences of at most 32 steps
done_1=False
active_reward=0
info={}

while True:

    if weight_file_timestamp != os.stat("checkpoint_actor.pth")[8]:
        weight_file_timestamp = os.stat("checkpoint_actor.pth")[8]
        print("reading new weights file")
        agent = Agent_TS(agent_mh_size = 3, agent_inventory_size = 18, \
            world_state_size = [3, 32, 64, 64], action_size=14, \
            random_seed=0, seq_len = sequence_len, actor_chkpt_file="checkpoint_actor.pth")


    if (done_1==False):
        with torch.no_grad():
            action_1, action_1_raw, _ , _  = agent.act(mainhand_a, inventory_a, pov_a)
            #print(action_1_raw)

            #zeros = np.zeros_like(action_1_raw.cpu())
            #a = action_1_raw[0].to(dtype=torch.uint8)
            #if not (a.any()):
            #    action_1 = env.action_space.sample()
        obs_1, reward_1, done_1, info = env.step(action_1)
 
        #print(action_1_raw)
        if (reward_1 >0):
            active_reward = active_reward+1
            print("REWARD !!!!!!!!!!!!!!!!!!!!!! {}".format(active_reward))
        if (active_reward > 0):
            print("REWARD !!!!! {}".format(active_reward))


        #writer.add_scalars('Camera', {'picth':action_1_raw[-1,-1,2], 'yaw':action_1_raw[-1,-1,3]}, global_step=eps_i)


        #writer.add_scalars('actions', {"attack":action_1_raw[-1,-1,0], "back":action_1_raw[-1,-1,1], \
        #    "craft":action_1_raw[-1,-1,4], "equip":action_1_raw[-1,-1,5], "forward":action_1_raw[-1,-1,6], \
        #    "jump":action_1_raw[-1,-1,7], "left":action_1_raw[-1,-1,8], "nearbyCraft":action_1_raw[-1,-1,9], \
        #    "nearbySmelt":action_1_raw[-1,-1,10], "place":action_1_raw[-1,-1,11], "right":action_1_raw[-1,-1,12], \
        #    "sneak":action_1_raw[-1,-1,13], "sprint":action_1_raw[-1,-1,14]}, global_step=eps_i)
    else:
        print("RESET ----------------")
        active_reward = 0
        env.seed(125)
        obs_1 = env.reset()
        done_1=False

    obs_a = obs_1
    mainhand_a.append(obs_a['equipped_items']['mainhand'])
    inventory_a.append(obs_a['inventory'])
    pov_a.append((obs_a['pov']))

    env.render()

print("DONE")

