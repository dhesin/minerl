import gym
import minerl
import logging
import torch
import torchvision.transforms as transforms
from importlib import reload
from collections import deque, namedtuple


from matplotlib.pyplot import imshow
import matplotlib.pyplot as pyplot
import numpy as np
from PIL import Image
from ddpg_agent import Agent
#%matplotlib inline

import ddpg_agent
reload(ddpg_agent)
from ddpg_agent import Agent

#logging.basicConfig(level=logging.DEBUG)
#pil_img = transforms.ToPILImage()(pov)
#imshow(pil_img)
 
def entropy(labels):
  """ Computes entropy of label distribution. """

  n_labels = len(labels)

  if n_labels <= 1:
    return 0

  value,counts = np.unique(labels, return_counts=True)
  probs = counts / n_labels
  n_classes = np.count_nonzero(probs)

  if n_classes <= 1:
    return 0

  ent = 0.

  # Compute entropy
  for i in probs:
    ent -= i * np.log(i)

  return ent


def extract_data_from_dict(current_state, action, reward, next_state, done):

    equipments = {"none":1, 'air':2, 'wooden_axe':3, 'wooden_pickaxe':4, 
              'stone_axe':5, 'stone_pickaxe':6, 'iron_axe':7, 'iron_pickaxe':8}


    # current state
    mainhand = current_state['equipped_items']['mainhand']
    inventory = current_state['inventory']
    pov = current_state['pov']


    agent_mh = []
    agent_mh.append(mainhand['damage'])
    agent_mh.append(mainhand['maxDamage'])
    agent_mh.append(mainhand['type'])
    agent_mh_ts = agent_mh

    agent_inventory = []
    agent_inventory.append(inventory['coal'])
    agent_inventory.append(inventory['cobblestone'])
    agent_inventory.append(inventory['crafting_table'])
    agent_inventory.append(inventory['dirt'])
    agent_inventory.append(inventory['furnace'])
    agent_inventory.append(inventory['iron_axe'])
    agent_inventory.append(inventory['iron_ingot'])
    agent_inventory.append(inventory['iron_ore'])
    agent_inventory.append(inventory['iron_pickaxe'])
    agent_inventory.append(inventory['log'])
    agent_inventory.append(inventory['planks'])
    agent_inventory.append(inventory['stick'])
    agent_inventory.append(inventory['stone'])
    agent_inventory.append(inventory['stone_axe'])
    agent_inventory.append(inventory['stone_pickaxe'])
    agent_inventory.append(inventory['torch'])
    agent_inventory.append(inventory['wooden_axe'])
    agent_inventory.append(inventory['wooden_pickaxe'])
    agent_inventory_ts = agent_inventory

    vertical_agent_mh = [np.vstack(item) for item in agent_mh]
    vertical_agent_invent = [np.vstack(item) for item in agent_inventory]
    concat_agent_mh = np.concatenate(vertical_agent_mh, axis=1)
    concat_agent_invent = np.concatenate(vertical_agent_invent, axis=1)
    

    swap_world_state = [np.swapaxes(item,0,2) for item in pov]
    
    #[print(item.shape) for item in swap_world_state]   
    
    vertical_world_state = np.stack(swap_world_state, axis=0)
    #print(vertical_world_state.shape)
    
    #[print(item.shape) for item in vertical_world_state] 

    
    
    
    # next_state
    mainhand = next_state['equipped_items']['mainhand']
    inventory = next_state['inventory']
    pov = next_state['pov']


    agent_mh = []
    agent_mh.append(mainhand['damage'])
    agent_mh.append(mainhand['maxDamage'])
    agent_mh.append(mainhand['type'])

    agent_inventory = []
    agent_inventory.append(inventory['coal'])
    agent_inventory.append(inventory['cobblestone'])
    agent_inventory.append(inventory['crafting_table'])
    agent_inventory.append(inventory['dirt'])
    agent_inventory.append(inventory['furnace'])
    agent_inventory.append(inventory['iron_axe'])
    agent_inventory.append(inventory['iron_ingot'])
    agent_inventory.append(inventory['iron_ore'])
    agent_inventory.append(inventory['iron_pickaxe'])
    agent_inventory.append(inventory['log'])
    agent_inventory.append(inventory['planks'])
    agent_inventory.append(inventory['stick'])
    agent_inventory.append(inventory['stone'])
    agent_inventory.append(inventory['stone_axe'])
    agent_inventory.append(inventory['stone_pickaxe'])
    agent_inventory.append(inventory['torch'])
    agent_inventory.append(inventory['wooden_axe'])
    agent_inventory.append(inventory['wooden_pickaxe'])

    #flat_list = [item for sublist in agent_state for item in sublist]
    vertical_next_agent_mh = [np.vstack(item) for item in agent_mh]
    vertical_next_agent_invent = [np.vstack(item) for item in agent_inventory]
    concat_next_agent_mh = np.concatenate(vertical_next_agent_mh, axis=1)
    concat_next_agent_invent = np.concatenate(vertical_next_agent_invent, axis=1)

    swap_next_world_state = [np.swapaxes(item,0,2) for item in pov]
    
    #[print(item.shape) for item in swap_world_state]   
    
    vertical_next_world_state = np.stack(swap_next_world_state, axis=0)


    
    # get action list
    
    cam_0 = action["camera"][:,0]
    cam_1 = action["camera"][:,1]
    
    
    agent_actions = []
    agent_actions.append(action["attack"])
    agent_actions.append(action["back"])
    agent_actions.append(cam_0)
    agent_actions.append(cam_1)
    agent_actions.append(action["craft"])
    agent_actions.append(action["equip"])
    agent_actions.append(action["forward"])
    agent_actions.append(action["jump"])
    agent_actions.append(action["left"])
    agent_actions.append(action["nearbyCraft"])
    agent_actions.append(action["nearbySmelt"])
    agent_actions.append(action["place"])
    agent_actions.append(action["right"])
    agent_actions.append(action["sneak"])
    agent_actions.append(action["sprint"])
    
    entropy_t = 0.0
    for i in range(len(agent_actions)):
        entropy_t = entropy_t + entropy(agent_actions[i])
    #print(entropy_t)
    #print(agent_actions[5])
    #print(entropy(agent_actions[5]))

    vertical_agent_actions = [np.vstack(item) for item in agent_actions]
    concat_agent_actions = np.concatenate(vertical_agent_actions, axis=1)

    experiences = zip(concat_agent_mh, concat_agent_invent, vertical_world_state, concat_agent_actions, reward, concat_next_agent_mh, concat_next_agent_invent, vertical_next_world_state, done)
    #print(concat_agent_invent)
    #print(concat_next_agent_invent)

    experiences = np.array(list(experiences))
    experiences[-1][4] = experiences[-1][4]+entropy_t
    return experiences[-1], agent_mh_ts, agent_inventory_ts
 


env = gym.make("MineRLObtainDiamondDense-v0") 

obs_a = env.reset()

action_a = env.action_space.sample()
mainhand_a = obs_a['equipped_items']['mainhand']
inventory_a = obs_a['inventory']
pov_a = obs_a['pov']

agent_state_s = len(list(mainhand_a.values())) + len(list(inventory_a.values()))
world_state_s = pov_a.shape
action_s = len(list(action_a.values()))


    
data = minerl.data.make(
    'MineRLObtainDiamondDense-v0',
    data_dir="/home/desin/minerl/data")

agent = Agent(agent_mh_size=3, agent_inventory_size = 18, world_state_size=(64, 64, 3), action_size=14, random_seed=0)

action_list = np.zeros((action_s-1, 10), dtype=int)
action_list_names = ("attack", "back", "craft", "equip",
                     "forward", "jump", "left", "nearbyCraft", "nearbySmelt", 
                     "place", "right", "sneak", "sprint")
camera_list = []
camera_action_names = ("yaw", "pitch")

agent_state_list_names = ['damage', 'maxDamage', 'type', 'coal', 'cobblestone', 'crafting_table', 
                    'dirt', 'furnace','iron_axe', 'iron_ingot', 'iron_ore', 'iron_pickaxe', 
                    'log', 'planks', 'stick', 'stone', 'stone_axe', 'stone_pickaxe', 
                    'torch', 'wooden_axe', 'wooden_pickaxe']
agent_state_list = []


pyplot.ion()
pyplot.show()
fig= pyplot.figure()
fig.set_size_inches(12, 6)


# Iterate through a single epoch gathering sequences of at most 32 steps
i=0
done_1=False
active_reward=0
for current_state, action, reward, next_state, done \
    in data.sarsd_iter(
        num_epochs=10, max_sequence_len=32):

        i = i+1

            #continue
        done = np.delete(done, -1)
        experiences, mh_ts, invent_ts = extract_data_from_dict(current_state, action, reward, next_state, done)
        agent.learn_from_players(experiences, mh_ts, invent_ts)
        
        if (np.any(reward)):
        	print("reward...{} ".format(active_reward))
        
        if (done_1==False):
            action_1, action_1_raw,  agent_mh_raw, agent_inventory_raw = agent.act(mainhand_a, inventory_a, pov_a)
            obs_1, reward_1, done_1, info = env.step(action_1)
        
            if (reward_1 >0):
                active_reward = active_reward+1
                print("REWARD !!!!!!!!!!!!!!!!!!!!!!")

            #camera_list.append(action_1_raw[0][2:4].cpu().numpy())
            #agent_state_list.append(agent_state_raw)


            #print((action_1_raw[0])[4:].cpu().int()) 

            #actions_1 = np.concatenate(((action_1_raw[0])[0:2].cpu().int(), (action_1_raw[0])[4:].cpu().int()))
            #camera = action_1_raw[2:4]
            #one_hot_actions =np.zeros((action_s-1,10), dtype=int)
            #one_hot_actions[np.arange(action_s+1), action_1_raw[0].cpu().int()]=1
            #one_hot_actions[np.arange(action_s-1), actions_1]=1

            #action_list = action_list+one_hot_actions

            if (i%100000==0):
                #print(agent_state_raw)
                #print(action_list)
                for i,x in enumerate(action_list_names):
                    max_s = 1+max(action_list[i,:])/100
                    pyplot.scatter([action_list_names[i]]*10, np.arange(10), s=action_list[i,:]/max_s)

                for xe, ye in zip(camera_action_names, [[(camera_list[li])[ci]  for li in range(len(camera_list))] for ci in range(len(camera_action_names)) ]):
                    pyplot.scatter([xe] * len(ye), ye)

                pyplot.draw()
                pyplot.pause(0.001)

        else:
            obs_1 = env.reset()
            done_1=False
            print("RESET ----------------")

        #action_1 = env.action_space.sample()
        #print(action_1)
        #experience = extract_data_from_dict_single(obs_a, action_1, reward_1, obs_1, done_1)
        #agent.add_memory(experience)

        obs_a = obs_1
        mainhand_a = obs_a['equipped_items']['mainhand']
        inventory_a = obs_a['inventory']
        pov_a = obs_a['pov']

        env.render()


print("DONE")

