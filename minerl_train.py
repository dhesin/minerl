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


def extract_data_from_dict_single(current_state, action, reward, next_state, done):

    equipments = {"none":1, 'air':2, 'wooden_axe':3, 'wooden_pickaxe':4, 
              'stone_axe':5, 'stone_pickaxe':6, 'iron_axe':7, 'iron_pickaxe':8}


    # current state
    mainhand = current_state['equipped_items']['mainhand']
    inventory = current_state['inventory']
    pov = current_state['pov']


    agent_state = []
    agent_state.append(mainhand['damage'])
    agent_state.append(mainhand['maxDamage'])
    agent_state.append(mainhand['type'])
    agent_state.append(inventory['coal'])
    agent_state.append(inventory['cobblestone'])
    agent_state.append(inventory['crafting_table'])
    agent_state.append(inventory['dirt'])
    agent_state.append(inventory['furnace'])
    agent_state.append(inventory['iron_axe'])
    agent_state.append(inventory['iron_ingot'])
    agent_state.append(inventory['iron_ore'])
    agent_state.append(inventory['iron_pickaxe'])
    agent_state.append(inventory['log'])
    agent_state.append(inventory['planks'])
    agent_state.append(inventory['stick'])
    agent_state.append(inventory['stone'])
    agent_state.append(inventory['stone_axe'])
    agent_state.append(inventory['stone_pickaxe'])
    agent_state.append(inventory['torch'])
    agent_state.append(inventory['wooden_axe'])
    agent_state.append(inventory['wooden_pickaxe'])

    
    swap_world_state = np.swapaxes(pov,0,2)
    
     
    # next_state
    mainhand = next_state['equipped_items']['mainhand']
    inventory = next_state['inventory']
    pov = next_state['pov']


    agent_state_next = []
    agent_state_next.append(mainhand['damage'])
    agent_state_next.append(mainhand['maxDamage'])
    agent_state_next.append(mainhand['type'])
    agent_state_next.append(inventory['coal'])
    agent_state_next.append(inventory['cobblestone'])
    agent_state_next.append(inventory['crafting_table'])
    agent_state_next.append(inventory['dirt'])
    agent_state_next.append(inventory['furnace'])
    agent_state_next.append(inventory['iron_axe'])
    agent_state_next.append(inventory['iron_ingot'])
    agent_state_next.append(inventory['iron_ore'])
    agent_state_next.append(inventory['iron_pickaxe'])
    agent_state_next.append(inventory['log'])
    agent_state_next.append(inventory['planks'])
    agent_state_next.append(inventory['stick'])
    agent_state_next.append(inventory['stone'])
    agent_state_next.append(inventory['stone_axe'])
    agent_state_next.append(inventory['stone_pickaxe'])
    agent_state_next.append(inventory['torch'])
    agent_state_next.append(inventory['wooden_axe'])
    agent_state_next.append(inventory['wooden_pickaxe'])

 
    swap_next_world_state = np.swapaxes(pov,0,2)
    

    
    # get action list
    
    agent_actions = []
    agent_actions.append(action["attack"])
    agent_actions.append(action["back"])
    agent_actions.append(action["camera"][0])
    agent_actions.append(action["camera"][1])
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
    


    experience = (np.array(agent_state), np.array(swap_world_state), np.array(agent_actions), reward, np.array(agent_state_next), np.array(swap_next_world_state), done)
    #print(agent_state_next)

    return experience
 

def extract_data_from_dict(current_state, action, reward, next_state, done):

    equipments = {"none":1, 'air':2, 'wooden_axe':3, 'wooden_pickaxe':4, 
              'stone_axe':5, 'stone_pickaxe':6, 'iron_axe':7, 'iron_pickaxe':8}


    # current state
    mainhand = current_state['equipped_items']['mainhand']
    inventory = current_state['inventory']
    pov = current_state['pov']


    agent_state = []
    agent_state.append(mainhand['damage'])
    agent_state.append(mainhand['maxDamage'])
    agent_state.append(mainhand['type'])
    agent_state.append(inventory['coal'])
    agent_state.append(inventory['cobblestone'])
    agent_state.append(inventory['crafting_table'])
    agent_state.append(inventory['dirt'])
    agent_state.append(inventory['furnace'])
    agent_state.append(inventory['iron_axe'])
    agent_state.append(inventory['iron_ingot'])
    agent_state.append(inventory['iron_ore'])
    agent_state.append(inventory['iron_pickaxe'])
    agent_state.append(inventory['log'])
    agent_state.append(inventory['planks'])
    agent_state.append(inventory['stick'])
    agent_state.append(inventory['stone'])
    agent_state.append(inventory['stone_axe'])
    agent_state.append(inventory['stone_pickaxe'])
    agent_state.append(inventory['torch'])
    agent_state.append(inventory['wooden_axe'])
    agent_state.append(inventory['wooden_pickaxe'])

    #flat_list = [item for sublist in agent_state for item in sublist]
    vertical_agent_state = [np.vstack(item) for item in agent_state]
    concat_agent_state = np.concatenate(vertical_agent_state, axis=1)
    #print(concat_agent_state)
    #[print(item.shape) for item in pov]
    
    swap_world_state = [np.swapaxes(item,0,2) for item in pov]
    
    #[print(item.shape) for item in swap_world_state]   
    
    vertical_world_state = np.stack(swap_world_state, axis=0)
    #print(vertical_world_state.shape)
    
    #[print(item.shape) for item in vertical_world_state] 

    
    
    
    # next_state
    mainhand = next_state['equipped_items']['mainhand']
    inventory = next_state['inventory']
    pov = next_state['pov']


    agent_state = []
    agent_state.append(mainhand['damage'])
    agent_state.append(mainhand['maxDamage'])
    agent_state.append(mainhand['type'])
    agent_state.append(inventory['coal'])
    agent_state.append(inventory['cobblestone'])
    agent_state.append(inventory['crafting_table'])
    agent_state.append(inventory['dirt'])
    agent_state.append(inventory['furnace'])
    agent_state.append(inventory['iron_axe'])
    agent_state.append(inventory['iron_ingot'])
    agent_state.append(inventory['iron_ore'])
    agent_state.append(inventory['iron_pickaxe'])
    agent_state.append(inventory['log'])
    agent_state.append(inventory['planks'])
    agent_state.append(inventory['stick'])
    agent_state.append(inventory['stone'])
    agent_state.append(inventory['stone_axe'])
    agent_state.append(inventory['stone_pickaxe'])
    agent_state.append(inventory['torch'])
    agent_state.append(inventory['wooden_axe'])
    agent_state.append(inventory['wooden_pickaxe'])

    #flat_list = [item for sublist in agent_state for item in sublist]
    vertical_next_agent_state = [np.vstack(item) for item in agent_state]
    concat_next_agent_state = np.concatenate(vertical_next_agent_state, axis=1)

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
    

    
    vertical_agent_actions = [np.vstack(item) for item in agent_actions]
    concat_agent_actions = np.concatenate(vertical_agent_actions, axis=1)

    experiences = zip(concat_agent_state, vertical_world_state, concat_agent_actions, reward, concat_next_agent_state, vertical_next_world_state, done)

    experiences = np.array(list(experiences))
    return experiences
 


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

agent = Agent(agent_state_size=21, world_state_size=(64, 64, 3), action_size=14, random_seed=0)

action_list =[]
action_list_names = ("attack", "back", "camera_0", "camera_1", "craft", "equip",
                     "forward", "jump", "left", "nearbyCraft", "nearbySmelt", 
                     "place", "right", "sneak", "sprint")
agent_state_list_names = ['damage', 'maxDamage', 'type', 'coal', 'cobblestone', 'crafting_table', 
                    'dirt', 'furnace','iron_axe', 'iron_ingot', 'iron_ore', 'iron_pickaxe', 
                    'log', 'planks', 'stick', 'stone', 'stone_axe', 'stone_pickaxe', 
                    'torch', 'wooden_axe', 'wooden_pickaxe']
agent_state_list = []


pyplot.ion()
pyplot.show()
fig= pyplot.figure()
fig.set_size_inches(16, 8)


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
        experiences = extract_data_from_dict(current_state, action, reward, next_state, done)
        agent.learn_from_players(experiences)
        
        if (np.any(reward)):
        	print("reward...{} ".format(active_reward))
        
        if (done_1==False):
            action_1, action_1_raw,  agent_state_raw = agent.act(mainhand_a, inventory_a, pov_a)
            obs_1, reward_1, done_1, info = env.step(action_1)
        
            if (reward_1 >0):
                active_reward = active_reward+1
                print("REWARD !!!!!!!!!!!!!!!!!!!!!!")

            action_list.append(action_1_raw.numpy())
            agent_state_list.append(agent_state_raw)

            if (i%10==0):
                for xe, ye in zip(action_list_names, [[(action_list[li])[0,ci]  for li in range(len(action_list))] for ci in range(len(action_list_names)) ]):
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

