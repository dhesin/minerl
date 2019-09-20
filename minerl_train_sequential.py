import gym
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


    vertical_agent_mh = [np.vstack(item) for item in agent_mh]
    vertical_agent_invent = [np.vstack(item) for item in agent_inventory]
    concat_agent_mh = np.concatenate(vertical_agent_mh, axis=1)
    concat_agent_invent = np.concatenate(vertical_agent_invent, axis=1)
    
    swap_world_state = [np.swapaxes(item,0,2) for item in pov]
    vertical_world_state = np.stack(swap_world_state, axis=0)
    vertical_world_state = np.swapaxes(vertical_world_state, 0, 1)
    
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
    vertical_next_world_state = np.stack(swap_next_world_state, axis=0)
    #print(vertical_next_world_state.shape)
    vertical_next_world_state = np.swapaxes(vertical_next_world_state, 0, 1)
    #print(vertical_next_world_state.shape)
    
    # get action list
    
    cam_0 = action["camera"][:,0]
    cam_1 = action["camera"][:,1]
    #print(cam_0)
    #print(cam_1)
    
    
    agent_actions = []
    agent_actions_onehot = []
    sequence_len = len(action['attack'])

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
    

    agent_actions_onehot.append(action['attack'])
    agent_actions_onehot.append(action['back'])
    agent_actions_onehot.append(cam_0)
    agent_actions_onehot.append(cam_1)
    craft = np.zeros((sequence_len,5))
    craft[np.arange(sequence_len), action['craft']] = 1
    agent_actions_onehot.append(craft.tolist())
    equip = np.zeros((sequence_len,8))
    equip[np.arange(sequence_len), action['equip']] = 1
    agent_actions_onehot.append(equip.tolist())
    agent_actions_onehot.append(action['forward'])
    agent_actions_onehot.append(action['jump'])
    agent_actions_onehot.append(action['left'])
    nearby_craft = np.zeros((sequence_len,8))
    nearby_craft[np.arange(sequence_len),action['nearbyCraft']] = 1
    agent_actions_onehot.append(nearby_craft.tolist())
    nearby_smelt = np.zeros((sequence_len,3))
    nearby_smelt[np.arange(sequence_len), action['nearbySmelt']] = 1
    agent_actions_onehot.append(nearby_smelt.tolist())
    place = np.zeros((sequence_len,7))
    place[np.arange(sequence_len), action['place']] = 1
    agent_actions_onehot.append(place.tolist())
    agent_actions_onehot.append(action['right'])
    agent_actions_onehot.append(action['sneak'])
    agent_actions_onehot.append(action['sprint'])

    #print(agent_actions_onehot)


    entropy_t = 0.0
    #for i in range(len(agent_actions)):
    #    print(entropy(agent_actions[i]))
    

    #print(agent_actions[5])
    #print(entropy(agent_actions[5]))

    vertical_agent_actions = [np.vstack(item) for item in agent_actions]
    concat_agent_actions = np.concatenate(vertical_agent_actions, axis=1)

    vertical_agent_actions_onehot = [np.vstack(item) for item in agent_actions_onehot]
    #print(vertical_agent_actions_onehot)
    concat_agent_actions_onehot = np.concatenate(vertical_agent_actions_onehot, axis=1)
    #print(concat_agent_actions_onehot)


    #print(concat_agent_invent)
    #print(concat_next_agent_invent)
    experiences = (concat_agent_mh, concat_agent_invent, vertical_world_state, concat_agent_actions, reward, concat_next_agent_mh, concat_next_agent_invent, vertical_next_world_state, done)
    #print(concat_agent_invent)
    #print(concat_next_agent_invent)
    #experiences[-1][4] = experiences[-1][4]+entropy_t
    return experiences
 



#action_counts = np.zeros((action_s-1, 10), dtype=int)
action_names = ("attack", "back", "craft", "equip",
                     "forward", "jump", "left", "nearbyCraft", "nearbySmelt", 
                     "place", "right", "sneak", "sprint")
camera_action_names = ("pitch", "yaw")

agent_state_list_names = ['damage', 'maxDamage', 'type', 'coal', 'cobblestone', 'crafting_table', 
                    'dirt', 'furnace','iron_axe', 'iron_ingot', 'iron_ore', 'iron_pickaxe', 
                    'log', 'planks', 'stick', 'stone', 'stone_axe', 'stone_pickaxe', 
                    'torch', 'wooden_axe', 'wooden_pickaxe']


pyplot.ion()
#pyplot.show()
#fig_1 = pyplot.figure(num=1)
#fig_1.set_size_inches(12, 6)
#pyplot.ylim(-50,50)
#fig_2 = pyplot.figure(num=2)
#pyplot.ylim(0,10)



writer = SummaryWriter()

data = minerl.data.make(
    'MineRLObtainDiamondDense-v0',
    data_dir="/home/desin/minerl/data")

sequence_len = 32
sample_len = 1
BUFFER_SIZE = int(1000)  # replay buffer size
#agent = Agent_TS(agent_mh_size = 3, agent_inventory_size = 18, \
#        world_state_size = [3, 32, 64, 64], action_size=14, \
#        random_seed=0, seq_len = sequence_len, actor_chkpt_file="checkpoint_actor.pth")

agent = Agent_TS(agent_mh_size = 3, agent_inventory_size = 18, \
        world_state_size = [3, 32, 64, 64], action_size=14, \
        random_seed=0, seq_len = sequence_len)




def learn_from_buffer():
    # Learn without the environment
    eps_i=0
    while True:
        eps_i = eps_i+1
        agent.iter = agent.iter+1    
        experiences = agent.memory.sample_sequence()  
        loss_1, loss_2 = agent.learn_2(experiences, 1., writer)
        print("stepping")
        agent.actor_scheduler.step()

        if eps_i % 50 == 0:
            print('\nEpisode:{}\t       '.format(eps_i), end="")
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        if eps_i >= BUFFER_SIZE:
            break


# Put data into memory
for epoch in range(10):
    for current_state, action, reward, next_state, done in data.sarsd_iter(num_epochs=1, max_sequence_len=sample_len, seed=0):
        done = np.delete(done, -1)
        experiences = extract_data_from_dict(current_state, action, reward, next_state, done)
        agent.memory.add(experiences)
        print("memory size:{}".format(len(agent.memory.memory)))
        if (len(agent.memory.memory) >= BUFFER_SIZE):
            agent.memory.update_priorities()
            learn_from_buffer()
            agent.memory.memory.clear()
            agent.memory.pos = 0


