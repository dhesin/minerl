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
from ddpg_agent import Agent
#%matplotlib inline

import ddpg_agent_sequential
reload(ddpg_agent_sequential)
from ddpg_agent_sequential import Agent_TS

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
 


env = gym.make("MineRLObtainDiamondDense-v0") 
env.seed(1255)
obs_a = env.reset()

action_a = deque(maxlen=16)
mainhand_a = deque(maxlen=16)
inventory_a = deque(maxlen=16)
pov_a = deque(maxlen=16)


action_a.append(env.action_space.sample())
mainhand_a.append(obs_a['equipped_items']['mainhand'])
inventory_a.append(obs_a['inventory'])
pov_a.append(obs_a['pov'])

agent_state_s = len(list(obs_a['equipped_items']['mainhand'].values())) + len(list(obs_a['inventory'].values()))
world_state_s = obs_a['pov'].shape
action_s = len(list(env.action_space.sample().values()))


writer = SummaryWriter()


data = minerl.data.make(
    'MineRLObtainDiamondDense-v0',
    data_dir="/home/darici/minerl/minerl/data")

agent = Agent_TS(agent_mh_size = 3, agent_inventory_size = 18, world_state_size = [3, 32, 64, 64], action_size=14, random_seed=0)


action_counts = np.zeros((action_s-1, 10), dtype=int)
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

loss_list = deque(maxlen=10000)

pyplot.ion()
#pyplot.show()
#fig_1 = pyplot.figure(num=1)
#fig_1.set_size_inches(12, 6)
#pyplot.ylim(-50,50)
#fig_2 = pyplot.figure(num=2)
#pyplot.ylim(0,10)


# Iterate through a single epoch gathering sequences of at most 32 steps
eps_i=0
done_1=False
active_reward=0
for epoch in range(1:10):
    for current_state, action, reward, next_state, done in data.sarsd_iter(num_epochs=1, max_sequence_len=16, seed=0):
            #print(action['camera'])
            eps_i = eps_i+1
            done = np.delete(done, -1)
            experiences = extract_data_from_dict(current_state, action, reward, next_state, done)
            agent.learn_from_players(experiences, writer)
            
            
            if (reward[-1] > 0):
                print("Training Reward:{}".format(reward[-1]))

            if (done_1==False):
                with torch.no_grad():

                    action_1, action_1_raw, _ , _  = agent.act(mainhand_a, inventory_a, pov_a)
                obs_1, reward_1, done_1, info = env.step(action_1)
                if (action_1["forward"] > 1):
                    print(info)
                    print(exit)

                print(obs_1['pov'].shape)
                print(experiences[2].shape)

                #print(action_1_raw)
                if (reward_1 >0):
                    active_reward = active_reward+1
                    print("REWARD !!!!!!!!!!!!!!!!!!!!!! {}".format(active_reward))
                if (active_reward > 0):
                    print("REWARD !!!!! {}".format(active_reward))


                writer.add_scalars('Camera', {'picth':action_1_raw[-1,-1,2], 'yaw':action_1_raw[-1,-1,3]}, global_step=eps_i)


                writer.add_scalars('actions', {"attack":action_1_raw[-1,-1,0], "back":action_1_raw[-1,-1,1], \
                    "craft":action_1_raw[-1,-1,4], "equip":action_1_raw[-1,-1,5], "forward":action_1_raw[-1,-1,6], \
                    "jump":action_1_raw[-1,-1,7], "left":action_1_raw[-1,-1,8], "nearbyCraft":action_1_raw[-1,-1,9], \
                    "nearbySmelt":action_1_raw[-1,-1,10], "place":action_1_raw[-1,-1,11], "right":action_1_raw[-1,-1,12], \
                    "sneak":action_1_raw[-1,-1,13], "sprint":action_1_raw[-1,-1,14]}, global_step=eps_i)


            else:
                print("RESET ----------------")
                active_reward = 0
                env.seed(1255)
                obs_1 = env.reset()
                done_1=False
                

            print(info)

            if eps_i % 1000 == 0:
                print('\nEpisode:{}\t       '.format(eps_i), end="")
                torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
     
            #action_1 = env.action_space.sample()
            #print(action_1)
            #experience = extract_data_from_dict_single(obs_a, action_1, reward_1, obs_1, done_1)
            #agent.add_memory(experience)

            obs_a = obs_1
            mainhand_a.append(obs_a['equipped_items']['mainhand'])
            inventory_a.append(obs_a['inventory'])
            pov_a.append((obs_a['pov']))

            env.render()

    print("num iterations in an epoch:{}".format(eps_i))
    eps_i = 0
    agent.actor_scheduler.step()
    print(agent.actor_scheduler.get_lr())

print("DONE")

