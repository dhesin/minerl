import numpy as np
import random
import copy
from collections import namedtuple, deque
from importlib import reload
from PIL import Image
from matplotlib.pyplot import imshow

import model_sequential
from model_sequential import Actor_TS, Critic_TS
reload(model_sequential)
from model_sequential import Actor_TS, Critic_TS


import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


BUFFER_SIZE = int(5e4)  # replay buffer size
BATCH_SIZE = 4         # minibatch size
GAMMA = 1.0            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.000   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

equipments = {"none":1, 'air':2, 'wooden_axe':3, 'wooden_pickaxe':4, 
              'stone_axe':5, 'stone_pickaxe':6, 'iron_axe':7, 'iron_pickaxe':8}



    
class Agent_TS():
    """Interacts with and learns from the environment."""

    def __init__(self, **kwargs):
        """Initialize an Agent object.
        
        Params
        ======
        """

        self.agent_mh_size = kwargs['agent_mh_size']
        self.agent_inventory_size = kwargs['agent_inventory_size']
        self.world_state_size = kwargs['world_state_size']
        self.action_size = kwargs['action_size']
        self.seed = kwargs['random_seed']
        self.iter = 0
        self.noise_scale = 1.0
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor_TS(self.agent_mh_size, self.agent_inventory_size, self.world_state_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor_TS(self.agent_mh_size, self.agent_inventory_size, self.world_state_size, self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        #self.actor_optimizer = optim.Adam(self.actor_local.parameters())
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=200, gamma=0.99)
        
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic_TS(self.agent_mh_size, self.agent_inventory_size, self.world_state_size, self.action_size, self.seed).to(device)
        self.critic_target = Critic_TS(self.agent_mh_size, self.agent_inventory_size, self.world_state_size, self.action_size, self.seed).to(device)

        params = list(self.critic_local.parameters()) + list(self.actor_local.parameters())
        #self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic_optimizer = optim.Adam(params, lr=LR_CRITIC)
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=200, gamma=0.99)
        
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)


        # Noise process
        self.noise = OUNoise(self.action_size, self.seed)

        # Replay memory
        #self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)
        # Prioritized replay memory
        self.memory = NaivePrioritizedBuffer(BUFFER_SIZE, BATCH_SIZE, self.seed)

        if 'actor_chkpt_file' in kwargs and 'critic_chkpt_file' in kwargs:
            checkpoint_actor = torch.load(kwargs['actor_chkpt_file'])
            checkpoint_critic = torch.load(kwargs['critic_chkpt_file'])
            self.actor_local.load_state_dict(checkpoint_actor)
            self.critic_local.load_state_dict(checkpoint_critic)
            checkpoint_actor_t = torch.load(kwargs['actor_chkpt_file_t'])
            checkpoint_critic_t = torch.load(kwargs['critic_chkpt_file_t'])
            self.actor_target.load_state_dict(checkpoint_actor_t)
            self.critic_target.load_state_dict(checkpoint_critic_t)

    def flatten_action(self, action):
        
        action_flat = []
        for x in action:
            if type(x) is list:
                for y in x:
                    action_flat.append(y)
            else:
                action_flat.append(x)
        return action_flat

    def get_states(self, mainhand, inventory, pov):


        mainhand = np.vstack(mainhand)
        inventory = np.vstack(inventory)
        

        agent_state_mainhand = []
        for i in range(len(mainhand)):
            agent_state_mainhand.append((mainhand[i,0]['damage'], mainhand[i,0]['maxDamage'], equipments.get(mainhand[i,0]['type'], -1)))

        agent_state_inventory = []
        for i in range(len(inventory)):
            agent_state_inventory.append((inventory[i,0]['coal'], inventory[i,0]['cobblestone'], \
                inventory[i,0]['crafting_table'], inventory[i,0]['dirt'], inventory[i,0]['furnace'], \
                inventory[i,0]['iron_axe'], inventory[i,0]['iron_ingot'], inventory[i,0]['iron_ore'], \
                inventory[i,0]['iron_pickaxe'], inventory[i,0]['log'], inventory[i,0]['planks'], \
                inventory[i,0]['stick'], inventory[i,0]['stone'], inventory[i,0]['stone_axe'], \
                inventory[i,0]['stone_pickaxe'], inventory[i,0]['torch'], inventory[i,0]['wooden_axe'], \
                inventory[i,0]['wooden_pickaxe']))


        agent_state_mainhand = np.expand_dims(np.array(agent_state_mainhand), axis=0)
        agent_state_inventory = np.expand_dims(np.array(agent_state_inventory), axis=0)

        #print(len(pov))
        world_state_a = np.stack(pov, axis=0)
        #print(world_state_a.shape)
        world_state_a = np.expand_dims(world_state_a, axis=0)
        #print(world_state_a.shape)
        world_state_a = np.swapaxes(world_state_a, 2, 4)
        #print(world_state_a.shape)
        world_state_a = np.swapaxes(world_state_a, 1, 2)
        #print(world_state_a.shape)

        return agent_state_mainhand, agent_state_inventory, world_state_a

        
    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def step(self, mainhand, inventory, pov, action, reward, mainhand_n, inventory_n, pov_n, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        agent_state_mainhand, agent_state_inventory, world_state = self.get_states(mainhand, inventory, pov)
        agent_state_mainhand_n, agent_state_inventory_n, world_state_n = self.get_states(mainhand_n, inventory_n, pov_n)
        
        self.memory.add(agent_state_mainhand, agent_state_inventory, world_state, action, reward, agent_state_mainhand_n, agent_state_inventory_n, world_state_n, done)
        
        # Learn, if enough samples are available in memory
        self.iter = self.iter+1
        self.iter = self.iter%1
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        #self.actor_scheduler.step()
        #self.critic_scheduler.step()


    def learn_from_players(self, experiences, writer):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        self.memory.add(experiences)

        # Learn, if enough samples are available in memory

        if len(self.memory) > BATCH_SIZE:
            
            
            #(states, states_2, actions, rewards, next_states, next_states_2, dones) = experiences      
            self.iter = self.iter+1    
            experiences = self.memory.sample()  
            loss_1, loss_2 = self.learn_2(experiences, GAMMA, writer)
            
            self.iter = self.iter+1
            experiences = self.memory.sample()
            loss_1, loss_2 = self.learn_2(experiences, GAMMA, writer)

        #self.actor_scheduler.step()
        #self.critic_scheduler.step()

    
    def act(self, mainhand, inventory, pov,  add_noise=True, noise_scale=1.0):
        """Returns actions for given state as per current policy."""

        agent_state_mainhand, agent_state_inventory, world_state = self.get_states(mainhand, inventory, pov)   

        #pil_img = transforms.ToPILImage()(world_state)
        #imshow(pil_img)
        
        s1 = torch.from_numpy(agent_state_mainhand).float().to(device)
        s3 = torch.from_numpy(agent_state_inventory).float().to(device)
        s2 = torch.from_numpy(world_state).float().to(device) 


        self.actor_local.eval()
        with torch.no_grad():
            action, action_raw ,_,  _ ,  _ , _ , _ , _, _, _= self.actor_local(s1,s2,s3)
            
        self.actor_local.train()
        
        return action, action_raw, agent_state_mainhand, agent_state_inventory

    def reset(self):
        self.noise.reset()

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def get_action_loss(self, writer, gt, onehot_probs, rewards, q_diff_loss=None, q_value_loss=None):

        onehot_probs = onehot_probs.view(-1,41)
        gt = gt.view(-1,15)

        #b_r = rewards.sum(dim=1)/16
        #for i in range(rewards.shape[0]):
        #    rewards[i,:] = rewards[i,:]+b_r[i]
        #rewards = rewards.view(-1)


        attack_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,0], gt[:,0])
        back_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,1], gt[:,1])
        pitch_loss = F.mse_loss(onehot_probs[:,2], gt[:,2])
        yaw_loss = F.mse_loss(onehot_probs[:,3], gt[:,3])
        craft_loss = F.cross_entropy(onehot_probs[:,4:9], gt[:,4].long())
        equip_loss = F.cross_entropy(onehot_probs[:,9:17], gt[:,5].long())
        forward_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,17], gt[:,6])
        jump_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,18], gt[:,7])
        left_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,19], gt[:,8])
        nearby_craft_loss = F.cross_entropy(onehot_probs[:,20:28], gt[:,9].long())
        nearby_smelt_loss = F.cross_entropy(onehot_probs[:,28:31], gt[:,10].long())
        place_loss = F.cross_entropy(onehot_probs[:,31:38], gt[:,11].long())
        right_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,38], gt[:,12])
        sneak_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,39], gt[:,13])
        sprint_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,40], gt[:,14])
        

        writer.add_scalars('Losses', {"attack":attack_loss, "back":back_loss, \
            "craft":craft_loss, "equip":equip_loss, "forward":forward_loss, \
            "jump":jump_loss, "left":left_loss, "nearbyCraft":nearby_craft_loss, \
            "nearbySmelt":nearby_smelt_loss, "place":place_loss, "right":right_loss, \
            "sneak":sneak_loss, "sprint":sprint_loss}, global_step=self.iter)

        writer.add_scalars('Camera Losses', {"pitch":pitch_loss, "yaw":yaw_loss}, global_step=self.iter)

        #writer.add_scalars('State Prediction Losses', {"MainHand":mh_state_loss, "Inventory":inventory_state_loss, "World":world_state_loss}, global_step=self.iter)


        self.actor_optimizer.zero_grad()
        #self.critic_optimizer.zero_grad()

        if q_value_loss is None and q_diff_loss is None:
            torch.autograd.backward([attack_loss,back_loss,pitch_loss,yaw_loss,craft_loss,equip_loss,\
                    forward_loss,jump_loss,left_loss,nearby_craft_loss,nearby_smelt_loss,place_loss, \
                    right_loss,sneak_loss,sprint_loss])
        else:

            q_value_loss = q_value_loss.detach()/14
            writer.add_scalars('Q Values', {"Q Value":q_value_loss, "Q Difference":q_diff_loss}, global_step=self.iter)

            torch.autograd.backward([attack_loss,back_loss,pitch_loss,yaw_loss,craft_loss,equip_loss,\
                    forward_loss,jump_loss,left_loss,nearby_craft_loss,nearby_smelt_loss,place_loss, \
                    right_loss,sneak_loss,sprint_loss, q_diff_loss])


        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        
        self.actor_optimizer.step()
        #self.critic_optimizer.step()
        
        #print(attack_loss)
        #print(back_loss)
        #print(camera_loss.item())
        #print(craft_loss)
        #print(equip_loss)
        #print(forward_loss)
        #print(jump_loss)
        #print(left_loss)
        #print(nearby_craft_loss)
        #print(nearby_smelt_loss)
        #print(place_loss)
        #print(right_loss)
        #print(sneak_loss)
        #print(sprint_loss)

        return pitch_loss, yaw_loss




    def learn_2(self, experiences, gamma, writer):
        
        #states, actions, rewards, next_states, dones, indices, weights = experiences
        #( a_states_mh, a_states_invent, w_states, actions, rewards, a_next_states_mh, a_next_states_invent, w_next_states, dones ) = experiences

        #pil_img = transforms.ToPILImage()(experiences[0][2])
        #imshow(pil_img)


        a_states_mh = torch.tensor([item[0] for item in experiences]).float()
        a_states_invent = torch.tensor([item[1] for item in experiences]).float()
        w_states = torch.tensor([item[2] for item in experiences]).float()
        actions = torch.tensor([item[3] for item in experiences]).float()
        rewards = torch.tensor([item[4] for item in experiences]).float()
        a_next_states_mh = torch.tensor([item[5] for item in experiences]).float()
        a_next_states_invent = torch.tensor([item[6] for item in experiences]).float()
        w_next_states = torch.tensor([item[7] for item in experiences]).float()
        dones = torch.tensor([item[8] for item in experiences]).float()


        a_states_mh = a_states_mh.to(device)
        a_states_invent = a_states_invent.to(device)
        w_states = w_states.to(device)
        actions = actions.to(device)
        rewards = rewards.to(device)

        a_next_states_mh = a_next_states_mh.to(device)
        a_next_states_invent = a_next_states_invent.to(device)
        w_next_states = w_next_states.to(device)
        dones = dones.to(device)

        #get next state (from experiences) descriptors and Q_next
        with torch.no_grad():
            _, _, _, Q_next, _, _, _, wsd_next, mhd_next, inventd_next = \
                self.actor_local(a_next_states_mh, w_next_states, a_next_states_invent)

    
            # Q_next = Q_next.squeeze().detach()
            # Q_current_2 = rewards.sum(dim=1) + (gamma * Q_next * (1 - dones[:,-1]))
            # wsd_next = wsd_next.detach()
            # mhd_next = mhd_next.detach()
            # inventd_next = inventd_next.detach()


        # predict actions and next state with 
        _, action_raw, action_logits, Q_current, n_wsd_predict, n_asmhd_predict, n_asinventd_predict, _, _, _ = \
                self.actor_local(a_states_mh, w_states, a_states_invent)

        #Q_current = Q_current.squeeze(dim=1)

        # calculate loss for actor
        #loss_1, loss_2 = self.get_action_loss(writer, actions[:,-1,:], action_logits, \
        #        F.mse_loss(mhd_next, n_asmhd_predict), F.mse_loss(inventd_next, n_asinventd_predict), \
        #        F.mse_loss(wsd_next, n_wsd_predict), F.mse_loss(Q_current, Q_current_2), -Q_current.mean())

        loss_1, loss_2 = self.get_action_loss(writer, actions, action_logits, rewards)

        print("Actor Losses:{} {}".format(loss_1.item(), loss_2.item()))
        return loss_1, loss_2



class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        #self.sigma = 0.99*self.sigma
        #self.theta = 0.99*self.theta
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


    
class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, batch_size, seed, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.memory     = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        np.random.seed(seed)
    
    def add(self, experiences):

        if len(experiences[0])<16:
            return

        if (experiences[4].sum() == 0 and random.random()<=0.25):
            return
   
        if len(self.memory) < self.capacity:
            print("event memory capacity below:")
            self.memory.append(experiences)
        else:
            self.memory[self.pos] = experiences

        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, beta=0.4):
        
        #indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        indices = np.random.choice(len(self.memory), self.batch_size, False)
        #indices = np.random.randint(0, len(self.memory), self.batch_size)

        
        experiences = [self.memory[idx] for idx in indices if indices is not None]
        return experiences
    
    def update_priorities(self, batch_indices, batch_priorities):
        
        #print(batch_indices)
        #print(batch_priorities)
        
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)        
