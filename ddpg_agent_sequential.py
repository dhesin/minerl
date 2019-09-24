import numpy as np
import random
import copy
from collections import namedtuple, deque
from importlib import reload
from PIL import Image
from matplotlib.pyplot import imshow

import model_sequential
from model_sequential import Actor_TS
reload(model_sequential)
from model_sequential import Actor_TS


import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

BUFFER_SIZE = int(5e4)  # replay buffer size
BATCH_SIZE = 4         # minibatch size
GAMMA = 1.0            # discount factor
TAU = 1e-5              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor 

LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.000   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

equipments = {"none":1, 'air':2, 'wooden_axe':3, 'wooden_pickaxe':4, 
              'stone_axe':5, 'stone_pickaxe':6, 'iron_axe':7, 'iron_pickaxe':8}

action_names = ['attack', 'back', 'pitch', 'yaw', 'craft', 'equip', 'forward', 'jump',\
        'left', 'nearbyCraft', 'nearbySmelt', 'place', 'right', 'sneak', 'sprint']


    
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
        self.seq_len = kwargs['seq_len']
        self.iter = 0
        self.iter_2 = 0
        self.noise_scale = 1.0
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor_TS(self.agent_mh_size, self.agent_inventory_size,\
                self.world_state_size, self.action_size, self.seed, self.seq_len).to(device)
        self.actor_target = Actor_TS(self.agent_mh_size, self.agent_inventory_size,\
                self.world_state_size, self.action_size, self.seed, self.seq_len).to(device)
        self.actor_optimizer = optim.Adam([{'params':self.actor_local.cnn.parameters()},\
                {'params':self.actor_local.normalize_inventory.parameters()},\
                {'params':self.actor_local.inventory_lstm.parameters()},\
                {'params':self.actor_local.normalize_mh.parameters()},\
                {'params':self.actor_local.mh_lstm.parameters()},\
                {'params':self.actor_local.cnn_mh_inventory.parameters()},\
                {'params':self.actor_local.output_action_modules['attack'].parameters()},\
                {'params':self.actor_local.output_action_modules['back'].parameters()},\
                #{'params':self.actor_local.output_action_modules['camera'].parameters()},\
                {'params':self.actor_local.output_action_modules['craft'].parameters()},\
                {'params':self.actor_local.output_action_modules['equip'].parameters()},\
                {'params':self.actor_local.output_action_modules['forward_'].parameters()},\
                {'params':self.actor_local.output_action_modules['jump'].parameters()},
                {'params':self.actor_local.output_action_modules['left'].parameters()},\
                {'params':self.actor_local.output_action_modules['nearbyCraft'].parameters()},\
                {'params':self.actor_local.output_action_modules['nearbySmelt'].parameters()},\
                {'params':self.actor_local.output_action_modules['place'].parameters()},\
                {'params':self.actor_local.output_action_modules['right'].parameters()},\
                {'params':self.actor_local.output_action_modules['sneak'].parameters()},\
                {'params':self.actor_local.output_action_modules['sprint'].parameters()},\
                {'params':self.actor_local.qvalue.parameters()},\
                {'params':self.actor_local.camera_yaw.parameters()},\
                {'params':self.actor_local.camera_pitch.parameters()},\
                ], lr=LR_ACTOR)
        #self.actor_optimizer = optim.Adam(self.actor_local.parameters())
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=1000, gamma=0.99)
        
        self.hard_copy_weights(self.actor_target, self.actor_local)

        # Replay memory
        #self.memory = ReplayBuffer(self.action_size, BUFFER_SIZE, BATCH_SIZE, self.seed)
        # Prioritized replay memory
        self.memory = NaivePrioritizedBuffer(BUFFER_SIZE, BATCH_SIZE, self.seed)

        if 'actor_chkpt_file' in kwargs:
            checkpoint_actor = torch.load(kwargs['actor_chkpt_file'])
            #checkpoint_critic = torch.load(kwargs['critic_chkpt_file'])
            self.actor_local.load_state_dict(checkpoint_actor)
            #self.critic_local.load_state_dict(checkpoint_critic)
            #checkpoint_actor_t = torch.load(kwargs['actor_chkpt_file_t'])
            #checkpoint_critic_t = torch.load(kwargs['critic_chkpt_file_t'])
            #self.actor_target.load_state_dict(checkpoint_actor_t)
            #self.critic_target.load_state_dict(checkpoint_critic_t)

    # def flatten_action(self, action):
        
    #     action_flat = []
    #     for x in action:
    #         if type(x) is list:
    #             for y in x:
    #                 action_flat.append(y)
    #         else:
    #             action_flat.append(x)
    #     return action_flat

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
            action, action_raw ,_,  _ = self.actor_local(s1,s2,s3)
            
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

    def get_action_loss(self, writer, gt, onehot_probs, q_exp, q_current, rewards):

        
        #onehot_probs = onehot_probs[:,-1,:]
        gt = gt[:,-1,:]
        onehot_probs = onehot_probs.view(-1,41)
        gt = gt.view(-1,15)

        #b_r = rewards.sum(dim=1)/16
        #for i in range(rewards.shape[0]):
        #    rewards[i,:] = rewards[i,:]+b_r[i]
        #rewards = rewards.view(-1)

        #print(rewards.shape)


        attack_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,0], gt[:,0])#-rewards[:,-1].sum()
        back_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,1], gt[:,1])#-rewards[:,-1].sum()
        pitch_loss = F.mse_loss(onehot_probs[:,2], gt[:,2])#-rewards[:,-1].sum()
        yaw_loss = F.mse_loss(onehot_probs[:,3], gt[:,3])#-rewards[:,-1].sum()
        craft_loss = F.cross_entropy(onehot_probs[:,4:9], gt[:,4].long())#-rewards[:,-1].sum()
        equip_loss = F.cross_entropy(onehot_probs[:,9:17], gt[:,5].long())#-rewards[:,-1].sum()
        forward_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,17], gt[:,6])#-rewards[:,-1].sum()
        jump_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,18], gt[:,7])#-rewards[:,-1].sum()
        left_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,19], gt[:,8])#-rewards[:,-1].sum()
        nearby_craft_loss = F.cross_entropy(onehot_probs[:,20:28], gt[:,9].long())#-rewards[:,-1].sum()
        nearby_smelt_loss = F.cross_entropy(onehot_probs[:,28:31], gt[:,10].long())#-rewards[:,-1].sum()
        place_loss = F.cross_entropy(onehot_probs[:,31:38], gt[:,11].long())#-rewards[:,-1].sum()
        right_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,38], gt[:,12])#-rewards[:,-1].sum()
        sneak_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,39], gt[:,13])#-rewards[:,-1].sum()
        sprint_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,40], gt[:,14])#-rewards[:,-1].sum()
        q_diff_loss = F.mse_loss(q_exp, q_current)
        q_loss = -q_current.sum()
        

        writer.add_scalars('Losses', {"attack":attack_loss, "back":back_loss, \
            "craft":craft_loss, "equip":equip_loss, "forward":forward_loss, \
            "jump":jump_loss, "left":left_loss, "nearbyCraft":nearby_craft_loss, \
            "nearbySmelt":nearby_smelt_loss, "place":place_loss, "right":right_loss, \
            "sneak":sneak_loss, "sprint":sprint_loss}, global_step=self.iter)

        writer.add_scalars('Camera Losses', {"pitch":pitch_loss, "yaw":yaw_loss}, global_step=self.iter)

        #writer.add_scalars('State Prediction Losses', {"MainHand":mh_state_loss, "Inventory":inventory_state_loss, "World":world_state_loss}, global_step=self.iter)


        self.actor_optimizer.zero_grad()
        #self.critic_optimizer.zero_grad()

        if q_exp is None and q_current is None:
            torch.autograd.backward([attack_loss,back_loss,pitch_loss,yaw_loss,craft_loss,equip_loss,\
                    forward_loss,jump_loss,left_loss,nearby_craft_loss,nearby_smelt_loss,place_loss, \
                    right_loss,sneak_loss, sprint_loss, q_diff_loss, q_loss])
        else:

            writer.add_scalars('Q Values_', {"Q Value":q_loss, "Q Difference":q_diff_loss}, global_step=self.iter)

            torch.autograd.backward([attack_loss,back_loss,pitch_loss,yaw_loss,craft_loss,equip_loss,\
                    forward_loss,jump_loss,left_loss,nearby_craft_loss,nearby_smelt_loss,place_loss, \
                    right_loss,sneak_loss, sprint_loss])


        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        
        self.actor_optimizer.step()
        #self.critic_optimizer.step()
        
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
            _, _, _, Q_next = self.actor_local(a_next_states_mh, w_next_states, a_next_states_invent)    
            #Q_next = Q_next[:,-1].detach()
            Q_next = Q_next.detach()


        rewards = self.actor_local.normalize_rewards(rewards)
        #for i in range(rewards.shape[0]):
        #    rewards[i,:] = rewards[i,:] + rewards[i].sum()

        Q_current_2 = rewards[:,-1].squeeze() + (gamma * Q_next * (1 - dones[:,-1,:].squeeze()))

        # predict actions and next state with 
        _, action_raw, action_logits, Q_current = self.actor_local(a_states_mh, w_states, a_states_invent)
        #Q_current = Q_current[:,-1]


        #loss_1, loss_2 = self.get_action_loss(writer, actions[:,-1,:].unsqueeze(dim=1), action_logits[:,-1,:].unsqueeze(dim=1), Q_current_2, Q_current, rewards)
        loss_1, loss_2 = self.get_action_loss(writer, actions[:,-1,:].unsqueeze(dim=1), action_logits, Q_current_2, Q_current, rewards)


        for i in range(action_raw.shape[0]):
            for k in range(action_raw.shape[1]):
                label = "Action_" + action_names[k] + "_"
                writer.add_scalars(label, {"GT":actions[i,-1,k], "Run":action_raw[i,k]}, global_step=self.iter_2)
            self.iter_2 = self.iter_2+1


        # ----------------------- update target networks ----------------------- #
        #self.soft_update(self.actor_local, self.actor_target, TAU)

        print("Actor Losses:{} {}".format(loss_1.item(), loss_2.item()))
        return loss_1, loss_2

    
class NaivePrioritizedBuffer(object):
    def __init__(self, capacity, batch_size, seed, prob_alpha=0.6):
        self.prob_alpha = prob_alpha
        self.capacity   = capacity
        self.memory     = []
        self.lottery    = []
        self.pos        = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.batch_size = batch_size
        self.seed = random.seed(seed)
        np.random.seed(seed)
    
    def add(self, experiences):

        if len(self.memory) < self.capacity:
            print("event memory capacity below:{}".format(experiences[4].sum().item()))
            self.memory.append(experiences)
        else:
            self.memory[self.pos] = experiences

        self.priorities[self.pos] = experiences[4]+0.01

        if (random.random()<0.005 and experiences[4] < 1):
            self.lottery.append(self.pos)
        elif self.pos >= 32:
            self.lottery.append(self.pos)


        self.pos = (self.pos + 1) % self.capacity

    
    def sample(self, beta=0.4):
        
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs /= probs.sum()
        
        #indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        indices = np.random.choice(len(self.memory), self.batch_size, False, p=probs)
        experiences = [self.memory[idx] for idx in indices if indices is not None]
        return experiences
    
    def update_priorities(self):

        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        action_priorities = np.zeros(len(self.memory[0][3][0]))

        print(action_priorities.shape)

        for i in range(len(self.memory)):
            actions = self.memory[i][3]
            actions_len = len(actions[0])
            for j in range(actions_len) :
                if np.any(actions[:,j]):
                    action_priorities[j] = action_priorities[j]+1

        action_priorities = len(self.memory)/(action_priorities+0.1)
        action_priorities = action_priorities ** self.prob_alpha
        print(action_priorities)

        for i in range(len(self.memory)):
            actions = self.memory[i][3]
            actions_len = len(actions[0])
            self.priorities[i] =  ((self.memory[i][4]))#*actions_len)*(self.memory[i][4]*actions_len))
            #for j in range(actions_len):
            #    if (np.any(actions[:,j])):
            #        self.priorities[i] = self.priorities[i] + action_priorities[j] 
                #for k in range(i-sequence_len, i+1):
                #    self.priorities[i] = self.priorities[i] + self.experiences[k][4] 

    def sample_sequence(self, beta=0.4):
        
        # if len(self.memory) == self.capacity:
        #     prios = self.priorities[32,:]
        # else:
        #     prios = self.priorities[32:self.pos]
        
        # probs  = prios ** 4#self.prob_alpha
        # probs = probs/probs.sum()
        
        # #indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        # indices = np.random.choice(np.arange(32,len(self.memory)), self.batch_size, False, p=probs)



        experiences = []
        
        for i in range(self.batch_size):
            ts_0 = []
            ts_1 = []
            ts_2 = []
            ts_3 = []
            ts_4 = []
            ts_5 = []
            ts_6 = []
            ts_7 = []
            ts_8 = []
            assert(self.indice_index<=len(self.sampled_indices))
            idx = self.sampled_indices[self.indice_index]
            self.indice_index +=1
            for j in range(32):
                ts_0.append(self.memory[idx-32+j][0])
                ts_1.append(self.memory[idx-32+j][1])
                ts_2.append(self.memory[idx-32+j][2])
                #print(self.memory[idx-32+j][2].shape)
                ts_3.append(self.memory[idx-32+j][3])
                ts_4.append(self.memory[idx-32+j][4])
                ts_5.append(self.memory[idx-32+j][5])
                ts_6.append(self.memory[idx-32+j][6])
                ts_7.append(self.memory[idx-32+j][7])
                ts_8.append(self.memory[idx-32+j][8])
            ts_00 = np.vstack(ts_0)
            ts_11 = np.vstack(ts_1)
            ts_22 = np.stack(ts_2, axis=1).squeeze()
            ts_33 = np.vstack(ts_3)
            ts_44 = np.vstack(ts_4).squeeze()
            ts_55 = np.vstack(ts_5)
            ts_66 = np.vstack(ts_6)
            ts_77 = np.stack(ts_7, axis=1).squeeze()
            ts_88 = np.vstack(ts_8)
            #print(ts_22.shape)
            experiences.append((ts_00, ts_11, ts_22, ts_33, ts_44, ts_55, ts_66, ts_77, ts_88))

        return experiences
    
    def sample_4_multiple_batch(self, num_batches):
        

        action_priorities = np.zeros(len(self.memory[0][3][0]))

        print(action_priorities.shape)

        for i in range(len(self.memory)):
            actions = self.memory[i][3]
            actions_len = len(actions[0])
            for j in range(actions_len) :
                if np.any(actions[:,j]):
                    action_priorities[j] = action_priorities[j]+1

        action_priorities = len(self.memory)/(action_priorities**2 + 0.1)
        #action_priorities = action_priorities ** 2#self.prob_alpha
        print(action_priorities)

        for i in range(len(self.memory)):
            actions = self.memory[i][3]
            actions_len = len(actions[0])
            for j in range(actions_len):
                if (np.any(actions[:,j])):
                    self.priorities[i] = self.priorities[i] + action_priorities[j] 


        if len(self.memory) == self.capacity:
            prios = self.priorities[32,:]
        else:
            prios = self.priorities[32:self.pos]
        
        prios = self.priorities[self.lottery]
        probs  = prios#self.prob_alpha
        probs = probs/probs.sum()
        
        #indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        assert(len(self.lottery) >= int(num_batches)*self.batch_size)
        self.sampled_indices = np.random.choice(self.lottery, int(num_batches)*self.batch_size, False, p=probs)
        self.indice_index = 0


    def __len__(self):
        return len(self.memory)        
