import numpy as np
import random
import copy
from collections import namedtuple, deque
from importlib import reload
from PIL import Image
from matplotlib.pyplot import imshow

import model
from model import Actor, Critic
reload(model)
from model import Actor, Critic


import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms


BUFFER_SIZE = int(5e4)  # replay buffer size
BATCH_SIZE = 16         # minibatch size
GAMMA = 1.0            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR_ACTOR = 1e-5         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0.000   # L2 weight decay

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

equipments = {"none":1, 'air':2, 'wooden_axe':3, 'wooden_pickaxe':4, 
              'stone_axe':5, 'stone_pickaxe':6, 'iron_axe':7, 'iron_pickaxe':8}



    
class Agent():
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
        self.actor_local = Actor(self.agent_mh_size, self.agent_inventory_size, self.world_state_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor(self.agent_mh_size, self.agent_inventory_size, self.world_state_size, self.action_size, self.seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        #self.actor_optimizer = optim.Adam(self.actor_local.parameters())
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=200, gamma=0.99)
        
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.agent_mh_size, self.agent_inventory_size, self.world_state_size, self.action_size, self.seed).to(device)
        self.critic_target = Critic(self.agent_mh_size, self.agent_inventory_size, self.world_state_size, self.action_size, self.seed).to(device)

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
        agent_state_mainhand = []
        agent_state_mainhand.append(mainhand['damage'])
        agent_state_mainhand.append(mainhand['maxDamage'])
        agent_state_mainhand.append(equipments.get(mainhand['type'], -1))

        agent_state_inventory = []
        agent_state_inventory.append(inventory['coal'])
        agent_state_inventory.append(inventory['cobblestone'])
        agent_state_inventory.append(inventory['crafting_table'])
        agent_state_inventory.append(inventory['dirt'])
        agent_state_inventory.append(inventory['furnace'])
        agent_state_inventory.append(inventory['iron_axe'])
        agent_state_inventory.append(inventory['iron_ingot'])
        agent_state_inventory.append(inventory['iron_ore'])
        agent_state_inventory.append(inventory['iron_pickaxe'])
        agent_state_inventory.append(inventory['log'])
        agent_state_inventory.append(inventory['planks'])
        agent_state_inventory.append(inventory['stick'])
        agent_state_inventory.append(inventory['stone'])
        agent_state_inventory.append(inventory['stone_axe'])
        agent_state_inventory.append(inventory['stone_pickaxe'])
        agent_state_inventory.append(inventory['torch'])
        agent_state_inventory.append(inventory['wooden_axe'])
        agent_state_inventory.append(inventory['wooden_pickaxe'])
                
        
        
        agent_state_mainhand = np.array(agent_state_mainhand)
        agent_state_inventory = np.array(agent_state_inventory)
        world_state_a = np.array(pov)
        world_state_b = np.swapaxes(world_state_a,0,2)
                
        return agent_state_mainhand, agent_state_inventory, world_state_b

        
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


    def learn_from_players(self, experiences, mh_ts, invent_ts, writer):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        #print(experiences)
        e = experiences
        self.memory.add(e[0], e[1], e[2], e[3], e[4], e[5], e[6], e[7], e[8])

        # Learn, if enough samples are available in memory
        self.iter = self.iter+1

        if len(self.memory) > BATCH_SIZE:
            
            experiences = self.memory.sample() 
            #(states, states_2, actions, rewards, next_states, next_states_2, dones) = experiences           
            loss_1, loss_2 = self.learn_2(experiences, GAMMA)
            writer.add_scalar('loss 1', loss_1)
            writer.add_scalar('loss 2', loss_2)
            
            experiences = self.memory.sample()
            loss_1, loss_2 = self.learn_2(experiences, GAMMA)
            writer.add_scalar('loss 1', loss_1)
            writer.add_scalar('loss 2', loss_2)

        #self.actor_scheduler.step()
        #self.critic_scheduler.step()

    
    def act(self, mainhand, inventory, pov,  add_noise=True, noise_scale=1.0):
        """Returns actions for given state as per current policy."""

        agent_state_mainhand, agent_state_inventory, world_state = self.get_states(mainhand, inventory, pov)        
        
        s1 = torch.from_numpy(agent_state_mainhand).float().unsqueeze(dim=0).to(device)
        s3 = torch.from_numpy(agent_state_inventory).float().unsqueeze(dim=0).to(device)

        s2 = torch.from_numpy(world_state).float().unsqueeze(dim=0).to(device) 

        
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

    def get_action_loss(self, gt, onehot_probs, mh_state_loss, inventory_state_loss, \
        world_state_loss, q_diff_loss=None, q_value_loss=None):

        attack_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,0], gt[:,0])
        back_loss = F.binary_cross_entropy_with_logits(onehot_probs[:,1], gt[:,1])
        camera_loss = F.mse_loss(onehot_probs[:,2:4], gt[:,2:4])
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
        

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        if q_value_loss is None and q_diff_loss is None:
            torch.autograd.backward([attack_loss,back_loss,camera_loss,craft_loss,equip_loss,\
                    forward_loss,jump_loss,left_loss,nearby_craft_loss,nearby_smelt_loss,place_loss, \
                    right_loss,sneak_loss,sprint_loss,mh_state_loss,inventory_state_loss, \
                    world_state_loss])
        else:
            torch.autograd.backward([attack_loss,back_loss,camera_loss,craft_loss,equip_loss,\
                    forward_loss,jump_loss,left_loss,nearby_craft_loss,nearby_smelt_loss,place_loss, \
                    right_loss,sneak_loss,sprint_loss,mh_state_loss,inventory_state_loss, \
                    world_state_loss, q_diff_loss, q_value_loss])


        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
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

        return camera_loss, q_value_loss


    def learn_1(self, experiences, gamma):

        ( a_states_mh, a_states_invent, w_states, actions, rewards, a_next_states_mh, a_next_states_invent, w_next_states, dones ) = experiences

        a_states_mh = a_states_mh.to(device)
        a_states_invent = a_states_invent.to(device)
        w_states = w_states.to(device)

        a_next_states_mh = a_next_states_mh.to(device)
        a_next_state_invent = a_next_states_invent.to(device)
        w_next_states = w_next_states.to(device)


        # predict next actions and next next state with actor
        with torch.no_grad():
            _ , _ , _ , Q_next , _ , _ , _ = self.actor_local(a_next_states_mh, w_next_states, a_next_states_invent)
            Q_current_2 = rewards + (gamma * Q_next * (1 - dones))


        #get next state (from experiences) descriptors
        with torch.no_grad():
            n_wsd = self.actor_local.get_wsd(w_next_states)
            n_asmhd = self.actor_local.get_asmhd(a_next_states_mh)
            n_asinventd = self.actor_local.get_asinventoryd(a_next_states_invent)

        # predict actions and next state with actor
        actions_pred, actions_pred_raw, action_logits, Q_current, n_wsd_predict, n_asmhd_predict, n_asinventd_predict = \
                self.actor_local(a_states_mh, w_states, a_states_invent)


        # calculate loss for actor
        loss_1, loss_2 = self.get_action_loss(actions, action_logits, \
                F.mse_loss(n_asmhd, n_asmhd_predict), F.mse_loss(n_asinventd, n_asinventd_predict), \
                F.mse_loss(n_wsd, n_wsd_predict), F.mse_loss(Q_current, Q_current_2.detach()))

        print("Actor Losses:{} {}".format(loss_1.item(), loss_2.item()))
        return loss_1, loss_2


    def learn_2(self, experiences, gamma):
        
        #states, actions, rewards, next_states, dones, indices, weights = experiences
        ( a_states_mh, a_states_invent, w_states, actions, rewards, a_next_states_mh, a_next_states_invent, w_next_states, dones ) = experiences

        a_states_mh = a_states_mh.to(device)
        a_states_invent = a_states_invent.to(device)
        w_states = w_states.to(device)

        a_next_states_mh = a_next_states_mh.to(device)
        a_next_state_invent = a_next_states_invent.to(device)
        w_next_states = w_next_states.to(device)



        #get next state (from experiences) descriptors and Q_next
        with torch.no_grad():
            _, _, _, Q_next, _, _, _, wsd_next, mhd_next, inventd_next = \
                self.actor_local(a_next_states_mh, w_next_states, a_next_states_invent)
            Q_next = Q_next.detach()
            Q_current_2 = rewards + (gamma * Q_next * (1 - dones))
            wsd_next = wsd_next.detach()
            mhd_next = mhd_next.detach()
            inventd_next = inventd_next.detach()



        # predict actions and next state with 
        _, _, action_logits, Q_current, n_wsd_predict, n_asmhd_predict, n_asinventd_predict, _, _, _ = \
                self.actor_local(a_states_mh, w_states, a_states_invent)

        # calculate loss for actor
        loss_1, loss_2 = self.get_action_loss(actions, action_logits, \
                F.mse_loss(mhd_next, n_asmhd_predict), F.mse_loss(inventd_next, n_asinventd_predict), \
                F.mse_loss(wsd_next, n_wsd_predict), F.mse_loss(Q_current, Q_current_2), -Q_current.mean())

        print("Actor Losses:{} {}".format(loss_1.item(), loss_2.item()))
        return loss_1, loss_2


    def learn_3(self, experiences, gamma):
        
        #states, actions, rewards, next_states, dones, indices, weights = experiences
        ( a_states_mh, a_states_invent, w_states, actions, rewards, a_next_states_mh, a_next_states_invent, w_next_states, dones ) = experiences

        a_states_mh = a_states_mh.to(device)
        a_states_invent = a_states_invent.to(device)
        w_states = w_states.to(device)

        a_next_states_mh = a_next_states_mh.to(device)
        a_next_state_invent = a_next_states_invent.to(device)
        w_next_states = w_next_states.to(device)


        # predict actions 
        
        _ , actions_pred_raw, action_logits, _ , _ , _ , _ = \
                self.actor_local(a_states_mh, w_states, a_states_invent)

        #get next state (from experiences) descriptors
        with torch.no_grad():
            n_wsd = self.critic_local.get_wsd(w_next_states)
            n_asmhd = self.critic_local.get_asmhd(a_next_states_mh)
            n_asinventd = self.critic_local.get_asinventoryd(a_next_states_invent)


        # Compute Q value of current state (from experiences)
        Q_current, n_wsd_predict, n_asmhd_predict, n_asinventd_predict = self.critic_local(a_states_mh, a_states_invent, w_states, actions)
        
        # calculate loss for actor/critic
        loss_1, _ = self.get_action_loss(actions, action_logits, \
                F.mse_loss(n_asmhd, n_asmhd_predict), F.mse_loss(n_asinventd, n_asinventd_predict), \
                F.mse_loss(n_wsd, n_wsd_predict))

        


        # Compute Q value of next state (next state from experiences and the rest is predicted with actor and critic

        # predict action in the next state
        actions_next, actions_next_raw, action_logits, _ , _ , _ , _ = self.actor_local(a_next_states_mh, w_next_states, a_next_states_invent)
        # predict Q value in the next state
        Q_next, _ , _ , _ = self.critic_local(a_next_states_mh, a_next_states_invent, w_next_states, actions_next_raw)
        
        
        # Alternative Q value through Bellman equations
        Q_current_2 = rewards + (gamma * Q_next * (1 - dones))


        # Compute critic loss
        critic_loss = F.mse_loss(Q_current.detach(), Q_current_2)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.critic_optimizer.step()


        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)

        print("Actor Losses:{} {}".format(loss_1.item(), critic_loss.item()))

        return loss_1, critic_loss


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

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", 
                        field_names=["state", "state_2", "action", "reward", "next_state", "next_state_2", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, state_2, action, reward, next_state, next_state_2, done):
        """Add a new experience to memory."""
        assert(len(state)==len(next_state))
        assert(len(state_2)==len(next_state_2))
        
        e = self.experience(state, state_2, action, reward, next_state, next_state_2, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        states_2 = torch.from_numpy(np.stack([e.state_2 for e in experiences if e is not None], axis=0)).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        next_states_2 = torch.from_numpy(np.stack([e.next_state_2 for e in experiences if e is not None], axis=0)).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, states_2, actions, rewards, next_states, next_states_2, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    
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
    
    def add(self, state_mh, state_invent, state_2, action, reward, next_state_mh, next_state_invent,  next_state_2, done):

        state_mh      = np.expand_dims(state_mh, 0)
        state_invent  = np.expand_dims(state_invent, 0)
        next_state_mh = np.expand_dims(next_state_mh, 0)
        next_state_invent = np.expand_dims(next_state_invent, 0)
                
        if len(self.memory) < self.capacity:
            print("event memory capacity below:")
            self.memory.append((state_mh, state_invent, state_2, action, reward, next_state_mh, next_state_invent, next_state_2, done))
        else:
            self.memory[self.pos] = (state_mh, state_invent, state_2,  action, reward, next_state_mh, next_state_invent, next_state_2, done)
        
        self.priorities[self.pos] = reward+0.1
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, beta=0.4):
        if len(self.memory) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]
        
        probs  = prios ** self.prob_alpha
        probs  = prios
        probs /= probs.sum()
        
        #indices = np.random.choice(len(self.memory), self.batch_size, p=probs)
        indices = np.random.choice(len(self.memory), self.batch_size, False, p=probs)
        #indices = np.random.randint(0, len(self.memory), self.batch_size)

        
        states_mh = torch.from_numpy(np.vstack([self.memory[idx][0] for idx in indices if indices is not None])).float().to(device)
        states_invent = torch.from_numpy(np.vstack([self.memory[idx][1] for idx in indices if indices is not None])).float().to(device)        
        states_2 = torch.from_numpy(np.stack([self.memory[idx][2] for idx in indices if indices is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[idx][3] for idx in indices if indices is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[idx][4] for idx in indices if indices is not None])).float().to(device)
        next_states_mh = torch.from_numpy(np.vstack([self.memory[idx][5] for idx in indices if indices is not None])).float().to(device)
        next_states_invent = torch.from_numpy(np.vstack([self.memory[idx][6] for idx in indices if indices is not None])).float().to(device)
        next_states_2 = torch.from_numpy(np.stack([self.memory[idx][7] for idx in indices if indices is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[idx][8] for idx in indices if indices is not None]).astype(np.uint8)).float().to(device)
        
        experiences = (states_mh, states_invent, states_2, actions, rewards, next_states_mh, next_states_invent, next_states_2, dones)
        return experiences
    
    def update_priorities(self, batch_indices, batch_priorities):
        
        #print(batch_indices)
        #print(batch_priorities)
        
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)        
