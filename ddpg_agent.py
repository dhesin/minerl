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


BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 16         # minibatch size
GAMMA = 0.5            # discount factor
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

        self.agent_state_size = kwargs['agent_state_size']
        self.world_state_size = kwargs['world_state_size']
        self.action_size = kwargs['action_size']
        self.seed = kwargs['random_seed']
        self.iter = 0
        self.noise_scale = 1.0
        
        # Actor Network (w/ Target Network)
        self.actor_local = Actor(self.agent_state_size, self.world_state_size, self.action_size, self.seed).to(device)
        self.actor_target = Actor(self.agent_state_size, self.world_state_size, self.action_size, self.seed).to(device)
        #self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters())
        self.actor_scheduler = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=200, gamma=0.99)
        
        # Critic Network (w/ Target Network)
        self.critic_local = Critic(self.agent_state_size, self.world_state_size, self.action_size, self.seed).to(device)
        self.critic_target = Critic(self.agent_state_size, self.world_state_size, self.action_size, self.seed).to(device)
        #self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters())
        self.critic_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=200, gamma=0.99)
        
        self.hard_copy_weights(self.actor_target, self.actor_local)
        self.hard_copy_weights(self.critic_target, self.critic_local)

        # reward net
        self.dist_net = torch.nn.Sequential()
        self.dist_net.add_module('norm', torch.nn.LayerNorm(2*self.agent_state_size))
        self.dist_net.add_module('linear1', torch.nn.Linear(2*self.agent_state_size, 50, bias=False))
        self.dist_net.add_module('tanh1', torch.nn.Tanh())
        self.dist_net.add_module('linear2', torch.nn.Linear(50, 1, bias=False))
        self.dist_net.add_module('tanh2', torch.nn.Tanh())
        self.dist_net.to(device)

        for m in self.dist_net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.uniform_(m.weight)


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
        agent_state = []
        agent_state.append(mainhand['damage'])
        agent_state.append(mainhand['maxDamage'])
        agent_state.append(equipments.get(mainhand['type'], -1))
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
                
        agent_state_a = np.array(agent_state)
        world_state_a = np.array(pov)
        world_state_b = np.swapaxes(world_state_a,0,2)
                
        return agent_state_a, world_state_b

        
    def hard_copy_weights(self, target, source):
        """ copy weights from source to target network (part of initialization)"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def step(self, mainhand, inventory, pov, action, reward, mainhand_n, inventory_n, pov_n, done):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        agent_state, world_state = self.get_states(mainhand, inventory, pov)
        agent_state_n, world_state_n = self.get_states(mainhand_n, inventory_n, pov_n)
                
        self.memory.add(agent_state, world_state, action, reward, agent_state_n, world_state_n, done)
        
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

    def add_memory(self, e):
        """Save experience in replay memory, and use random sample from buffer to learn."""
                        
        self.memory.add(e[0], e[1], e[2], e[3], e[4], e[5], e[6])
        

    def learn_from_players(self, experiences):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        
        for e in experiences:
            if (random.random() < 0.1) or e[3] > 0.:
                self.memory.add(e[0], e[1], e[2], e[3], e[4], e[5], e[6])
            
        # Learn, if enough samples are available in memory
        self.iter = self.iter+1
        self.iter = self.iter%1
        if len(self.memory) > BATCH_SIZE:
            
            experiences = self.memory.sample() 
            (states, states_2, actions, rewards, next_states, next_states_2, dones) = experiences           
            self.learn(experiences, GAMMA)
            
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        #self.actor_scheduler.step()
        #self.critic_scheduler.step()

        
        
    def act(self, mainhand, inventory, pov, add_noise=True, noise_scale=1.0):
        """Returns actions for given state as per current policy."""

        agent_state, world_state = self.get_states(mainhand, inventory, pov)        
        
        s1 = torch.from_numpy(agent_state).float().unsqueeze(dim=0).to(device)   
        s2 = torch.from_numpy(world_state).float().unsqueeze(dim=0).to(device) 

        #s1 = torch.from_numpy(agent_state).to(device)   
        #s2 = torch.from_numpy(world_state).to(device) 


        
        self.actor_local.eval()
        with torch.no_grad():
            action, action_raw = self.actor_local(s1,s2)
            
        self.actor_local.train()
        
        return action, action_raw

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #states, actions, rewards, next_states, dones, indices, weights = experiences
        ( a_states_, w_states_, actions, rewards, a_next_states_, w_next_states_, dones ) = experiences
                
        a_states = a_states_.to(device)   
        w_states = w_states_.to(device) 

        a_next_states = a_next_states_.to(device)   
        w_next_states = w_next_states_.to(device) 

        
        #a_state_change = torch.eq(a_states, a_next_states).int()
        
        #ones = torch.ones_like(a_state_change)
        #a_state_change = ones-a_state_change
        #a_state_change = a_state_change.sum(dim=1,keepdim=True).float()/21
        #cos_dis = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        states_concat = torch.cat((a_states, a_next_states), dim=1)

        a_state_change = self.dist_net(states_concat)
        #a_state_change = cos_dis(a_states, a_next_states)
        #a_state_change = a_state_change.unsqueeze(dim=1)
        #normalize = torch.nn.LayerNorm(1).to(device)
        #a_state_change = normalize(a_state_change)
        

        #threshold = torch.nn.Threshold(0, 1000)
        #a_state_change = threshold(a_state_change)



        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next, actions_next_raw = self.actor_target(a_next_states, w_next_states)
        #print(actions_next_raw)        
        Q_targets_next = self.critic_target(a_next_states, w_next_states, actions_next_raw)
        

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones)) + a_state_change 
        Q_expected = self.critic_local(a_states, w_states, actions)
        #print("{} {} \r".format(Q_targets_next, Q_expected))
        #print("{} {}. \r".format(Q_expected.mean().item(), Q_targets.mean().item()))
        
        # Compute critic loss
        #print(a_state_change.shape)
        #print(Q_expected.shape)
        #print(Q_targets.shape)
        critic_loss = F.mse_loss(Q_expected, Q_targets)   
                
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred, actions_pred_raw = self.actor_local(a_states, w_states)
        actor_loss = -self.critic_local(a_states, w_states, actions_pred_raw).mean()
        
        print("{} {} {} \r".format(critic_loss.item(), actor_loss.item(), a_state_change.mean()))
        
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.actor_local.parameters(), 1)
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)                     

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
    
    def add(self, state, state_2, action, reward, next_state, next_state_2, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
                
        if len(self.memory) < self.capacity:
            self.memory.append((state, state_2, action, reward, next_state, next_state_2, done))
        else:
            self.memory[self.pos] = (self, state, state_2, action, reward, next_state, next_state_2, done)
        
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
        
        states = torch.from_numpy(np.vstack([self.memory[idx][0] for idx in indices if indices is not None])).float().to(device)
        states_2 = torch.from_numpy(np.stack([self.memory[idx][1] for idx in indices if indices is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([self.memory[idx][2] for idx in indices if indices is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([self.memory[idx][3] for idx in indices if indices is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([self.memory[idx][4] for idx in indices if indices is not None])).float().to(device)
        next_states_2 = torch.from_numpy(np.stack([self.memory[idx][5] for idx in indices if indices is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([self.memory[idx][6] for idx in indices if indices is not None]).astype(np.uint8)).float().to(device)
    
        
        experiences = (states, states_2, actions, rewards, next_states, next_states_2, dones)
        return experiences
    
    def update_priorities(self, batch_indices, batch_priorities):
        
        #print(batch_indices)
        #print(batch_priorities)
        
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.memory)        
