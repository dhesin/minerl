import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, agent_state_size, world_state_size, action_size, seed, growth_rate=128):
        """Initialize parameters and build model.
        Params
        ======
        """
        
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        n_c = world_state_size[2]
        i_h = world_state_size[0]
        i_w = world_state_size[1]
         
        # agent's pov
        self.cnn = nn.Sequential()
        self.cnn.add_module('norm1', nn.BatchNorm2d(n_c))
        self.cnn.add_module('relu1', nn.ReLU(inplace=True))
        self.cnn.add_module('conv1', nn.Conv2d(n_c, growth_rate, kernel_size=1, stride=1, bias=False))
        self.cnn.add_module('norm2', nn.BatchNorm2d(growth_rate))
        self.cnn.add_module('relu2', nn.ReLU(inplace=True))
        self.cnn.add_module('conv2', nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.cnn.add_module('pool1', nn.AvgPool2d(kernel_size=(i_h,i_w), stride=2))
        self.cnn.add_module('norm3', nn.BatchNorm2d(growth_rate))
        self.cnn.add_module('relu3', nn.ReLU(inplace=True))   
        self.fc1 = nn.Linear(growth_rate,20)
        
        # agent's mainhand and inventory
        self.mh_inventory = nn.Sequential()
        self.mh_inventory.add_module('norm', nn.LayerNorm(agent_state_size))
        self.mh_inventory.add_module('linear1', nn.Linear(agent_state_size, 100, bias=False))
        self.mh_inventory.add_module('relu1', nn.ReLU(inplace=True))
        self.mh_inventory.add_module('linear2', nn.Linear(100, 20, bias=False))   
        self.mh_inventory.add_module('relu2', nn.ReLU(inplace=True))

        self.cnn_mh_inventory = nn.Sequential()
        self.cnn_mh_inventory.add_module('norm', nn.LayerNorm(40))
        self.cnn_mh_inventory.add_module('linear1', nn.Linear(40, 100, bias=False))
        self.cnn_mh_inventory.add_module('relu1', nn.ReLU(inplace=True))
        self.cnn_mh_inventory.add_module('linear2', nn.Linear(100, 40, bias=False))
        self.cnn_mh_inventory.add_module('relu2', nn.ReLU(inplace=True))


        self.action_modules = nn.ModuleDict({
            'attack': nn.Linear(40,1, bias=False),
            'back': nn.Linear(40,1, bias=False),
            'camera': nn.Linear(40,2, bias=False),
            'craft': nn.Linear(40,5, bias=False),
            'equip': nn.Linear(40,8, bias=False),
            'forward_': nn.Linear(40,1, bias=False),
            'jump': nn.Linear(40,1, bias=False),
            'left': nn.Linear(40,1, bias=False),
            'nearbyCraft': nn.Linear(40,8, bias=False),
            'nearbySmelt': nn.Linear(40,3, bias=False),
            'place': nn.Linear(40,7, bias=False),
            'right': nn.Linear(40,1, bias=False),
            'sneak': nn.Linear(40,1, bias=False),
            'sprint': nn.Linear(40,1, bias=False)
                                            })
 
        self.activation_modules = nn.ModuleDict({
            'attack': nn.Tanh(),
            'back': nn.Tanh(),
            'camera': nn.Identity(),
            'craft': nn.Tanh(),
            'equip': nn.Tanh(),
            'forward_': nn.Tanh(),
            'jump': nn.Tanh(),
            'left': nn.Tanh(),
            'nearbyCraft': nn.Tanh(),
            'nearbySmelt': nn.Tanh(),
            'place': nn.Tanh(),
            'right': nn.Tanh(),
            'sneak': nn.Tanh(),
            'sprint': nn.Tanh(),
        })
    
        self.reset_parameters()

    def reset_parameters(self):
        
        for m in self.cnn:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            
        for m in self.mh_inventory:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
 
        for m in self.cnn_mh_inventory:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

            
        for m in self.action_modules:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                
                                                    
    def forward(self, agent_state, world_state):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.cnn(world_state).squeeze(dim=2).squeeze(dim=2)
        x = self.fc1(x)
        x = F.relu(x)


        y = self.mh_inventory(agent_state)


        z = torch.cat([x, y], 1)
        z = self.cnn_mh_inventory(z)
        
        actions = {}
        actions_raw = []
        
        for action in self.action_modules:
            
            out = self.action_modules[action](z)
            out = self.activation_modules[action](out)
            
            if action == "forward_":
                action = "forward"
            
            if action in ["craft", "equip","nearbyCraft","nearbySmelt","place"]:
                out = F.softmax(out, dim=1)
                out = out.argmax(dim=1, keepdim=True).float()
            elif action != "camera":
                zeros = torch.zeros_like(out)
                ones = torch.ones_like(out)
                out = torch.where(out > 0., ones, zeros).squeeze(dim=0).float()
            elif action == "camera":
                out = torch.clamp(out, min=-180, max=180)
                out = out.float()
                                
                
            # raw action tensor for processing    
            if (len(out.shape)) is 1:
                out = out.unsqueeze(dim=0)
            actions_raw.append(out)
            

            # action dictionary for environment            
            if action in ["craft", "equip","nearbyCraft","nearbySmelt","place"]:
                actions[action] = out[0].int().item()
            #elif action == "forward":
            #    actions[action] = 1                               
            elif action != "camera":
                actions[action] = out[0].int().item()
            elif action == "camera":
                actions[action] = out[0].tolist()

            


        
        actions_raw = torch.cat(actions_raw, dim=1)
        
        
        return actions, actions_raw


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, agent_state_size, world_state_size, action_size, seed, growth_rate=128):
        """Initialize parameters and build model.
        Params
        ======
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        n_c = world_state_size[2]
        i_h = world_state_size[0]
        i_w = world_state_size[1]
        
        self.previous_agent_state = None

        # agent's pov
        self.cnn = nn.Sequential()
        self.cnn.add_module('norm1', nn.BatchNorm2d(n_c))
        self.cnn.add_module('relu1', nn.ReLU(inplace=True))
        self.cnn.add_module('conv1', nn.Conv2d(3, growth_rate, kernel_size=1, stride=1, bias=False))
        self.cnn.add_module('norm2', nn.BatchNorm2d(growth_rate))
        self.cnn.add_module('relu2', nn.ReLU(inplace=True))
        self.cnn.add_module('conv2', nn.Conv2d(growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.cnn.add_module('pool1', nn.AvgPool2d(kernel_size=(i_h,i_w), stride=2))
        self.cnn.add_module('norm3', nn.BatchNorm2d(growth_rate))
        self.cnn.add_module('relu3', nn.ReLU(inplace=True))   
        self.fc1 = nn.Linear(growth_rate,20)
        #self.cnn.add_module('linear1', nn.Linear(growth_rate, 20, bias=False))
        #self.cnn.add_module('relu4', nn.ReLU(inplace=True))   
       
        # agent's mainhand and inventory
        self.mh_inventory = nn.Sequential()
        self.mh_inventory.add_module('norm', nn.LayerNorm(agent_state_size))
        self.mh_inventory.add_module('linear1', nn.Linear(agent_state_size, 100, bias=False))
        self.mh_inventory.add_module('relu1', nn.ReLU(inplace=True))
        self.mh_inventory.add_module('linear2', nn.Linear(100, 20, bias=False))    
        self.mh_inventory.add_module('relu2', nn.ReLU(inplace=True))
                         

        # agent's mainhand, inventory, pov, action combined
        self.action_modules = nn.ModuleDict({
            'attack': nn.Identity(),
            'back': nn.Identity(),
            'camera_yaw': nn.Identity(),
            'camera_pitch': nn.Identity(),
            'craft': nn.Embedding(5, 3),
            'equip': nn.Embedding(8, 3),
            'forward_': nn.Identity(),
            'jump': nn.Identity(),
            'left': nn.Identity(),
            'nearbyCraft': nn.Embedding(8, 3),
            'nearbySmelt': nn.Embedding(3, 2),
            'place': nn.Embedding(7, 3),
            'right': nn.Identity(),
            'sneak': nn.Identity(),
            'sprint': nn.Identity()
        })
 

        # agent's actions
        self.combined_actions = nn.Sequential()
        self.combined_actions.add_module('norm', nn.LayerNorm(24))
        self.combined_actions.add_module('linear1', nn.Linear(24, 100))
        self.combined_actions.add_module('relu1', nn.ReLU(inplace=True))
        self.combined_actions.add_module('linear2', nn.Linear(100, 20))    
        self.combined_actions.add_module('relu2', nn.ReLU(inplace=True))

        # agent's actions
        self.combined = nn.Sequential()
        self.combined.add_module('norm', nn.LayerNorm(60))
        self.combined.add_module('linear1', nn.Linear(60, 1, bias=False))
        self.combined.add_module('relu1', nn.Tanh())
        #self.combined.add_module('linear2', nn.Linear(25, 1, bias=False))    
        #self.combined.add_module('relu2', nn.ReLU(inplace=True))
    

        # reward predictor net
        self.dist_net = torch.nn.Sequential()
        self.dist_net.add_module('norm', torch.nn.LayerNorm(2*agent_state_size))
        self.dist_net.add_module('linear1', torch.nn.Linear(2*agent_state_size, 50, bias=False))
        self.dist_net.add_module('tanh1', torch.nn.Tanh())
        self.dist_net.add_module('linear2', torch.nn.Linear(50, 1, bias=False))
        self.dist_net.add_module('tanh2', torch.nn.Tanh())

        self.reset_parameters()

    def reset_parameters(self):
        
        for m in self.cnn:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        for m in self.mh_inventory:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        for m in self.action_modules:
            if isinstance(self.action_modules[m], nn.Embedding):
                torch.nn.init.xavier_normal_(self.action_modules[m].weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(self.action_modules[m].weight, gain=nn.init.calculate_gain('relu'))

                
        for m in self.combined_actions:
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.normal_(m.bias)

                
        for m in self.combined:
            if isinstance(m, nn.Conv2d):
                nn.init.uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))

        for m in self.dist_net:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain('tanh'))
       

    def forward(self, agent_state, world_state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
                
        x = self.cnn(world_state).squeeze(dim=2).squeeze(dim=2)
        x = self.fc1(x)
        x = F.relu(x)

        y = self.mh_inventory(agent_state)
           
        actions = []
        for i, a in enumerate(self.action_modules):
            if a in ["craft", "equip", "nearbyCraft", "nearbySmelt", "place"]:
                out = self.action_modules[a](action[:,i].long())
            else:
                out = self.action_modules[a](action[:,i])

            if (len(out.shape)) is 1:
                out = out.unsqueeze(dim=0).transpose(0,1)

            
            actions.append(out)

        actions_embedded_combined = torch.cat((actions), 1)   
        z = self.combined_actions(actions_embedded_combined)


        c = torch.cat([x,y,z], 1)
        c = self.combined(c)
 
        # curiosity reward
        if self.previous_agent_state is None:
            self.previous_agent_state = torch.zeros_like(agent_state)

        states_concat = torch.cat((self.previous_agent_state, agent_state), dim=1)
        a_state_change_reward = self.dist_net(states_concat)
        self.previous_agent_state = agent_state

        return c, a_state_change_reward



