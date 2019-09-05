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

    def __init__(self, agent_mh_size, agent_inventory_size, world_state_size, action_size, seed, growth_rate=128):
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


        # agent's pov after flattening 2D to 1D
        self.pov_flat = nn.Sequential()
        self.pov_flat.add_module('norm', nn.BatchNorm1d(n_c*i_h*i_w))
        self.pov_flat.add_module('linear1', nn.Linear(n_c*i_h*i_w, 1000, bias=False))
        self.pov_flat.add_module('relu1', nn.ReLU(inplace=True))
        self.pov_flat.add_module('linear2', nn.Linear(1000, 200, bias=False))   
        self.pov_flat.add_module('relu2', nn.ReLU(inplace=True))
        self.pov_flat.add_module('linear3', nn.Linear(200, 20, bias=False))   
        self.pov_flat.add_module('relu3', nn.ReLU(inplace=True))

        
        # agent's mainhand
        self.mh = nn.Sequential()
        self.mh.add_module('norm', nn.BatchNorm1d(agent_mh_size))
        self.mh.add_module('linear1', nn.Linear(agent_mh_size, 100, bias=False))
        self.mh.add_module('relu1', nn.ReLU(inplace=True))
        self.mh.add_module('linear2', nn.Linear(100, 20, bias=False))   
        self.mh.add_module('relu2', nn.ReLU(inplace=True))

        
        # agent'inventory
        self.inventory = nn.Sequential()
        self.inventory.add_module('norm', nn.BatchNorm1d(agent_inventory_size))
        self.inventory.add_module('linear1', nn.Linear(agent_inventory_size, 100, bias=False))
        self.inventory.add_module('relu1', nn.ReLU(inplace=True))
        self.inventory.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.inventory.add_module('relu2', nn.ReLU(inplace=True))
       
        
        self.cnn_mh_inventory = nn.Sequential()
        self.cnn_mh_inventory.add_module('norm', nn.BatchNorm1d(80))
        self.cnn_mh_inventory.add_module('linear1', nn.Linear(80, 100, bias=False))
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
            'attack': nn.Identity(),
            'back': nn.Identity(),
            'camera': nn.Tanhshrink(),
            'craft': nn.Identity(),
            'equip': nn.Identity(),
            'forward_': nn.Identity(),
            'jump': nn.Identity(),
            'left': nn.Identity(),
            'nearbyCraft': nn.Identity(),
            'nearbySmelt': nn.Identity(),
            'place': nn.Identity(),
            'right': nn.Identity(),
            'sneak': nn.Identity(),
            'sprint': nn.Identity(),
        })

        self.next_state_predict_cnn = nn.Sequential()
        self.next_state_predict_cnn.add_module('norm', nn.BatchNorm1d(40+action_size+1))
        self.next_state_predict_cnn.add_module('linear1', nn.Linear(40+action_size+1, 100, bias=False))
        self.next_state_predict_cnn.add_module('relu1', nn.ReLU(inplace=True))
        self.next_state_predict_cnn.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.next_state_predict_cnn.add_module('relu2', nn.ReLU(inplace=True))
 
        self.next_state_predict_agent_mh = nn.Sequential()
        self.next_state_predict_agent_mh.add_module('norm', nn.BatchNorm1d(40+action_size+1))
        self.next_state_predict_agent_mh.add_module('linear1', nn.Linear(40+action_size+1, 100, bias=False))
        self.next_state_predict_agent_mh.add_module('relu1', nn.ReLU(inplace=True))
        self.next_state_predict_agent_mh.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.next_state_predict_agent_mh.add_module('relu2', nn.ReLU(inplace=True))

        self.next_state_predict_agent_inventory = nn.Sequential()
        self.next_state_predict_agent_inventory.add_module('norm', nn.BatchNorm1d(40+action_size+1))
        self.next_state_predict_agent_inventory.add_module('linear1', nn.Linear(40+action_size+1, 100, bias=False))
        self.next_state_predict_agent_inventory.add_module('relu1', nn.ReLU(inplace=True))
        self.next_state_predict_agent_inventory.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.next_state_predict_agent_inventory.add_module('relu2', nn.ReLU(inplace=True))
       
        self.predict_qvalue = nn.Sequential()
        self.predict_qvalue.add_module('norm', nn.BatchNorm1d(40+action_size+1))
        self.predict_qvalue.add_module('linear1', nn.Linear(40+action_size+1, 100, bias=False))
        self.predict_qvalue.add_module('relu1', nn.ReLU(inplace=True))
        self.predict_qvalue.add_module('linear2', nn.Linear(100, 50, bias=False))
        self.predict_qvalue.add_module('relu2', nn.ReLU(inplace=True))
        self.predict_qvalue.add_module('linear3', nn.Linear(50, 1, bias=False))
        self.predict_qvalue.add_module('relu3', nn.ReLU(inplace=True))

        self.reset_parameters()

    def reset_parameters(self):
        
        for m in self.cnn:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            
        for m in self.pov_flat:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        for m in self.mh:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        for m in self.inventory:
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
                nn.init.xavier_uniform_(m.weight)
                
        for m in self.next_state_predict_cnn:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
 
        for m in self.next_state_predict_agent_mh:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        for m in self.next_state_predict_agent_inventory:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        for m in self.predict_qvalue:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
                                                    
    def forward(self, agent_state_mh, world_state, agent_state_inventory):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.cnn(world_state).squeeze(dim=2).squeeze(dim=2)
        x = self.fc1(x)
        wsd = F.relu(x)  # world state descriptor

        x = torch.flatten(world_state, start_dim=1)
        pov_flat = self.pov_flat(x)

        asmhd = self.mh(agent_state_mh) # agent state mainhand descriptor
        asinventoryd = self.inventory(agent_state_inventory) # agent state inventory descriptor

        z = torch.cat([wsd, asmhd, asinventoryd, pov_flat], 1)
        combined_state = self.cnn_mh_inventory(z)
        self.combined_state = combined_state.detach()
        
        actions = {}
        actions_raw = []
        action_logits = []
        
        for action in self.action_modules:
            
            out = self.action_modules[action](combined_state)
            out = self.activation_modules[action](out)
            
            if action == "forward_":
                action = "forward"
            
            if action in ["craft", "equip","nearbyCraft","nearbySmelt","place"]:
                action_logits.append(out)
                out = F.softmax(out, dim=1)
                out = out.argmax(dim=1, keepdim=True).float()
            elif action != "camera":
                action_logits.append(out)
                out = torch.sigmoid(out)
                zeros = torch.zeros_like(out)
                ones = torch.ones_like(out)
                out = torch.where(out > 0.5, ones, zeros).squeeze(dim=0).float()
            elif action == "camera":
                out = torch.clamp(out, min=-180, max=180)
                out = out.float()
                #out[0][0] = out[0][0]/100.0
                #out[0][1] = out[0][1]
                action_logits.append(out)
                                
                
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
        self.actions_raw = actions_raw.detach()
        #print(actions_raw)
        action_logits = torch.cat(action_logits, dim=1)
        #print(action_probs)

        z = torch.cat([combined_state, actions_raw], 1)
        n_wsd_predict = self.next_state_predict_cnn(z)

        z = torch.cat([combined_state, actions_raw], 1)
        n_asmhd_predict = self.next_state_predict_agent_mh(z)

        z = torch.cat([combined_state, actions_raw], 1)
        n_asinventoryd_predict = self.next_state_predict_agent_inventory(z)

        z = torch.cat([combined_state, self.actions_raw], 1)
        q_current = self.predict_qvalue(z)

        
        return actions, actions_raw, action_logits, q_current, n_wsd_predict, n_asmhd_predict, n_asinventoryd_predict, wsd, asmhd, asinventoryd

    def get_wsd(self, world_state):

        x = self.cnn(world_state).squeeze(dim=2).squeeze(dim=2)
        x = self.fc1(x)
        wsd = F.relu(x)  # world state descriptor
        return wsd

    def get_asmhd(self, agent_state_mh):
        asmhd = self.mh(agent_state_mh) # agent state descriptor
        return asmhd

    def get_asinventoryd(self, agent_state_inventory):
        asinventoryd = self.inventory(agent_state_inventory) # agent state descriptor
        return asinventoryd

    def get_qvalue(self):
        q_predict = torch.cat([self.combined_state, self.actions_raw], 1)
        q_predict = self.predict_qvalue(q_predict)
        return q_predict



class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, agent_mh_size, agent_inventory_size, world_state_size, action_size, seed, growth_rate=128):
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


        # agent's pov after flattening 2D to 1D
        self.pov_flat = nn.Sequential()
        self.pov_flat.add_module('norm', nn.BatchNorm1d(n_c*i_h*i_w))
        self.pov_flat.add_module('linear1', nn.Linear(n_c*i_h*i_w, 1000, bias=False))
        self.pov_flat.add_module('relu1', nn.ReLU(inplace=True))
        self.pov_flat.add_module('linear2', nn.Linear(1000, 200, bias=False))   
        self.pov_flat.add_module('relu2', nn.ReLU(inplace=True))
        self.pov_flat.add_module('linear3', nn.Linear(200, 20, bias=False))   
        self.pov_flat.add_module('relu3', nn.ReLU(inplace=True))
       
        # agent's mainhand
        self.mh = nn.Sequential()
        self.mh.add_module('norm', nn.BatchNorm1d(agent_mh_size))
        self.mh.add_module('linear1', nn.Linear(agent_mh_size, 100, bias=False))
        self.mh.add_module('relu1', nn.ReLU(inplace=True))
        self.mh.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.mh.add_module('relu2', nn.ReLU(inplace=True))

       
        # agent's inventory
        self.inventory = nn.Sequential()
        self.inventory.add_module('norm', nn.BatchNorm1d(agent_inventory_size))
        self.inventory.add_module('linear1', nn.Linear(agent_inventory_size, 100, bias=False))
        self.inventory.add_module('relu1', nn.ReLU(inplace=True))
        self.inventory.add_module('linear2', nn.Linear(100, 20, bias=False))    
        self.inventory.add_module('relu2', nn.ReLU(inplace=True))
                         

        # agent's action / some are embedded
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
 

        # agent's actions combined
        self.combined_actions = nn.Sequential()
        self.combined_actions.add_module('norm', nn.BatchNorm1d(24))
        self.combined_actions.add_module('linear1', nn.Linear(24, 100))
        self.combined_actions.add_module('relu1', nn.ReLU(inplace=True))
        self.combined_actions.add_module('linear2', nn.Linear(100, 20))    
        self.combined_actions.add_module('relu2', nn.ReLU(inplace=True))

        # agent's actions/pov/pov flat/mh/inventory combined - Q value estimation
        self.combined = nn.Sequential()
        self.combined.add_module('norm', nn.BatchNorm1d(100))
        self.combined.add_module('linear1', nn.Linear(100, 25, bias=False))
        self.combined.add_module('relu1', nn.ReLU(inplace=True))
        self.combined.add_module('linear2', nn.Linear(25, 1, bias=False))    
        self.combined.add_module('relu2', nn.Tanh())
    
        self.next_state_predict_cnn = nn.Sequential()
        self.next_state_predict_cnn.add_module('norm', nn.BatchNorm1d(100))
        self.next_state_predict_cnn.add_module('linear1', nn.Linear(100, 100, bias=False))
        self.next_state_predict_cnn.add_module('relu1', nn.ReLU(inplace=True))
        self.next_state_predict_cnn.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.next_state_predict_cnn.add_module('relu2', nn.ReLU(inplace=True))
 
        self.next_state_predict_agent_mh = nn.Sequential()
        self.next_state_predict_agent_mh.add_module('norm', nn.BatchNorm1d(100))
        self.next_state_predict_agent_mh.add_module('linear1', nn.Linear(100, 100, bias=False))
        self.next_state_predict_agent_mh.add_module('relu1', nn.ReLU(inplace=True))
        self.next_state_predict_agent_mh.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.next_state_predict_agent_mh.add_module('relu2', nn.ReLU(inplace=True))

        self.next_state_predict_agent_inventory = nn.Sequential()
        self.next_state_predict_agent_inventory.add_module('norm', nn.BatchNorm1d(100))
        self.next_state_predict_agent_inventory.add_module('linear1', nn.Linear(100, 100, bias=False))
        self.next_state_predict_agent_inventory.add_module('relu1', nn.ReLU(inplace=True))
        self.next_state_predict_agent_inventory.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.next_state_predict_agent_inventory.add_module('relu2', nn.ReLU(inplace=True))
        self.reset_parameters()

    def reset_parameters(self):
        
        for m in self.cnn:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        for m in self.pov_flat:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        for m in self.mh:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        for m in self.inventory:
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

        for m in self.next_state_predict_cnn:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
 
        for m in self.next_state_predict_agent_mh:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        for m in self.next_state_predict_agent_inventory:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))


    def forward(self, agent_mh_state, agent_inventory_state, world_state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
                
        x = self.cnn(world_state).squeeze(dim=2).squeeze(dim=2)
        x = self.fc1(x)
        pov_descriptor = F.relu(x)

        x = torch.flatten(world_state, start_dim=1)
        pov_flat = self.pov_flat(x)


        mh_descriptor = self.mh(agent_mh_state)
        inventory_descriptor = self.inventory(agent_inventory_state)
           
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


        combined_state = torch.cat([pov_descriptor, mh_descriptor, inventory_descriptor, z, pov_flat], 1)
        q_value = self.combined(combined_state)
 
        n_wsd_predict = self.next_state_predict_cnn(combined_state)

        n_asmhd_predict = self.next_state_predict_agent_mh(combined_state)

        n_asinventoryd_predict = self.next_state_predict_agent_inventory(combined_state)


        return q_value, n_wsd_predict, n_asmhd_predict, n_asinventoryd_predict

    def get_wsd(self, world_state):

        x = self.cnn(world_state).squeeze(dim=2).squeeze(dim=2)
        x = self.fc1(x)
        wsd = F.relu(x)  # world state descriptor
        return wsd

    def get_asmhd(self, agent_state_mh):
        asmhd = self.mh(agent_state_mh) # agent state descriptor
        return asmhd

    def get_asinventoryd(self, agent_state_inventory):
        asinventoryd = self.inventory(agent_state_inventory) # agent state descriptor
        return asinventoryd

