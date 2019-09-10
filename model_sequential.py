import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms as transforms
from matplotlib.pyplot import imshow
import matplotlib.pyplot as pyplot

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor_TS(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, agent_mh_size, agent_inventory_size, world_state_size, action_size, seed, growth_rate=128):
        """Initialize parameters and build model.
        Params
        ======
        """
        
        super(Actor_TS, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        n_c = world_state_size[0]
        n_d = world_state_size[1]
        i_h = world_state_size[2]
        i_w = world_state_size[3]
         
        # agent's pov
        self.cnn = nn.Sequential()
        self.cnn.add_module('norm1', nn.BatchNorm3d(n_c))
        self.cnn.add_module('relu1', nn.ReLU(inplace=True))
        self.cnn.add_module('conv1', nn.Conv3d(n_c, growth_rate, kernel_size=1, stride=1, bias=False))
        self.cnn.add_module('norm2', nn.BatchNorm3d(growth_rate))
        self.cnn.add_module('relu2', nn.ReLU(inplace=True))
        self.cnn.add_module('conv2', nn.Conv3d(growth_rate, growth_rate, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False))
        self.cnn.add_module('pool1', nn.AvgPool3d(kernel_size=(1,i_h,i_w), stride=(1,1,1)))
        self.cnn.add_module('norm3', nn.BatchNorm3d(growth_rate))
        self.cnn.add_module('relu3', nn.ReLU(inplace=True)) 

        self.pov_lstm = torch.nn.LSTM(growth_rate, 20, num_layers=2, batch_first=True, bias=False)  

        self.inventory_lstm = torch.nn.LSTM(agent_inventory_size, 20, num_layers=2, batch_first=True, bias=False)  

        self.mh_lstm = torch.nn.LSTM(agent_mh_size, 20, num_layers=2, batch_first=True, bias=False)  

 
        self.cnn_mh_inventory = torch.nn.LSTM(60, 40, num_layers=2, batch_first=True, bias=False)  
              
        
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
        self.next_state_predict_cnn.add_module('norm', nn.LayerNorm(40+action_size+1))
        self.next_state_predict_cnn.add_module('linear1', nn.Linear(40+action_size+1, 100, bias=False))
        self.next_state_predict_cnn.add_module('relu1', nn.ReLU(inplace=True))
        self.next_state_predict_cnn.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.next_state_predict_cnn.add_module('relu2', nn.ReLU(inplace=True))
 
        self.next_state_predict_agent_mh = nn.Sequential()
        self.next_state_predict_agent_mh.add_module('norm', nn.LayerNorm(40+action_size+1))
        self.next_state_predict_agent_mh.add_module('linear1', nn.Linear(40+action_size+1, 100, bias=False))
        self.next_state_predict_agent_mh.add_module('relu1', nn.ReLU(inplace=True))
        self.next_state_predict_agent_mh.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.next_state_predict_agent_mh.add_module('relu2', nn.ReLU(inplace=True))

        self.next_state_predict_agent_inventory = nn.Sequential()
        self.next_state_predict_agent_inventory.add_module('norm', nn.LayerNorm(40+action_size+1))
        self.next_state_predict_agent_inventory.add_module('linear1', nn.Linear(40+action_size+1, 100, bias=False))
        self.next_state_predict_agent_inventory.add_module('relu1', nn.ReLU(inplace=True))
        self.next_state_predict_agent_inventory.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.next_state_predict_agent_inventory.add_module('relu2', nn.ReLU(inplace=True))
       
        self.qvalue = nn.Sequential()
        self.qvalue.add_module('norm', nn.LayerNorm(40+action_size+1))
        self.qvalue.add_module('linear1', nn.Linear(40+action_size+1, 100, bias=False))
        self.qvalue.add_module('relu1', nn.ReLU(inplace=True))
        self.qvalue.add_module('linear2', nn.Linear(100, 1, bias=False))
        self.qvalue.add_module('relu2', nn.ReLU(inplace=True))


        self.reset_parameters()

    def reset_parameters(self):
        
        for m in self.cnn:
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

        for m in self.qvalue:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

                                                    
    def forward(self, agent_state_mh, world_state, agent_state_inventory):
        """Build an actor (policy) network that maps states -> actions."""

        #print(world_state[0,:,0,:,:].shape)
        pil_img = transforms.ToPILImage()(world_state[0,:,0,:,:])
        imshow(pil_img)
        pyplot.show()

        x = self.cnn(world_state).squeeze(dim=3).squeeze(dim=3)
        x = x.permute(0,2,1)
        world_state, (hidden, cell) = self.pov_lstm(x)

        agent_state_inventory, (hidden, cell) = self.inventory_lstm(agent_state_inventory)

        agent_state_mh, (hidden, cell) = self.mh_lstm(agent_state_mh)


        combined_state = torch.cat([world_state, agent_state_mh, agent_state_inventory], 2)
        combined_state, (hidden, cell) = self.cnn_mh_inventory(combined_state) 


        actions = {}
        actions_raw = []
        action_logits = []
        
        for action in self.action_modules:
            
            out = self.action_modules[action](combined_state[:,-1,:])
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
        z = torch.cat([combined_state[:,-1,:], actions_raw], 1)
        n_wsd_predict = self.next_state_predict_cnn(z)
        n_asmhd_predict = self.next_state_predict_agent_mh(z)
        n_asinventoryd_predict = self.next_state_predict_agent_inventory(z)
        q_value = self.qvalue(z)
        
        action_logits = torch.cat(action_logits, dim=1)

        
        return actions, actions_raw, action_logits, q_value, n_wsd_predict, n_asmhd_predict, \
            n_asinventoryd_predict, world_state[:,-1,:], agent_state_mh[:,-1,:], agent_state_inventory[:,-1,:]

    def get_wsd(self, world_state):

        x = self.cnn(world_state).squeeze(dim=2).squeeze(dim=2[:,-1,:])
        x = self.fc1(x)
        wsd = F.relu(x)  # world state descriptor
        return wsd

    def get_asmhd(self, agent_state_mh):
        asmhd = self.mh(agent_state_mh) # agent state descriptor
        return asmhd

    def get_asinventoryd(self, agent_state_inventory):
        asinventoryd = self.inventory(agent_state_inventory) # agent state descriptor
        return asinventoryd



class Critic_TS(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, agent_mh_size, agent_invent_size, world_state_size, action_size, seed, growth_rate=128):
        """Initialize parameters and build model.
        Params
        ======
        """

        super(Critic_TS, self).__init__()
        self.seed = torch.manual_seed(seed)

        n_c = world_state_size[2]
        i_h = world_state_size[0]
        i_w = world_state_size[1]
        agent_state_size = agent_mh_size+agent_invent_size
        
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



