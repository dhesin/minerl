import numpy as np
import random
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import torchvision.transforms as transforms
from matplotlib.pyplot import imshow
import matplotlib.pyplot as pyplot

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor_TS(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, agent_mh_size, agent_inventory_size, world_state_size,\
            action_size, seed, seq_len, growth_rate=128):
        """Initialize parameters and build model.
        Params
        ======
        """
        
        super(Actor_TS, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.seq_len = seq_len
        

        n_c = world_state_size[0]
        n_d = world_state_size[1]
        i_h = world_state_size[2]
        i_w = world_state_size[3]
         
        # agent's pov
        self.cnn = nn.Sequential()
        self.cnn.add_module('norm1', nn.BatchNorm3d(n_c))
        self.cnn.add_module('conv1', nn.Conv3d(n_c, growth_rate, kernel_size=(1,1,1), stride=1, bias=False))
        self.cnn.add_module('relu1', nn.ReLU(inplace=True))
        self.cnn.add_module('norm2', nn.BatchNorm3d(growth_rate))
        self.cnn.add_module('conv2', nn.Conv3d(growth_rate, growth_rate, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False))
        self.cnn.add_module('relu2', nn.ReLU(inplace=True))
        self.cnn.add_module('norm3', nn.BatchNorm3d(growth_rate))        
        self.cnn.add_module('conv3', nn.Conv3d(growth_rate, growth_rate, kernel_size=(self.seq_len,1,1), stride=1, bias=False))
        self.cnn.add_module('relu3', nn.ReLU(inplace=True))
        self.cnn.add_module('pool1', nn.AvgPool3d(kernel_size=(1,i_h,i_w), stride=(1,1,1)))


        # # agent's pov
        # self.cnn_2 = nn.Sequential()
        # self.cnn_2.add_module('norm1', nn.BatchNorm3d(n_c))
        # self.cnn_2.add_module('relu1', nn.ReLU(inplace=True))
        # self.cnn_2.add_module('conv1', nn.Conv3d(n_c, growth_rate, kernel_size=(1,1,1), stride=1, bias=False))
        # self.cnn_2.add_module('norm2', nn.BatchNorm3d(growth_rate))
        # self.cnn_2.add_module('relu2', nn.ReLU(inplace=True))
        # self.cnn_2.add_module('conv2', nn.Conv3d(growth_rate, growth_rate, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False))
        # self.cnn_2.add_module('pool1', nn.AvgPool3d(kernel_size=(1,i_h,i_w), stride=(1,1,1)))
        # self.cnn_2.add_module('norm3', nn.BatchNorm3d(growth_rate))
        # self.cnn_2.add_module('relu3', nn.ReLU(inplace=True)) 


        self.normalize_inventory = nn.BatchNorm1d(self.seq_len)
        self.inventory_lstm = torch.nn.LSTM(agent_inventory_size, 60, num_layers=2, batch_first=True, bias=False)  
        
        self.normalize_mh = nn.BatchNorm1d(self.seq_len)
        self.mh_lstm = torch.nn.LSTM(agent_mh_size, 20, num_layers=2, batch_first=True, bias=False)  

 
        self.cnn_mh_inventory = nn.Sequential()
        self.cnn_mh_inventory.add_module('norm', nn.LayerNorm(growth_rate+80))
        self.cnn_mh_inventory.add_module('linear1', nn.Linear(growth_rate+80, 100, bias=False))
        self.cnn_mh_inventory.add_module('relu1', nn.ReLU(inplace=True))
        self.cnn_mh_inventory.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.cnn_mh_inventory.add_module('relu2', nn.ReLU(inplace=True))
        #self.cnn_mh_inventory_lstm.register_backward_hook(self.back_hook) 
              
        self.normalize_action_inputs = nn.LayerNorm(20)
        self.output_action_modules = nn.ModuleDict({
            'attack': nn.Linear(20,1, bias=False),
            'back': nn.Linear(20,1, bias=False),
            'camera': nn.Linear(20,2, bias=False),
            'craft': nn.Linear(20,5, bias=False),
            'equip': nn.Linear(20,8, bias=False),
            'forward_': nn.Linear(20,1, bias=False),
            'jump': nn.Linear(20,1, bias=False),
            'left': nn.Linear(20,1, bias=False),
            'nearbyCraft': nn.Linear(20,8, bias=False),
            'nearbySmelt': nn.Linear(20,3, bias=False),
            'place': nn.Linear(20,7, bias=False),
            'right': nn.Linear(20,1, bias=False),
            'sneak': nn.Linear(20,1, bias=False),
            'sprint': nn.Linear(20,1, bias=False)
        })
        #self.action_modules_1["attack"].register_backward_hook(self.back_hook) 

        self.action_modules_1_output_size = {'attack':1, 'back':1, 'camera':2, 'craft':5,\
            'equip':8, 'forward_':1, 'jump':1, 'left':1, 'nearbyCraft':8, 'nearbySmelt':3,\
            'place':7, 'right':1, 'sneak':1, 'sprint':1}       

        self.output_action_activation_modules = nn.ModuleDict({
            'attack': nn.Tanhshrink(),
            'back': nn.Tanhshrink(),
            'camera': nn.Tanhshrink(),
            'craft': nn.Tanhshrink(),
            'equip': nn.Tanhshrink(),
            'forward_': nn.Tanhshrink(),
            'jump': nn.Tanhshrink(),
            'left': nn.Tanhshrink(),
            'nearbyCraft': nn.Tanhshrink(),
            'nearbySmelt': nn.Tanhshrink(),
            'place': nn.Tanhshrink(),
            'right': nn.Tanhshrink(),
            'sneak': nn.Tanhshrink(),
            'sprint': nn.Tanhshrink(),
        })
 
       
        self.normalize_rewards = nn.LayerNorm(self.seq_len)


        self.qvalue = nn.Sequential()
        self.qvalue.add_module('norm', nn.LayerNorm(20+action_size+1))
        self.qvalue.add_module('linear1', nn.Linear(20+action_size+1, 100, bias=False))
        self.qvalue.add_module('relu1', nn.ReLU(inplace=True))
        self.qvalue.add_module('linear2', nn.Linear(100, 1, bias=False))
        self.qvalue.add_module('relu2', nn.ReLU(inplace=True))

        # Noise process
        self.noise = OUNoise(action_size+1, self.seed)

        self.reset_parameters()

    def init_hidden(self, batch_size, num_layers, hidden_size):
        h0 = torch.zeros(num_layers, batch_size, hidden_size)
        c0 = torch.zeros(num_layers, batch_size, hidden_size)
        h0 = torch.nn.init.xavier_uniform_(h0)
        c0 = torch.nn.init.xavier_uniform_(c0)
        return h0.to(device), c0.to(device)

    def back_hook(self, m, i, o):
        print(m)
        print("------------Input Grad------------")

        for grad in i:
            try:
                print(grad)
            except AttributeError: 
                print ("None found for Gradient")

        print("------------Output Grad------------")
        for grad in o:  
            try:
                print(grad)
            except AttributeError: 
                print ("None found for Gradient")
        print("\n")

    def reset_parameters(self):
        
        for m in self.cnn:
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

        for m in self.output_action_modules:
            nn.init.xavier_uniform_(self.output_action_modules[m].weight)
            #nn.init.constant_(self.action_modules_1[m].bias, 0)

        for m in self.qvalue:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

                                                    
    def forward(self, agent_state_mh, world_state, agent_state_inventory):
        """Build an actor (policy) network that maps states -> actions."""

        #print(world_state[0,:,0,:,:].shape)
        #pil_img = transforms.ToPILImage()(world_state[0,:,0,:,:].cpu())
        #imshow(pil_img)
        #pyplot.show

        batch_size = world_state.shape[0]
        num_layers = 2

        x = self.cnn(world_state).squeeze(dim=3).squeeze(dim=3)
        world_state = x.permute(0,2,1)

        agent_state_inventory = self.normalize_inventory(agent_state_inventory)
        h0, c0 = self.init_hidden(batch_size, num_layers, 60)
        agent_state_inventory, (hidden, cell) = self.inventory_lstm(agent_state_inventory, (h0, c0))

        agent_state_mh = self.normalize_mh(agent_state_mh)
        h0, c0 = self.init_hidden(batch_size, num_layers, 20)
        agent_state_mh, (hidden, cell) = self.mh_lstm(agent_state_mh, (h0, c0))

        combined_state = torch.cat([world_state, agent_state_mh[:,-1,:].unsqueeze(dim=1), agent_state_inventory[:,-1,:].unsqueeze(dim=1)], 2)
        combined_state = combined_state.squeeze(dim=1)

        combined_state = self.cnn_mh_inventory(combined_state) 
        combined_state = self.normalize_action_inputs(combined_state)

        actions = {}
        actions_raw = []
        action_logits = []

        
        for action in self.output_action_modules:
            
            out = self.output_action_modules[action](combined_state)
            #out = self.action_modules_1[action](combined_state)
            out = self.output_action_activation_modules[action](out)
            #if not self.output_action_activation_modules.training:
            #    noise = np.random.normal(scale=0.05, size=out.shape)
            #    out = out + torch.tensor(noise).float().to(device)
            
            
            
            if action == "forward_":
                action = "forward"
            
            if action in ["craft", "equip","nearbyCraft","nearbySmelt","place"]:
                action_logits.append(out)
                out = F.softmax(out, dim=1)
                out = out.argmax(keepdim=True, dim=1).float()
            elif action != "camera":
                action_logits.append(out)
                out = torch.sigmoid(out)
                zeros = torch.zeros_like(out)
                ones = torch.ones_like(out)
                out = torch.where(out > 0.5, ones, zeros).float()
            elif action == "camera":
                out = torch.clamp(out, min=-180, max=180)
                out = out.float()
                action_logits.append(out)
                #out /= 180.0
                                       
            actions_raw.append(out)
            
            # action dictionary for environment            
            if action in ["craft", "equip","nearbyCraft","nearbySmelt","place"]:
                actions[action] = out[-1].int().item()
            #elif action == "forward":
            #    actions[action] = 1                               
            elif action != "camera":
                actions[action] = out[-1].int().item()
            elif action == "camera":
                actions[action] = out[-1].tolist()            

      
        actions_raw = torch.cat(actions_raw, axis=1)
        z = torch.cat([combined_state, actions_raw], axis=1)


        q_value = self.qvalue(z)
        q_value = q_value.squeeze()
        
        action_logits = torch.cat(action_logits, axis=1)
        
        return actions, actions_raw, action_logits, q_value


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


