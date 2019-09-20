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
        self.cnn.add_module('relu1', nn.ReLU(inplace=True))
        self.cnn.add_module('conv1', nn.Conv3d(n_c, growth_rate, kernel_size=(seq_len,1,1), stride=1, bias=False))
        self.cnn.add_module('norm2', nn.BatchNorm3d(growth_rate))
        self.cnn.add_module('relu2', nn.ReLU(inplace=True))
        self.cnn.add_module('conv2', nn.Conv3d(growth_rate, growth_rate, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False))
        self.cnn.add_module('pool1', nn.AvgPool3d(kernel_size=(1,i_h,i_w), stride=(1,1,1)))
        self.cnn.add_module('norm3', nn.BatchNorm3d(growth_rate))
        self.cnn.add_module('relu3', nn.ReLU(inplace=True)) 

        self.pov_lstm = torch.nn.LSTM(growth_rate, 60, num_layers=2, batch_first=True, bias=False)  

        self.normalize_inventory = nn.BatchNorm1d(self.seq_len)
        self.inventory_lstm = torch.nn.LSTM(agent_inventory_size, 60, num_layers=2, batch_first=True, bias=False)  
        
        self.normalize_mh = nn.BatchNorm1d(self.seq_len)
        self.mh_lstm = torch.nn.LSTM(agent_mh_size, 20, num_layers=2, batch_first=True, bias=False)  

 
        self.cnn_mh_inventory_lstm = torch.nn.LSTM(140, 80, num_layers=2, batch_first=True, bias=False) 
        #self.cnn_mh_inventory_lstm.register_backward_hook(self.back_hook) 
              
        self.normalize_action_inputs = nn.LayerNorm(80)
        self.action_modules_lstm = nn.ModuleDict({
            'attack': nn.LSTM(80,1, num_layers=2, batch_first=True, bias=True),
            'back': nn.LSTM(80,1, num_layers=2, batch_first=True, bias=True),
            'camera': nn.LSTM(80,2, num_layers=2, batch_first=True, bias=True),
            'craft': nn.LSTM(80,5, num_layers=2, batch_first=True, bias=True),
            'equip': nn.LSTM(80,8, num_layers=2, batch_first=True, bias=True),
            'forward_': nn.LSTM(80,1, num_layers=2, batch_first=True, bias=True),
            'jump': nn.LSTM(80,1, num_layers=2, batch_first=True, bias=True),
            'left': nn.LSTM(80,1, num_layers=2, batch_first=True, bias=True),
            'nearbyCraft': nn.LSTM(80,8, num_layers=2, batch_first=True, bias=True),
            'nearbySmelt': nn.LSTM(80,3, num_layers=2, batch_first=True, bias=True),
            'place': nn.LSTM(80,7, num_layers=2, batch_first=True, bias=True),
            'right': nn.LSTM(80,1, num_layers=2, batch_first=True, bias=True),
            'sneak': nn.LSTM(80,1, num_layers=2, batch_first=True, bias=True),
            'sprint': nn.LSTM(80,1, num_layers=2, batch_first=True, bias=True)
        })

 
        self.action_modules_1 = nn.ModuleDict({
            'attack': nn.Linear(80,1, bias=False),
            'back': nn.Linear(80,1, bias=True),
            'camera': nn.Linear(80,2, bias=True),
            'craft': nn.Linear(80,5, bias=True),
            'equip': nn.Linear(80,8, bias=True),
            'forward_': nn.Linear(80,1, bias=False),
            'jump': nn.Linear(80,1, bias=True),
            'left': nn.Linear(80,1, bias=True),
            'nearbyCraft': nn.Linear(80,8, bias=True),
            'nearbySmelt': nn.Linear(80,3, bias=True),
            'place': nn.Linear(80,7, bias=True),
            'right': nn.Linear(80,1, bias=True),
            'sneak': nn.Linear(80,1, bias=True),
            'sprint': nn.Linear(80,1, bias=True)
        })
        #self.action_modules_1["attack"].register_backward_hook(self.back_hook) 

        self.action_modules_1_output_size = {'attack':1, 'back':1, 'camera':2, 'craft':5,\
            'equip':8, 'forward_':1, 'jump':1, 'left':1, 'nearbyCraft':8, 'nearbySmelt':3,\
            'place':7, 'right':1, 'sneak':1, 'sprint':1}       

        self.activation_modules = nn.ModuleDict({
            'attack': nn.Tanh(),
            'back': nn.ReLU(),
            'camera': nn.Tanhshrink(),
            'craft': nn.ReLU(),
            'equip': nn.ReLU(),
            'forward_': nn.Tanh(),
            'jump': nn.ReLU(),
            'left': nn.ReLU(),
            'nearbyCraft': nn.ReLU(),
            'nearbySmelt': nn.ReLU(),
            'place': nn.ReLU(),
            'right': nn.ReLU(),
            'sneak': nn.ReLU(),
            'sprint': nn.ReLU(),
        })
 
        self.next_state_predict_cnn_lstm = torch.nn.LSTM(80+action_size+1, growth_rate, num_layers=2, batch_first=True, bias=False)  
        self.next_state_predict_cnn = nn.Sequential()
        self.next_state_predict_cnn.add_module('norm', nn.LayerNorm(80+action_size+1))
        self.next_state_predict_cnn.add_module('linear1', nn.Linear(80+action_size+1, 100, bias=False))
        self.next_state_predict_cnn.add_module('relu1', nn.ReLU(inplace=True))
        self.next_state_predict_cnn.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.next_state_predict_cnn.add_module('relu2', nn.ReLU(inplace=True))
 
        self.next_state_predict_agent_mh = nn.Sequential()
        self.next_state_predict_agent_mh.add_module('norm', nn.LayerNorm(80+action_size+1))
        self.next_state_predict_agent_mh.add_module('linear1', nn.Linear(80+action_size+1, 100, bias=False))
        self.next_state_predict_agent_mh.add_module('relu1', nn.ReLU(inplace=True))
        self.next_state_predict_agent_mh.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.next_state_predict_agent_mh.add_module('relu2', nn.ReLU(inplace=True))

        self.next_state_predict_agent_inventory = nn.Sequential()
        self.next_state_predict_agent_inventory.add_module('norm', nn.LayerNorm(80+action_size+1))
        self.next_state_predict_agent_inventory.add_module('linear1', nn.Linear(80+action_size+1, 100, bias=False))
        self.next_state_predict_agent_inventory.add_module('relu1', nn.ReLU(inplace=True))
        self.next_state_predict_agent_inventory.add_module('linear2', nn.Linear(100, 20, bias=False))
        self.next_state_predict_agent_inventory.add_module('relu2', nn.ReLU(inplace=True))
       
        self.normalize_rewards = nn.LayerNorm(self.seq_len)

        self.normalize_q_value = nn.BatchNorm1d(1)
        self.qvalue_lstm = torch.nn.LSTM(80+action_size+1, 1, num_layers=2, batch_first=True, bias=False)  
        self.qvalue_lstm_2 = torch.nn.LSTM(80+action_size+1, 1, num_layers=2, batch_first=True, bias=False)  

        self.qvalue = nn.Sequential()
        self.qvalue.add_module('norm', nn.LayerNorm(80+action_size+1))
        self.qvalue.add_module('linear1', nn.Linear(80+action_size+1, 100, bias=False))
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

        for m in self.action_modules_1:
            nn.init.xavier_uniform_(self.action_modules_1[m].weight, gain=nn.init.calculate_gain('relu'))
            #nn.init.constant_(self.action_modules_1[m].bias, 0)


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
        #pil_img = transforms.ToPILImage()(world_state[0,:,0,:,:].cpu())
        #imshow(pil_img)
        #pyplot.show

        batch_size = world_state.shape[0]
        num_layers = 2

        x = self.cnn(world_state).squeeze(dim=3).squeeze(dim=3)
        x = x.permute(0,2,1)
        h0, c0 = self.init_hidden(batch_size, num_layers, 60)
        #h0 = h0.permute(0,2,1)
        #c0 = c0.permute(0,2,1)
        world_state, (hidden, cell) = self.pov_lstm(x, (h0, c0))

        agent_state_inventory = self.normalize_inventory(agent_state_inventory)
        h0, c0 = self.init_hidden(batch_size, num_layers, 60)
        agent_state_inventory, (hidden, cell) = self.inventory_lstm(agent_state_inventory, (h0, c0))

        agent_state_mh = self.normalize_mh(agent_state_mh)
        h0, c0 = self.init_hidden(batch_size, num_layers, 20)
        agent_state_mh, (hidden, cell) = self.mh_lstm(agent_state_mh, (h0, c0))

        world_state = world_state
        combined_state = torch.cat([world_state, agent_state_mh[:,-1,:].unsqueeze(dim=1), agent_state_inventory[:,-1,:].unsqueeze(dim=1)], 2)
        h0, c0 = self.init_hidden(batch_size, num_layers, 80)
        combined_state, (hidden, cell) = self.cnn_mh_inventory_lstm(combined_state, (h0, c0)) 
        combined_state = self.normalize_action_inputs(combined_state)

        actions = {}
        actions_raw = []
        action_logits = []

        
        for action in self.action_modules_lstm:
            
            h0, c0 = self.init_hidden(batch_size, num_layers, self.action_modules_1_output_size[action])
            out, (hidden, cell) = self.action_modules_lstm[action](combined_state, (h0, c0))
            #out = self.action_modules_1[action](combined_state)
            out = self.activation_modules[action](out)
            #if not self.activation_modules.training:
            #    noise = np.random.normal(scale=0.1, size=out.shape)
            #    out = out + torch.tensor(noise).float().to(device)
            
            
            
            if action == "forward_":
                action = "forward"
            
            if action in ["craft", "equip","nearbyCraft","nearbySmelt","place"]:
                action_logits.append(out)
                out = F.softmax(out, dim=2)
                out = out.argmax(dim=2, keepdim=True).float()
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
                actions[action] = out[-1,-1,:].int().item()
            #elif action == "forward":
            #    actions[action] = 1                               
            elif action != "camera":
                actions[action] = out[-1,-1,:].int().item()
            elif action == "camera":
                actions[action] = out[-1,-1,:].tolist()
      

        actions_raw = torch.cat(actions_raw, dim=2)
        z = torch.cat([combined_state, actions_raw], 2)
        #n_wsd_predict = self.next_state_predict_cnn(z)
        #n_asmhd_predict = self.next_state_predict_agent_mh(z)
        #n_asinventoryd_predict = self.next_state_predict_agent_inventory(z)
        z = self.normalize_q_value(z)
        h0, c0 = self.init_hidden(batch_size, num_layers, 1)
        q_value, (hidden, cell) = self.qvalue_lstm(z, (h0, c0))
        #q_value, (hidden, cell) = self.qvalue_lstm_2(q_value)
        q_value = q_value.squeeze()
        
        action_logits = torch.cat(action_logits, dim=2)
        
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


