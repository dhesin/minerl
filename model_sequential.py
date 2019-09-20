import numpy as np

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
        self.cnn.add_module('conv1', nn.Conv3d(n_c, growth_rate, kernel_size=1, stride=1, bias=False))
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
            'attack': nn.Linear(80,1, bias=True),
            'back': nn.Linear(80,1, bias=True),
            'camera': nn.Linear(80,2, bias=True),
            'craft': nn.Linear(80,5, bias=True),
            'equip': nn.Linear(80,8, bias=True),
            'forward_': nn.Linear(80,1, bias=True),
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
            'attack': nn.Softsign(),
            'back': nn.Softsign(),
            'camera': nn.Softsign(),
            'craft': nn.Softsign(),
            'equip': nn.Softsign(),
            'forward_': nn.Sigmoid(),
            'jump': nn.Softsign(),
            'left': nn.Softsign(),
            'nearbyCraft': nn.Softsign(),
            'nearbySmelt': nn.Softsign(),
            'place': nn.Softsign(),
            'right': nn.Sigmoid(),
            'sneak': nn.Softsign(),
            'sprint': nn.Softsign(),
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

        self.normalize_q_value = nn.BatchNorm1d(self.seq_len)
        self.qvalue_lstm = torch.nn.LSTM(80+action_size+1, 1, num_layers=2, batch_first=True, bias=False)  
        self.qvalue_lstm_2 = torch.nn.LSTM(80+action_size+1, 1, num_layers=2, batch_first=True, bias=False)  

        self.qvalue = nn.Sequential()
        self.qvalue.add_module('norm', nn.LayerNorm(80+action_size+1))
        self.qvalue.add_module('linear1', nn.Linear(80+action_size+1, 100, bias=False))
        self.qvalue.add_module('relu1', nn.ReLU(inplace=True))
        self.qvalue.add_module('linear2', nn.Linear(100, 1, bias=False))
        self.qvalue.add_module('relu2', nn.ReLU(inplace=True))



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
            nn.init.constant_(self.action_modules_1[m].bias, 0)


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


        combined_state = torch.cat([world_state, agent_state_mh, agent_state_inventory], 2)
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
                #out = torch.clamp(out, min=-180, max=180)
                out = out.float()
                action_logits.append(out)
                out /= 180.0
                                       
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



