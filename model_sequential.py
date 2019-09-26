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

class Linear_Net(nn.Module):

    def __init__(self, name, input_size, hidden_sizes, output_size, seq_len):
        super(Linear_Net, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.seq_len = seq_len 

        self.net = nn.Sequential()
        self.net.add_module(name+'_norm0_0', nn.LayerNorm(input_size))
        self.net.add_module(name+"_linear0", nn.Linear(input_size, hidden_sizes[0], bias=False))
        self.net.add_module(name+'_relu0', nn.ReLU(inplace=True))
        self.net.add_module(name+'_norm0', nn.LayerNorm(hidden_sizes[0]))

        for i in range(len(hidden_sizes)-1):
            self.net.add_module(name+'_linear'+str(i+1), nn.Linear(hidden_sizes[i], hidden_sizes[i+1], bias=False))
            self.net.add_module(name+'_relu'+str(i+1), nn.ReLU(inplace=True))
            self.net.add_module(name+'_norm'+str(i+1), nn.LayerNorm(hidden_sizes[i+1]))

        self.net.add_module(name+'_linear'+str(i+2), nn.Linear(hidden_sizes[-1], output_size, bias=False))
        self.net.add_module(name+'_relu'+str(i+2), nn.ReLU(inplace=True))
        self.net.add_module(name+'_norm'+str(i+2), nn.LayerNorm(output_size))

        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        print(self.net)

    def forward(self, state):
        return self.net(state)






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
        self.cnn.add_module('conv1', nn.Conv3d(n_c, growth_rate, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False))
        self.cnn.add_module('relu1', nn.ReLU(inplace=True))
        self.cnn.add_module('norm2', nn.BatchNorm3d(growth_rate))        
        self.cnn.add_module('conv2', nn.Conv3d(growth_rate, growth_rate, kernel_size=(self.seq_len,1,1), stride=1, bias=False))
        self.cnn.add_module('relu2', nn.ReLU(inplace=True))
        self.cnn.add_module('norm3', nn.BatchNorm3d(growth_rate))
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


        self.normalize_inventory = nn.LayerNorm([self.seq_len, 18])
        self.inventory_lstm = torch.nn.LSTM(agent_inventory_size, 100, num_layers=2, batch_first=True, bias=False)  
        
        self.normalize_mh = nn.LayerNorm([self.seq_len,3])
        self.mh_lstm = torch.nn.LSTM(agent_mh_size, 20, num_layers=2, batch_first=True, bias=False)  

 
        # self.cnn_mh_inventory = Linear_Net(growth_rate+120, [200, 100], 50, seq_len)
        self.cnn_mh_inventory = nn.Sequential()
        self.cnn_mh_inventory.add_module('norm', nn.LayerNorm(growth_rate+120))
        self.cnn_mh_inventory.add_module('linear1', nn.Linear(growth_rate+120, 200, bias=False))
        self.cnn_mh_inventory.add_module('relu1', nn.ReLU(inplace=True))
        self.cnn_mh_inventory.add_module('norm2', nn.LayerNorm(200))
        self.cnn_mh_inventory.add_module('linear2', nn.Linear(200, 100, bias=False))
        self.cnn_mh_inventory.add_module('relu2', nn.ReLU(inplace=True))
        self.cnn_mh_inventory.add_module('norm3', nn.LayerNorm(100))
        self.cnn_mh_inventory.add_module('linear3', nn.Linear(100, 50, bias=False))
        self.cnn_mh_inventory.add_module('relu3', nn.ReLU(inplace=True))
        #self.cnn_mh_inventory_lstm.register_backward_hook(self.back_hook) 
              
        self.normalize_action_inputs = nn.LayerNorm(50)
        self.output_action_modules = nn.ModuleDict({
            'attack': nn.Linear(50,1, bias=False),
            'back': nn.Linear(50,1, bias=False),
            'camera': nn.Linear(50,2, bias=False),
            'craft': nn.Linear(50,5, bias=False),
            'equip': nn.Linear(50,8, bias=False),
            'forward_': nn.Linear(50,1, bias=False),
            'jump': nn.Linear(50,1, bias=False),
            'left': nn.Linear(50,1, bias=False),
            'nearbyCraft': nn.Linear(50,8, bias=False),
            'nearbySmelt': nn.Linear(50,3, bias=False),
            'place': nn.Linear(50,7, bias=False),
            'right': nn.Linear(50,1, bias=False),
            'sneak': nn.Linear(50,1, bias=False),
            'sprint': nn.Linear(50,1, bias=False)
        })
        #self.action_modules_1["attack"].register_backward_hook(self.back_hook) 

        self.action_modules_1_output_size = {'attack':1, 'back':1, 'camera':2, 'craft':5,\
            'equip':8, 'forward_':1, 'jump':1, 'left':1, 'nearbyCraft':8, 'nearbySmelt':3,\
            'place':7, 'right':1, 'sneak':1, 'sprint':1}       

        self.output_action_activation_modules = nn.ModuleDict({
            'attack': nn.ReLU(),
            'back': nn.ReLU(),
            'camera': nn.Identity(),
            'craft': nn.ReLU(),
            'equip': nn.ReLU(),
            'forward_': nn.ReLU(),
            'jump': nn.ReLU(),
            'left': nn.ReLU(),
            'nearbyCraft': nn.ReLU(),
            'nearbySmelt': nn.ReLU(),
            'place': nn.ReLU(),
            'right': nn.ReLU(),
            'sneak': nn.ReLU(),
            'sprint': nn.ReLU(),
        })
 

        self.output_action_normalize = nn.ModuleDict({
            'attack': nn.LayerNorm(1),
            'back': nn.LayerNorm(1),
            'camera': nn.Identity(),
            'craft': nn.LayerNorm(5),
            'equip': nn.LayerNorm(8),
            'forward_': nn.LayerNorm(1),
            'jump': nn.LayerNorm(1),
            'left': nn.LayerNorm(1),
            'nearbyCraft': nn.LayerNorm(8),
            'nearbySmelt': nn.LayerNorm(3),
            'place': nn.LayerNorm(7),
            'right': nn.LayerNorm(1),
            'sneak': nn.LayerNorm(1),
            'sprint': nn.LayerNorm(1),
        })       
        self.normalize_rewards = nn.LayerNorm(self.seq_len)


        self.camera_pitch = nn.Sequential()
        self.camera_pitch.add_module('norm', nn.LayerNorm(50))
        self.camera_pitch.add_module('linear1', nn.Linear(50, 100, bias=False))
        self.camera_pitch.add_module('tanhshrik1', nn.Tanhshrink())
        self.camera_pitch.add_module('norm2', nn.LayerNorm(100))
        self.camera_pitch.add_module('linear2', nn.Linear(100, 1, bias=False))
        self.camera_pitch.add_module('tanhshrink2', nn.Tanhshrink())

        self.camera_yaw = nn.Sequential()
        self.camera_yaw.add_module('norm', nn.LayerNorm(50))
        self.camera_yaw.add_module('linear1', nn.Linear(50, 100, bias=False))
        self.camera_yaw.add_module('tanhshrik1', nn.Tanhshrink())
        self.camera_yaw.add_module('norm2', nn.LayerNorm(100))
        self.camera_yaw.add_module('linear2', nn.Linear(100, 1, bias=False))
        self.camera_yaw.add_module('tanhshrink2', nn.Tanhshrink())

        self.qvalue = nn.Sequential()
        self.qvalue.add_module('norm', nn.LayerNorm(50+action_size+1))
        self.qvalue.add_module('linear1', nn.Linear(50+action_size+1, 100, bias=False))
        self.qvalue.add_module('relu1', nn.ReLU(inplace=True))
        self.qvalue.add_module('norm2', nn.LayerNorm(100))
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

        for m in self.camera_pitch:
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('tanh'))
                #nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))

        for m in self.camera_yaw:
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
        world_state = x.permute(0,2,1)

        agent_state_inventory = self.normalize_inventory(agent_state_inventory)
        h0, c0 = self.init_hidden(batch_size, num_layers, 100)
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
            out = self.output_action_normalize[action](out)
            #if not self.output_action_activation_modules.training:
            #    noise = np.random.normal(scale=0.05, size=out.shape)
            #    out = out + torch.tensor(noise).float().to(device)
            
            
            
            if action == "forward_":
                action = "forward"
            
            if action in ["craft", "equip","nearbyCraft","nearbySmelt","place"]:
                out = F.softmax(out, dim=1)
                action_logits.append(out)
                out = out.argmax(keepdim=True, dim=1).float()
            elif action != "camera":
                out = torch.sigmoid(out)
                action_logits.append(out)
                zeros = torch.zeros_like(out)
                ones = torch.ones_like(out)
                out = torch.where(out > 0.5, ones, zeros).float()
            elif action == "camera":
                #out = torch.clamp(out, min=-180, max=180)
                #out = out.float()/180.0

                combined_state_2 = combined_state.detach()
                yaw = self.camera_yaw(combined_state_2)
                yaw = torch.clamp(yaw, min=-180, max=180)

                pitch = self.camera_pitch(combined_state_2)
                pitch = torch.clamp(pitch, min=-180, max=180)

                out = torch.cat([pitch,yaw], axis=1)
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


class Critic_TS(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, agent_mh_size, agent_inventory_size, world_state_size,\
            action_size, action_logits_size,  seed, seq_len, growth_rate=128):
        """Initialize parameters and build model.
        Params
        ======
        """        
        super(Critic_TS, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.seq_len = seq_len
        
        n_c = world_state_size[0]
        n_d = world_state_size[1]
        i_h = world_state_size[2]
        i_w = world_state_size[3]
         
        # agent's pov
        self.cnn = nn.Sequential()
        self.cnn.add_module('norm1', nn.BatchNorm3d(n_c))        
        self.cnn.add_module('conv1', nn.Conv3d(n_c, growth_rate, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False))
        self.cnn.add_module('relu1', nn.ReLU(inplace=True))
        self.cnn.add_module('norm2', nn.BatchNorm3d(growth_rate))        
        self.cnn.add_module('conv2', nn.Conv3d(growth_rate, growth_rate, kernel_size=(self.seq_len,1,1), stride=1, bias=False))
        self.cnn.add_module('relu2', nn.ReLU(inplace=True))
        self.cnn.add_module('norm3', nn.BatchNorm3d(growth_rate))
        self.cnn.add_module('pool1', nn.AvgPool3d(kernel_size=(1,i_h,i_w), stride=(1,1,1)))


        self.normalize_inventory = nn.LayerNorm([self.seq_len, 18])
        self.inventory_lstm = torch.nn.LSTM(agent_inventory_size, 100, num_layers=2, batch_first=True, bias=False)  
        
        self.normalize_mh = nn.LayerNorm([self.seq_len,3])
        self.mh_lstm = torch.nn.LSTM(agent_mh_size, 20, num_layers=2, batch_first=True, bias=False)  


        self.cnn_mh_inventory = Linear_Net("cnn_mh_invent", growth_rate+120, [200, 100], 50, seq_len)              
        self.qvalue = Linear_Net("q_value", 50+action_logits_size, [100, 25], 1, seq_len)


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

                                                    
    def forward(self, agent_state_mh, world_state, agent_state_inventory, action_logits):
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
        h0, c0 = self.init_hidden(batch_size, num_layers, 100)
        agent_state_inventory, (hidden, cell) = self.inventory_lstm(agent_state_inventory, (h0, c0))

        agent_state_mh = self.normalize_mh(agent_state_mh)
        h0, c0 = self.init_hidden(batch_size, num_layers, 20)
        agent_state_mh, (hidden, cell) = self.mh_lstm(agent_state_mh, (h0, c0))

        combined_state = torch.cat([world_state, agent_state_mh[:,-1,:].unsqueeze(dim=1), agent_state_inventory[:,-1,:].unsqueeze(dim=1)], 2)        
        combined_state = combined_state.squeeze(dim=1)
        combined_state = self.cnn_mh_inventory(combined_state) 
     
        z = torch.cat([combined_state, action_logits], axis=1)

        q_value = self.qvalue(z)
        q_value = q_value.squeeze()
        
        
        return q_value

