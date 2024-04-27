import torch
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from collections import deque, namedtuple
import random
from agents.baseagent import BaseAgent
import pandas as pd
import os
import sys   

class DQN(BaseAgent, nn.Module):
    def __init__(self,
                 name: str,
                 environment,
                 reward_function,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 activation_function,
                 num_hidden_layers: int,
                 buffer_size: int, 
                 batch_size: int,        
                 alpha: float = 0.1,
                 gamma: float = 0.9,
                 device: str = 'cpu',
                 opt_lr: float = 0.001,
                 opt_wgt_dcy: float = 0.0,
                 dropout_rate: float = 0.25,
                 sub_agents = None):
        
        # Call the initialization of both parent classes
        BaseAgent.__init__(self, name, reward_function, environment, sub_agents)
        nn.Module.__init__(self)
        
        # Device to Compute Tensors
        self.device = device
        
        # Initialize Q Network
        self.Q_nn = Q_Network(input_size = input_size,
                                hidden_size = hidden_size,
                                output_size = output_size,
                                activation_function = activation_function,
                                num_hidden_layers = num_hidden_layers,
                                dropout_rate = dropout_rate,
                                opt_lr = opt_lr,
                                opt_wgt_dcy = opt_wgt_dcy,
                                device = self.device)
        
        self.Q_tgt_nn = self._create_tgt_nn(self.Q_nn,self.device)

        
        ## Optimizer
        self.update_q_freq = 1 ## How many steps before triggering batch training       
        self.alpha = alpha        ## learning rate
        self.gamma = gamma        ## Discount factor for future rewards
        self.state = None
        
        
        # Dictionary to translate actions between NN and Env
        self._act_nn_to_env = {0: 'S', 1: 'H', 2 :'B'}
        self._act_env_to_nn = {'S': 0,'H': 1, 'B': 2 }
        
        # Initialize Replay Memory
        self.replay_memory = ExperienceBuffer(buffer_size, self.device)
        self.batch_size = batch_size
        
    
    @torch.no_grad()  
    def test(self, start_idx:int, end_idx:int, testing_episodes=1):
        
        ## Disables Dropout Layers in Q-network
        self.Q_nn.eval()
        
        print(f'{self.get_name()}: Testing Initialized on [{start_idx}:{end_idx}]')
        episodic_data = []

        self.env.update_idx(start_idx,end_idx)
        
                
        for episode_num in range(1, testing_episodes+1):

            
            
            reward, loss = self._play_episode(0, None , None , 'testing')
            epi_data = {"Testing Episode": episode_num, 
                        "Total Reward": reward,
                        'Loss': loss}
            episodic_data.append(epi_data)
            print(
                (f'\r{self.get_name()} - {self.env.get_name()}[{start_idx}:{end_idx}] ' +
                f'- Testing Finished - EPIDSODE - {episode_num} of {testing_episodes}' +
                f'-> Reward = {reward:.2f}'), end="", flush=True)
        
        
        self.testing_episodic_data = episodic_data
        print(f'\n{self.get_name()}: {self.env.get_name()} Testing Complete')              
            
    def train(self, start_idx:int, 
              end_idx:int, 
              training_episodes,
              epsilon_decya_func, 
              initial_epsilon, 
              final_epsilon, 
              update_q_freq = None,
              update_tgt_freq = None):
        
        # Q-Network is trained by step by default
        if update_q_freq is None:
            self.update_q_freq = 1
        
        # Target Network is updated by 51% of buffer size by default
        if update_tgt_freq is None:
            self.update_tgt_freq = np.round(self.buffer.maxlen * 0.51).astype(int)


        ## Enable Dropout Layers
        self.Q_nn.train()
        
        print(f'{self.get_name()}: Training Initialized on [{start_idx}:{end_idx}]')
        
        episodic_data = []

        self.env.update_idx(start_idx,end_idx)
        
                
        for episode_num in range(1, training_episodes+1):

            
            epsilon = epsilon_decya_func(initial_epsilon, 
                                            final_epsilon,
                                            episode_num,
                                            training_episodes)
            
            reward, loss = self._play_episode(epsilon, update_q_freq, update_tgt_freq, 'training')
            epi_data = {"Training Episode": episode_num, 
                        "Total Reward": reward,
                        'Loss': loss,
                        "Epsilon": epsilon}
            episodic_data.append(epi_data)
            print(
                (f'\r{self.get_name()}: EPIDSODE {episode_num} of {training_episodes} Finished ' +
                f'-> Loss = {loss:.2f}, Reward = {reward:.2f}'), end="", flush=True)
        
        
        self.training_episodic_data = episodic_data
        print(f'\n{self.get_name()}: Training finished on {self.env.get_name()}[{start_idx}:{end_idx}]]')       
    
    def _play_episode(self,epsilon, update_q_freq,update_tgt_freq, step_type):
        total_reward = 0.0
        self.step_info = []
        self.env.reset()
        is_done = False
        total_steps = 0
        loss = None


        while not is_done:
            if step_type == "testing": # Always use best action
                _ , action, reward, _ , end , action_type, q_vals = self._act(0, step_type)
            
            elif step_type == 'training':    
                
                # Act On Ev
                state, action, reward, new_state, end, action_type, q_vals = self._act(epsilon, step_type)
                exp = Experience(state, self._act_env_to_nn[action], reward,end, new_state)
                self.replay_memory.append(exp)
                
                # Training Q-Network 
                q_nn_update_bool = total_steps % update_tgt_freq == 0                    
                if self.replay_memory.is_full() and q_nn_update_bool:
                    loss = self._learn()
                
                # Update Tgt-Network
                tgt_nn_update_bool = total_steps % update_q_freq == 0                    
                if tgt_nn_update_bool:
                    self.Q_tgt_nn = self._create_tgt_nn(self.Q_nn,self.device)

            else:
                raise ValueError(f'Invalid step_type: {step_type}')
                    

            total_reward += reward
            step_data = {f'{self.name} Action': action, 
                        f'{self.name} Action Type': action_type,
                        f'{self.name} Q_Val Sell': q_vals[0],
                        f'{self.name} Q_Val Hold': q_vals[1],
                        f'{self.name} Q_Val Buy': q_vals[2],
                        f'{self.name} Reward': reward}
            self.step_info.append(step_data)
            is_done = end
            
            total_steps += 1
            total_reward += reward
            
        return total_reward, loss
    
    def _learn(self):
              
        b_states, b_actions, b_rewards, b_done, b_new_states,  = self.replay_memory.sample(self.batch_size)

        b_Q_nn_qvals = self.Q_nn(b_states).to(self.device)
        b_tgt_nn_max_qvals= self.Q_tgt_nn(b_new_states).detach().max(1)[0]
        target = b_Q_nn_qvals.clone().to(self.device)
        
        # Ensure batch_size is moved to CPU for indexing
        b_actions = b_actions.cpu()
        
        for i in range(self.batch_size):
            target[i, b_actions[i]] = b_rewards[i] + self.gamma * b_tgt_nn_max_qvals[i] * (not b_done[i])

        # Loss and optimize
        self.Q_nn.optimizer.zero_grad()
        loss = nn.functional.smooth_l1_loss(b_Q_nn_qvals, target)
        #loss = nn.functional.mse_loss(b_Q_nn_qvals, target)
        loss.backward()
        self.Q_nn.optimizer.step()
        
        return loss.item()        

    def _create_tgt_nn(self, Q_nn, device):
        """
        Create a deep copy of the Q_nn for the target network.
        """
        # Serialize the original model's state
        target_network = copy.deepcopy(Q_nn)
        
        # Move the copied model to the specified device
        target_network.to(device)
        
        return target_network
        
    def _act(self, epsilon, step_type):
        
        state = self.env.get_observation()
        q_vals = self.Q_nn(torch.tensor(state,dtype=torch.float32).to(self.device)).tolist()
        if np.random.rand() < epsilon:
            action = random.choice(self.get_avail_actions())
            action_type = "Explore"
        else:
            action = self._act_nn_to_env[self._best_action(state, step_type)]
            action_type = "Best"
                    
        new_state, reward, is_done = self.env.step(self, action, step_type) #Passing Self to allow enviornment to get Agent connected functions
        
        return(state, action, reward, new_state, is_done, action_type, q_vals)

    def _best_action(self,state,step_type):
        q_values = self.Q_nn(torch.tensor(state,dtype=torch.float32).to(self.device))
        # Q-values are only attached for Gradient calc during mini-batch training
        if (step_type == 'testing') or (not self.replay_memory.is_full()):
            q_values = q_values.detach()
        # Compute the best action based on the Q-values
        avaliable_actions = [self._act_env_to_nn[act] for act in self.get_avail_actions()]
        selected_qvals = torch.index_select(q_values, 0 ,torch.tensor(avaliable_actions))
        agrmax_index = torch.argmax(selected_qvals)
        best_action = avaliable_actions[agrmax_index]
        return best_action

    def _reset_parameters(self, nn):
        for _, module in nn.named_children():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def reset_nn(self):
        self._reset_parameters(self.Q_nn)
        print(f'{self.get_name()}: Q-Network Reset Successfully')
            
    
    def export_Q_nn(self,filenamepath):
        torch.save(self.Q_nn,filenamepath)
        print(f'{self.get_name()}: Q-Network Exported to file "{filenamepath}"')
    
    def import_Q_nn(self,filenamepath):
        self.Q_nn = torch.load(filenamepath)
        print(f'{self.get_name()}: Q-Network Imported from file "{filenamepath}"\n{self.Q_nn}')
        
    def get_training_episodic_data(self):
        return pd.DataFrame(self.training_episodic_data)  # Generate a DataFrame from stored step information  
    
    def get_testing_episodic_data(self):
        return pd.DataFrame(self.testing_episodic_data)  # Generate a DataFrame from stored step information
    
    def get_step_data(self):
        return pd.DataFrame(self.step_info)  # Generate a DataFrame from stored step information   


Experience = namedtuple('Experience', field_names=['state',
                                                   'action',
                                                   'reward',
                                                   'is_done',
                                                   'new_state'])
           
class ExperienceBuffer:
    def __init__(self,capacity: int, device: str) -> None:
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.device = device
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self,experience):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        sampled_experiences = [self.buffer[idx] for idx in indices]
        
        # Extract individual fields from sampled experiences
        states = [exp.state for exp in sampled_experiences]
        actions = [exp.action for exp in sampled_experiences]
        rewards = [exp.reward for exp in sampled_experiences]
        dones = [exp.is_done for exp in sampled_experiences]
        next_states = [exp.new_state for exp in sampled_experiences]
        
        # Convert lists to PyTorch tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int8).to(self.device) 
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.bool).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        
        return states, actions, rewards, dones, next_states
    def reset(self):
        self.buffer = deque(maxlen = self.capacity)
    
    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen

class Q_Network(nn.Module):
    def __init__(self,
                input_size: int,
                hidden_size: int,
                output_size: int,
                activation_function,
                num_hidden_layers: int,   
                device: str = 'cpu',
                opt_lr: float = 0.001,
                opt_wgt_dcy: float = 0.0,
                dropout_rate: float = 0.25):
        
        super(Q_Network, self).__init__()
    
        ## Q Network Layers
        layers = []
        
        ## Input Layer
        layers.append(nn.Linear(input_size, hidden_size))

        ## Hidden Layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(activation_function)
            layers.append(nn.Dropout(dropout_rate))

        ## Output Layer
        layers.append(nn.Linear(hidden_size, output_size))


        # Apply Glorot Normal initialization to the linear layers' weights
        for i in range(0, len(layers), 3):
            if isinstance(layers[i], nn.Linear):
                nn.init.kaiming_normal_(layers[i].weight)
                
        ## Create Q Network
        self.Q_nn = nn.Sequential(*layers)
        self.Q_nn.to(torch.device(device))
        
        # Initialize Optimizer
        self.optimizer = optim.Adam(self.Q_nn.parameters(), lr=opt_lr, weight_decay=opt_wgt_dcy)

    def forward(self, x):
        # Forward pass through the network
        return self.Q_nn(x)
        

class EarlyStopping:
    # From https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience # how many times will you tolerate for loss not being on decrease
        self.verbose = verbose  # whether to print tip info
        self.counter = 0 # now how many times loss not on decrease
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path) -> str:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            msg = (f' -> Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.save_checkpoint(val_loss, model, path)

        # meaning: current score is not 'delta' better than best_score, representing that 
        # further training may not bring remarkable improvement in loss. 
        elif score < self.best_score + self.delta:  
            self.counter += 1
            msg = (f' -> EarlyStopping counter: {self.counter} out of {self.patience}')
            # 'No Improvement' times become higher than patience --> Stop Further Training
            if self.counter >= self.patience:
                self.early_stop = True

        else: #model's loss is still on decrease, save the now best model and go on training
            self.best_score = score
            msg = (f' -> Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
            self.save_checkpoint(val_loss, model, path)
            
            self.counter = 0
        
        return msg
    
    def new_target(self,target):
        # val_loss input to __call__ is based on a new target, requiring reset of counter
        # best_score, and early stoppage flag
        
        self.counter = 0 
        self.best_score = None
        self.early_stop = False
        msg = (f' -> New Target Established {target:.2f} - Reset Early Stopping')
        
        return msg
        

    def save_checkpoint(self, val_loss, model, path):
    ### used for saving the current best model
         
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss
    
    def load_checkpoint(self, model, path):
    ### Used for loading the saved model from the checkpoint file
        model.load_state_dict(torch.load(path))
        print('Last Checkpoint loaded')