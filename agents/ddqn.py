import torch # For something
import torch.nn as nn
import torch.optim as optim
import copy
import numpy as np
from collections import deque, namedtuple
import random
from agents.baseagent import BaseAgent
import os
import sys
import pandas as pd
from decimal import Decimal
import json
import importlib
import time



class DdqnAgent(BaseAgent, nn.Module):
    """
    A Double Deep Q-Network (DDQN) reinforcement learning agent with two Q-learning networks complete
    with each target networks for each Q-network. 

    Args:
        name (str): 
            The name of the agent, also used in the environment class.
        environment (object): 
            The environment with which the agent interacts.
        reward_function (function): 
            Linked reward function to the agent, which is called by the environment class.
        environment (object): 
            The environment with which the agent interacts.
        input_size (int):
            Number of neurons in the input layer, should match the size of enviornment size
        hidden_size (int):
            Number of neurons in each fully connnected layer
        output_size (int)
            Number of neurons in the output layer, should match actions space in assigned enviornment
        activation_function (Torch.func)
            Activation function used between each fully connected layer
        num_hidden_layers (int):
            Number of fully connected neuron layers in ANN
        buffer_size (int):
            Memory size of agent to draw upon during ANN update             
        batch_size (int):
            Number of samples randomly selected from buffer during each ANN update        
        alpha (float):  0.1
            Learning Rate
        gamma (float): 0.9
            Discount rate of future rewards 
        device (str): cpu
            Device assignment for tensor calculation            
        opt_lr (float): 0.001,
            Learning rate of the ANN
        opt_wgt_dcy (float):  0.0
            Decay rate of weights in ANN
        dropout_rate (float): 0.25 
            Percentage of neurons dropped out during training
        env_state_mod_func (func):
            Linked enviornment state modifier function that modifies the observed env state prior
            to digestion by agent
        env_state_mod_params (dic):
            Dictionary where arugments for the linked environment state modifier function can be passed
        reward_params (dic):
            Dictionary where arugments for the linked reward function can be passed
    """
    def __init__(self, name: str, 
                 environment,
                 reward_function,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 activation_function,
                 num_hidden_layers: int,
                 buffer_size: int, 
                 batch_size: int,
                 device,        
                 gamma: float = 0.9,
                 alpha: float = 0.001,
                 opt_wgt_dcy: float = 0.0,
                 dropout_rate: float = 0.25,
                 env_state_mod_func = None,
                 env_state_mod_params = {},
                 reward_params = None):
        
        # Call the initialization of both parent classes
        BaseAgent.__init__(self, name, reward_function, environment, reward_params)
        nn.Module.__init__(self)
        
        # Prepossing of Environment State before Agent digest
        self.env_state_mod_func = env_state_mod_func
        self.env_state_mod_params = env_state_mod_params
        
        # Ensure the device is valid
        if not isinstance(device, torch.device):
            raise ValueError("The 'device' argument must be an instance of torch.device.")
        
        # Device to Compute Tensors
        self.device = device
        
        # Initialize Q Network
        self.Q1_nn = Q_Network(input_size = input_size,
                                hidden_size = hidden_size,
                                output_size = output_size,
                                activation_function = activation_function,
                                num_hidden_layers = num_hidden_layers,
                                dropout_rate = dropout_rate,
                                opt_lr = alpha,
                                opt_wgt_dcy = opt_wgt_dcy,
                                device = self.device)
        
        self.Q1_tgt_nn = self._create_tgt_nn(self.Q1_nn,self.device)
            
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
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
        print(f'{self.get_name()} initialized on {self.device}')
       
    
    @torch.no_grad()  
    def test(self, start_idx:int, end_idx:int, testing_episodes=1, metric_func = None, metric_func_arg = {}):
        
        self.metric_func = metric_func
        self.metric_func_arg = metric_func_arg
        
        ## Disables Dropout Layers in Q-networks
        self.Q1_nn.eval()
        
        print(f'\n{self.get_name()}: Testing Initialized on {self.env.get_name()}[{start_idx}:{end_idx}]')
        episodic_data = []

        self.env.update_idx(start_idx,end_idx)
        
                
        for episode_num in range(1, testing_episodes+1):
          
            tot_reward, mean_reward, std_reward, _, actions = self._play_episode(0, None , None , 'testing')
            epi_data = {"tst_ep": episode_num, 
                        "tot_r": tot_reward,
                        "avg_r": mean_reward,
                        "std_r": std_reward,
                        'tst_actions': actions}
            episodic_data.append(epi_data)
                      
            print(
                (f'\r{self.get_name()} - {self.env.get_name()}[{start_idx}:{end_idx}] ' +
                f'- Testing Finished - EP - {episode_num} of {testing_episodes}' +
                f'-> ∑R = {tot_reward:.2f}, μR = {mean_reward:.2f}, ' +
                f'σR = {std_reward:.2f}'), end="", flush=True)
        
        
        self.testing_episodic_data = episodic_data
        print(f'\n{self.get_name()}: Testing Complete on {self.env.get_name()}[{start_idx}:{end_idx}]\n')              
    
    @torch.no_grad()
    def _validate(self, val_start_idx:int, val_end_idx:int):
        
        ## Disables Dropout Layers in Q-networks
        self.Q1_nn.eval()
        
        self.env.update_idx(val_start_idx,val_end_idx)
        
        tot_val_metric, mean_val_metric, std_val_metric, _, actions = self._play_episode(0, None , None , 'validating')
        ror = self.env.step_info[-1]['Portfolio Value'] / self.env.initial_cash ## likely to remove because of val_metric_func
        cost = self.env.step_info[-1]['Total Commission Cost'] ## likely to remove because of val_metric_func
        

        ## Reenable Dropout Layers in Q-networks
        self.Q1_nn.train()
      
        return tot_val_metric, mean_val_metric, std_val_metric, ror, cost, actions
        
            
    def train(self, start_idx:int, 
              end_idx:int, 
              training_episodes,
              epsilon_decya_func, 
              initial_epsilon, 
              final_epsilon,
              val_start_idx:int = None,
              val_end_idx:int = None,
              save_path:str = None,
              early_stop = False,
              min_training_episodes:int = None,
              metric_func = None,
              metric_func_arg = {}, 
              stop_metric = 'val_tot_r',
              stop_patience = 7,
              stop_delta = 0.01,
              update_q_freq = None,
              update_tgt_freq = None):
        
        # Metric Function connection and agrumetns
        self.metric_func = metric_func
        self.metric_func_arg = metric_func_arg
        
        # Signifance for validiation loss
        self.val_loss_sig = int(np.log10(stop_delta))
        
        # Validation Option
        self.validate = (val_start_idx is not None) and (val_end_idx is not None)
        
        if self.validate:
            if metric_func == None:
               raise ValueError("Metric function not defined for validation")
        
        if not self.validate and (stop_metric in {'val_tot_r', 'val_avg_r', 'val_std_r'}):
            raise ValueError("stop_metric chosen is based on validation set, validation set not defined (val_start_idx, val_end_idx)")
        
        # Q-Network is trained by step by default
        if update_q_freq is None:
            self.update_q_freq = 1
        
        # Target Network is updated by 51% of buffer size by default
        if update_tgt_freq is None:
            self.update_tgt_freq = np.round(self.buffer.maxlen * 0.51).astype(int)

        ## Enable Dropout Layers
        self.Q1_nn.train()

        if self.validate:
            print(f'\n{self.get_name()}: Training Initialized on {self.env.get_name()}[{start_idx}:{end_idx}] -> Validation on {self.env.get_name()}[{val_start_idx}:{val_end_idx}]')
        else:
            print(f'\n{self.get_name()}: Training Initialized on {self.env.get_name()}[{start_idx}:{end_idx}]')
        
        
        model_save_path = save_path + "/" + self.name
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        
        episodic_data = []

        self.env.update_idx(start_idx,end_idx)
        
        if early_stop:
                        
            early_stopping = EarlyStopping( patience = stop_patience, verbose=True, delta=stop_delta, min_training=min_training_episodes)
            loss_type, target =  self._setup_early_stop(stop_metric)

  
        for episode_num in range(1, training_episodes+1):

            
            epsilon = epsilon_decya_func(initial_epsilon, 
                                            final_epsilon,
                                            episode_num,
                                            training_episodes)
            trn_start_time = time.time()
            tot_reward, mean_reward, std_reward, loss, trn_actions = self._play_episode(epsilon, update_q_freq, update_tgt_freq, 'training')
            trn_finish_time = time.time()
            trn_time = trn_finish_time - trn_start_time
            # Rewards based on Validation Set
            if self.validate:
                val_tot_reward, val_avg_reward, val_std_reward, ror, cost, val_actions = self._validate(val_start_idx,val_end_idx)
                # Reset Enviornment to Training Indices
                self.env.update_idx(start_idx,end_idx)              
                epi_data = {"trn_ep": episode_num,
                            "tot_r": tot_reward,
                            "avg_r": mean_reward,
                            "std_r": std_reward,
                            'Q_loss': loss,                      
                            "epsilon": epsilon,
                            'trn_actions': trn_actions,
                            'val_tot_r': val_tot_reward,
                            'val_avg_r': val_avg_reward,
                            'val_std_r': val_std_reward,
                            'val_ror': ror,
                            'val_comm_cost': cost,
                            'val_actions': val_actions}
                
            else: 
                epi_data = {"trn_ep": episode_num, 
                            "tot_r": tot_reward,
                            "avg_r": mean_reward,
                            "std_r": std_reward,
                            'Q_loss': loss,
                            "epsilon": epsilon,
                            'trn_actions': trn_actions}
            
            episodic_data.append(epi_data)
            
            if early_stop:
                current_val = episodic_data[-1][stop_metric]
                
                if loss_type == "Loss":
                    val_loss = current_val
                    stop_msg = early_stopping(val_loss, self.Q1_nn, model_save_path)
                
                else:
                    val_loss = np.round((current_val - target)**2,self.val_loss_sig)
                    stop_msg = early_stopping(val_loss, self.Q1_nn, model_save_path)
                    
                    if loss_type == "Max" and np.round(current_val,self.val_loss_sig) > np.round(target,self.val_loss_sig):
                        target = current_val
                        stop_msg = early_stopping.new_target(target)
                    
                    elif loss_type == "Min" and np.round(current_val,self.val_loss_sig) < np.round(target,self.val_loss_sig):
                        target = current_val
                        stop_msg = early_stopping.new_target(target)

                # Print a line with blank spaces to clear the existing content
                sys.stdout.write('\r' + ' ' * 250)  # Assuming 250 characters wide terminal
                
                if loss is not None:
                    loss_str = f'{loss:.2f}'
                else:
                    loss_str = 'None'
    
                # Print Update
                print(
                    f'\r{self.get_name()}: EP {episode_num} of {training_episodes} Finished ' +
                    f'-> Q_Loss = {loss_str} | Time = {trn_time:.3f} s | ∑R = {tot_reward:.2f}, μR = {mean_reward:.2f} ' +
                    f'σR = {std_reward:.2f} | {loss_type}: {stop_metric} = {current_val:.2f}' + stop_msg, end="", flush=False)
            
            
                if early_stopping.early_stop:
                    self.Q1_nn.load_state_dict(torch.load(model_save_path + '/checkpoint.pth'))
                    print(f'\n{self.get_name()}: Early Stoppage on EP {episode_num} -> Best QNet Loaded from EP {early_stopping.model_save_idx}')
                    break
            else:

                # Print a line with blank spaces to clear the existing content
                sys.stdout.write('\r' + ' ' * 250)  # Assuming 250 characters wide terminal

                # Print Update
                print(
                    f'\r{self.get_name()}: EP {episode_num} of {training_episodes} Finished ' +
                    f'-> ΔQ_Loss = {loss:.2f} | Time = {trn_time:.3f} s | ∑R = {tot_reward:.2f}, ' +
                    f'μR = {mean_reward:.2f} σR = {std_reward:.2f}', end="", flush=False)
        
        # Saving Episodic Data     
        self.training_episodic_data = episodic_data
        
        # For consistent output of final trainig line
        if early_stop:
            if not early_stopping.early_stop:
                txt_format = '\n'
            else:
                txt_format = ''
        else:
            txt_format = '\n'
        
        
        print(f'{txt_format}{self.get_name()}: Training finished on {self.env.get_name()}[{start_idx}:{end_idx}]\n')      
    
    def _play_episode(self,epsilon, update_q_freq,update_tgt_freq, step_type):
        rewards = np.array([])
        actions = []
        self.step_info = []
        self.env.reset()
        self.replay_memory.reset()
        is_done = False
        total_steps = 0
        loss = None


        while not is_done:
            if step_type == "testing" or step_type == 'validating': # Always use best action
                state , action, reward, _ , end , action_type, q_vals = self._act(0, step_type)
            
            elif step_type == 'training':    
                
                # Act On Ev
                state, action, reward, new_state, end, action_type, q_vals = self._act(epsilon, step_type)
                exp = Experience(state, self._act_env_to_nn[action], reward,end, new_state)
                self.replay_memory.append(exp)
                
                # Update Tgt-Network
                tgt_nn_update_bool = total_steps % update_tgt_freq == 0                    
                if tgt_nn_update_bool:
                    self.Q1_tgt_nn = self._create_tgt_nn(self.Q1_nn,self.device)

                # Training Q-Network 
                q_nn_update_bool = total_steps % update_q_freq == 0                    
                if self.replay_memory.is_full() and q_nn_update_bool:
                    loss = self._learn()
               

            else:
                raise ValueError(f'Invalid step_type: {step_type}')
                    

            
            step_data = {f'{self.name} State': state, 
                        f'{self.name} Action': action, 
                        f'{self.name} Action Type': action_type,
                        f'{self.name} Q_Val Sell': q_vals[0],
                        f'{self.name} Q_Val Hold': q_vals[1],
                        f'{self.name} Q_Val Buy': q_vals[2],
                        f'{self.name} Reward': reward}
            self.step_info.append(step_data)
            is_done = end
            
            total_steps += 1
            rewards = np.append(rewards, reward)
            actions = np.append(actions, action)
            
        return np.sum(rewards), np.mean(rewards), np.std(rewards), loss, actions
    
   
    def _learn(self):
        # Sample a batch from replay memory
        b_states, b_actions, b_rewards, b_done, b_new_states = self.replay_memory.sample(self.batch_size)
        
        b_states = b_states.to(self.device)
        b_actions = b_actions.to(self.device)
        b_rewards = b_rewards.to(self.device)
        b_done = b_done.to(self.device)
        b_new_states = b_new_states.to(self.device)

        # Compute the action selection using the main Q-network
        with torch.no_grad():
            selected_actions = self.Q1_nn(b_new_states).argmax(dim=1)
            next_q_values = self.Q1_tgt_nn(b_new_states).gather(1, selected_actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute the target Q-values
        target_q_values = b_rewards + self.gamma * next_q_values * (1 - b_done)
        
        # Compute the current Q-values using the main Q-network
        current_q_values = self.Q1_nn(b_states).gather(1, b_actions.unsqueeze(-1)).squeeze(-1)
        
        # Compute the loss
        loss = nn.functional.smooth_l1_loss(current_q_values, target_q_values)
        
        # Optimize the network
        self.Q1_nn.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.Q1_nn.parameters(), max_norm=1.0)
        self.Q1_nn.optimizer.step()
        
        return loss.item()

    def _create_tgt_nn(self, Q_nn, device):
        """
        Create a deep copy of the Q_nn model and move it to the specified device.
        
        Args:
            Q_nn (torch.nn.Module): The original neural network model.
            device (torch.device): The device to which the copied model will be moved.

        Returns:
            torch.nn.Module: The copied model moved to the specified device.
        """
       
        # Create a deep copy of the model
        target_network = copy.deepcopy(Q_nn)
        
        # Move the copied model to the specified device
        target_network.to(device)
        
        return target_network
        
    def _act(self, epsilon, step_type):
        
        # Agent observes current enviromental state
        state = self.env.get_observation()
        print(state)
        # Apply State Modifier Function, if defined, to current state prior to agent digest
        if self.env_state_mod_func!= None:
            state = self.env_state_mod_func(self.env, **self.env_state_mod_params)
        
        # Choose between best or explore action
        if np.random.rand() < epsilon:
            action = random.choice(self.get_avail_actions())
            action_type = "Explore"
        else:
            action = self._act_nn_to_env[self._best_action(state, step_type)]
            action_type = "Best"
        
        # Record all value of current state and action
        q_vals = self.Q1_nn(torch.tensor(state,dtype=torch.float32).to(self.device)).tolist()
        
        # Act on enviornment with selected action and observe new state, reward            
        new_state, reward, is_done = self.env.step(self, action, step_type) #Passing Self to allow enviornment to get Agent connected functions
        
        # Apply State Modifier Function, if defined, to next state prior to agent digest
        if self.env_state_mod_func != None:
            new_state = self.env_state_mod_func(self.env, **self.env_state_mod_params)
        
        return(state, action, reward, new_state, is_done, action_type, q_vals)

    def _best_action(self,state,step_type):
        q_values = self.Q1_nn(torch.tensor(state,dtype=torch.float32).to(self.device))
        # Q-values are only attached for Gradient calc during mini-batch training
        if (step_type == 'testing') or (not self.replay_memory.is_full()) or (step_type == 'validating'):
            q_values = q_values.detach()
        # Compute the best action based on the Q-values
        avaliable_actions = [self._act_env_to_nn[act] for act in self.get_avail_actions()]
        selected_qvals = torch.index_select(q_values, 0 ,torch.tensor(avaliable_actions).to(self.device))
        agrmax_index = torch.argmax(selected_qvals)
        best_action = avaliable_actions[agrmax_index]
        return best_action

    def _reset_parameters(self, nn):
        for _, module in nn.named_children():
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def reset_nn(self):
        self._reset_parameters(self.Q1_nn)
        self._reset_parameters(self.Q2_nn)
        print(f'{self.get_name()}: Q-Networks Reset Successfully')
            
    
    def export_Q_nn(self,filenamepath):
        torch.save(self.Q1_nn,filenamepath)
        print(f'{self.get_name()}: Q-Network Exported to file "{filenamepath}"')
    
    def import_Q_nn(self,filenamepath):
        self.Q1_nn = torch.load(filenamepath)
        print(f'{self.get_name()}: Q-Network Imported from file "{filenamepath}"\n{self.Q1_nn}')
    
    def _setup_early_stop(self, stop_metric):
        
        # Early Stoppage of Training due to flat validation
        dic =  {'Max':[-np.inf,['val_tot_r', 'val_avg_r', 'val_ror', 'tot_r','avg_r']], 
                'Min':[np.inf,['val_std_r','val_comm_cost','std_r']],
                'Loss': [None, ['Q1_loss', 'Q2_loss']]} 

        for key in dic.keys():
            if stop_metric in dic[key][1]:
                loss_type = key
                init_tgt_val = dic[key][0]
                return loss_type, init_tgt_val
    
    def get_training_episodic_data(self):
        return pd.DataFrame(self.training_episodic_data)  # Generate a DataFrame from stored step information  
    
    def get_testing_episodic_data(self):
        return pd.DataFrame(self.testing_episodic_data)  # Generate a DataFrame from stored step information
    
    def get_step_data(self):
        return pd.DataFrame(self.step_info)  # Generate a DataFrame from stored step information
    
    def get_metric(self):
        """
        Returns the value of the metric function
        """
        return self.metric_func(self.env, **self.metric_func_arg)
       
    def forward(self, x):
        # Forward pass through the network
        return self.Q1_nn.forward(x)
    
    def set_env_stat_modify_func(self,env_state_mod_func,env_state_mod_params:dict={}):
        #Prepossing of Environment State before Agent digest
        self.env_state_mod_func = env_state_mod_func
        self.env_state_mod_params = env_state_mod_params
    
    def save_config(self, filenamepath):
        agent_conifg = {
            'name':self.name,
            'reward_function':self.reward_function.__name__,
            'reward_params': self.reward_params, 
            'buffer_size':self.buffer_size,
            'batch_size': self.batch_size,
            'alpha': self.alpha,
            'gamma': self.gamma}
        
        nn_config = {
            'input_size': self.Q1_nn.input_size,
            'hidden_size': self.Q1_nn.hidden_size,
            'output_size': self.Q1_nn.output_size,
            'num_hidden_layers': self.Q1_nn.num_hidden_layers,
            'activation_function': self.Q1_nn.activation_function_name,
            'opt_wgt_dcy': self.Q1_nn.opt_wgt_dcy,
            'dropout_rate': self.Q1_nn.dropout_rate
        }
        
        config = agent_conifg | nn_config
        
        with open(filenamepath, 'w') as f:
            json.dump(config, f)

Experience = namedtuple('Experience', field_names=['state',
                                                   'action',
                                                   'reward',
                                                   'is_done',
                                                   'new_state'])
           
class ExperienceBuffer:
    def __init__(self, capacity: int, device) -> None:
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
        self.device = device
    
    def __len__(self):
        return len(self.buffer)
    
    def append(self, experience: tuple):
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        sampled_experiences = [self.buffer[idx] for idx in indices]
        
        # Extract individual fields from sampled experiences
        states = torch.stack([torch.tensor(exp.state,dtype=torch.float32) for exp in sampled_experiences]).to(self.device)
        actions = torch.tensor([exp.action for exp in sampled_experiences], dtype=torch.long).to(self.device)
        rewards = torch.tensor([exp.reward for exp in sampled_experiences], dtype=torch.float32).to(self.device)
        dones = torch.tensor([exp.is_done for exp in sampled_experiences], dtype=torch.float32).to(self.device)
        next_states = torch.stack([torch.tensor(exp.new_state,dtype=torch.float32) for exp in sampled_experiences]).to(self.device)
        
        return states, actions, rewards, dones, next_states
    
    def reset(self):
        self.buffer.clear()
    
    def is_full(self):
        return len(self.buffer) == self.buffer.maxlen
    

class Q_Network(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 output_size: int,
                 activation_function,
                 num_hidden_layers: int,   
                 device,
                 opt_lr: float = 0.001,
                 opt_wgt_dcy: float = 0.0,
                 dropout_rate: float = 0.25):
        
        super(Q_Network, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        if isinstance(activation_function, str):
            self.activation_function_name = activation_function
            self.activation_function = self.get_activation_function(activation_function)
        else:
            self.activation_function = activation_function
            self.activation_function_name = f"{self.activation_function.__module__}.{self.activation_function.__class__.__name__}"
        
        self.num_hidden_layers = num_hidden_layers
        self.device = device
        self.opt_lr = opt_lr
        self.opt_wgt_dcy = opt_wgt_dcy
        self.dropout_rate = dropout_rate
        
        ## Q Network Layers
        layers = []
        
        ## Input Layer
        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(activation_function)

        ## Hidden Layers
        for _ in range(self.num_hidden_layers):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(self.activation_function)
            layers.append(nn.Dropout(self.dropout_rate))

        ## Output Layer
        layers.append(nn.Linear(self.hidden_size, self.output_size))

        # Apply Kaiming Normal initialization to the linear layers' weights
        for i in range(0, len(layers), 3):
            if isinstance(layers[i], nn.Linear):
                nn.init.kaiming_normal_(layers[i].weight)
                
        ## Create Q Network
        self.Q_nn = nn.Sequential(*layers)
        self.Q_nn.to(self.device)
        
        # Initialize Optimizer
        self.optimizer = optim.Adam(self.Q_nn.parameters(), lr=self.opt_lr, 
                                    weight_decay=self.opt_wgt_dcy)

 
    def forward(self, x: torch.tensor):
        # Ensure the input is moved to the correct device
        x = x.to(self.device)
        # Forward pass through the network
        return self.Q_nn(x)
    
    @staticmethod
    def get_activation_function(activation_function_name):
        module_name, class_name = activation_function_name.rsplit('.', 1)
        module = importlib.import_module(module_name)
        activation_class = getattr(module, class_name)
        return activation_class()        
       
class EarlyStopping:
    # From https://github.com/thuml/Time-Series-Library/blob/main/tutorial/TimesNet_tutorial.ipynb
    def __init__(self, patience:int=7, verbose=False, delta=0, min_training:int = None):
        self.patience = patience # how many times will you tolerate for loss not being on decrease
        self.verbose = verbose  # whether to print tip info
        self.patience_counter = 0 # now how many times loss not on decrease
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.min_trn = min_training # Minimum number of training epochs
        self.call_counter = 0  # Number of Times EarlyStopping was called
        self.model_save_idx = None  # Training Epoch where last mode was saved
        self.target = None # Current Value of Target Metric
        
        if min_training is None or min_training == 0:
            self.min_training_done = True
        elif min_training < 0:
            raise ValueError('Positive Integer expected for min_training')
        else:
            self.min_training_done = False
        
    def __call__(self, val_loss, model, path) -> str:
        self.call_counter +=1
        
        if not self.min_training_done and self.call_counter == self.min_trn:
            self.min_training_done = True
            self.save_checkpoint
            
            
        score = -val_loss
        
        if self.min_training_done:
            if self.best_score is None:
                self.best_score = score
                msg = (f' -> Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
                self.save_checkpoint(val_loss, model, path)

            # meaning: current score is not 'delta' better than best_score, representing that 
            # further training may not bring remarkable improvement in loss. 
            elif score < self.best_score + self.delta:  
                self.patience_counter += 1
                msg = (f' -> EarlyStopping counter: {self.patience_counter} out of {self.patience}')
                # 'No Improvement' times become higher than patience --> Stop Further Training
                if self.patience_counter >= self.patience:
                    self.early_stop = True

            else: #model's loss is still on decrease, save the now best model and go on training
                self.best_score = score
                msg = (f' -> Validation loss decreased ({self.val_loss_min:.4f} --> {val_loss:.4f}).  Saving model ...')
                self.save_checkpoint(val_loss, model, path)
                self.patience_counter = 0
        else:
            msg = (f' -> Minimum training phase {self.call_counter} of {self.min_trn}')
        
        return msg
    
    def new_target(self,target):
        # val_loss input to __call__ is based on a new target, requiring reset of counter
        # best_score, and early stoppage flag
        if self.min_training_done:
            
            if self.target is not None:
                formatted_target = f'{self.target:.2f}'
            else:
                formatted_target = 'N/A'  

            self.patience_counter = 0 
            self.best_score = None
            self.val_loss_min = np.Inf
            self.early_stop = False
            msg = (f' -> New Target Established ({formatted_target} -> {target:.2f}) - Reset Early Stopping')
            self.target = target
        else:
            msg = (f' -> Minimum training phase {self.call_counter} of {self.min_trn}')
            
        return msg
        

    def save_checkpoint(self, val_loss, model, path):
    ### used for saving the current best model
         
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.model_save_idx = self.call_counter
        self.val_loss_min = val_loss
    
    def load_checkpoint(self, model, path):
    ### Used for loading the saved model from the checkpoint file
        model.load_state_dict(torch.load(path))
        print('Last Checkpoint loaded')


