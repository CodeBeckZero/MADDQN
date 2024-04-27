import sys  
import numpy as np
from agents.baseagent import BaseAgent
import pandas as pd

class ManualAgent(BaseAgent):
    def __init__(self,name: str, environment, reward_function, sub_agents = None):
        BaseAgent.__init__(self, name, reward_function, environment, sub_agents)
        self.training_episodic_data = None
        self.testing_episodic_data = None
        self.step_info = None
    
    def input_training_sequence(self, training_seq):
        self.training_seq = training_seq
    
    def input_testing_sequence(self, testing_seq):
        self.testing_seq = testing_seq
    
    def train(self, start_idx:int, end_idx:int, training_episodes=1):

            trn_act_len = len(self.training_seq)
            trn_interval_len = (end_idx-start_idx)

            if trn_act_len < trn_interval_len :
                raise ValueError(f'{self.get_name()}:' +
                                 f'Training Action Sequnce len {trn_act_len} '+
                                 f'is smaller than training interval {trn_interval_len}')
            
            print(f'{self.get_name()}: Training Initialized on {self.env.get_name()}[{start_idx}:{end_idx}]')
            
            episodic_data = []

            self.env.update_idx(start_idx,end_idx)
            
            for episode_num in range(1, training_episodes+1):
                tot_reward, mean_reward, std_reward = self._play_episode('training')
                epi_data = {"trn_ep": episode_num, 
                        "tot_r": tot_reward,
                        "avg_r": mean_reward,
                        "std_r": std_reward}
            
                episodic_data.append(epi_data)
            
                # Print a line with blank spaces to clear the existing content
                sys.stdout.write('\r' + ' ' * 200)  # Assuming 200 characters wide terminal

                # Print Update
                print(
                    f'\r{self.get_name()}: EP {episode_num} of {training_episodes} Finished ' +
                    f'-> ∑R = {tot_reward:.2f}, μR = {mean_reward:.2f}, σR = {std_reward:.2f} ', end="", flush=False)
            
            self.training_episodic_data = episodic_data
            print(f'\n{self.get_name()}: Training Complete on {self.env.get_name()}[{start_idx}:{end_idx}]')     

    
    def test(self, start_idx:int, end_idx:int, testing_episodes=1):
        
        tst_act_len = len(self.testing_seq)
        tst_interval_len = (end_idx-start_idx)

        if tst_act_len < tst_interval_len :
            raise ValueError(f'{self.get_name()}:' +
                                f'Testing Action Sequnce len {tst_act_len} '+
                                f'is smaller than tst interval {tst_interval_len}')
        
        print(f'{self.get_name()}: Testing Initialized on {self.env.get_name()}[{start_idx}:{end_idx}]')
                
        episodic_data = []

        self.env.update_idx(start_idx,end_idx)
        
        for episode_num in range(1, testing_episodes+1):
            tot_reward, mean_reward, std_reward = self._play_episode('testing')
            epi_data = {"trn_ep": episode_num, 
                    "tot_r": tot_reward,
                    "avg_r": mean_reward,
                    "std_r": std_reward}
        
            episodic_data.append(epi_data)
        
            # Print a line with blank spaces to clear the existing content
            sys.stdout.write('\r' + ' ' * 200)  # Assuming 200 characters wide terminal

            # Print Update
            print(f'\r{self.get_name()}: EP {episode_num} of {testing_episodes} Finished ' +
                  f'-> ∑R = {tot_reward:.2f}, μR = {mean_reward:.2f}, σR = {std_reward:.2f} ', end="", flush=False)
        
        self.testing_episodic_data = episodic_data
        print(f'\n{self.get_name()}: Testing Complete on {self.env.get_name()}[{start_idx}:{end_idx}]')              
    
            
    
    def _play_episode(self, step_type):
        rewards = np.array([])
        self.step_info = []
        self.env.reset()
        is_done = False
        self.act_idx = 0
        while not is_done:
            
            if step_type == "testing" or step_type == 'training': 
                _ , action, reward, _ , end , action_type = self._act(step_type)
            
            else:
                raise ValueError(f'Invalid step_type: {step_type}')

            step_data = {f'{self.name} Action': action, 
                        f'{self.name} Action Type': action_type,
                        f'{self.name} Reward': reward}
            self.step_info.append(step_data)
            is_done = end
            
            
            self.act_idx += 1
            rewards = np.append(rewards, reward)
            
        return np.sum(rewards), np.mean(rewards), np.std(rewards)

    def _act(self,step_type):
        
        state = self.env.get_observation()
        available_actions = self.get_avail_actions()
        
        if step_type == 'training':
            action = self.training_seq[self.act_idx]
        elif step_type == 'testing':
            action = self.testing_seq[self.act_idx]
        
        action_type ='Manual'
        if action in available_actions:
            new_state, reward, is_done = self.env.step(self, action, step_type) #Passing Self to allow enviornment to get Agent connected functions
        else:
            raise ValueError(f'Action {action} not in available actions {available_actions}')
        
        return(state, action, reward, new_state, is_done, action_type)
    
    def get_training_episodic_data(self):
        return pd.DataFrame(self.training_episodic_data)  # Generate a DataFrame from stored step information  
    
    def get_testing_episodic_data(self):
        return pd.DataFrame(self.testing_episodic_data)  # Generate a DataFrame from stored step information
    
    def get_step_data(self):
        return pd.DataFrame(self.step_info)  # Generate a DataFrame from stored step information  
    
