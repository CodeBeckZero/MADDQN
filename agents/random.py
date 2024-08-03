"""Module providing providing printout of blanks for feedback text"""
import sys
import random
import numpy as np
from agents.baseagent import BaseAgent
import pandas as pd

class RandomAgent(BaseAgent):
    """
    Random agent executes random action on enivornment    
    """
    def __init__(self, name: str, environment, reward_function, reward_params = None, sub_agents = None):
        
        # Call the initialization of both parent classes
        BaseAgent.__init__(self, name, reward_function, environment, reward_params, sub_agents)

        self.testing_episodic_data = None
        self.step_info = None
        
        # Dictionary to translate actions between NN and Env
        self._act_nn_to_env = {0: 'S', 1: 'H', 2 :'B'}
        self._act_env_to_nn = {'S': 0,'H': 1, 'B': 2 }

    def test(self, start_idx:int, end_idx:int, testing_episodes=1, metric_func = None, metric_func_arg = {}):
        
        self.metric_func = metric_func
        self.metric_func_arg = metric_func_arg
        
        print(f'{self.get_name()}: Testing Initialized on {self.env.get_name()}[{start_idx}:{end_idx}]')
        episodic_data = []

        self.env.update_idx(start_idx,end_idx)
        for episode_num in range(1, testing_episodes+1):

            tot_reward, mean_reward, std_reward, actions = self._play_episode('testing')
            epi_data = {"tst_ep": episode_num, 
                        "tot_r": tot_reward,
                        "avg_r": mean_reward,
                        "std_r": std_reward,
                        'tst_actions': actions}
            episodic_data.append(epi_data)
            
            
            # Print a line with blank spaces to clear the existing content
            sys.stdout.write('\r' + ' ' * 200)  # Assuming 150 characters wide terminal

            # Print Update
            print(
                f'\r{self.get_name()}: EP {episode_num} of {testing_episodes} Finished ' +
                f'-> ∑R = {tot_reward:.2f}, μR = {mean_reward:.2f}, σR = {std_reward:.2f}', end="", flush=False)

        self.testing_episodic_data = episodic_data
        print(f'\n{self.get_name()}: Testing Complete on {self.env.get_name()}[{start_idx}:{end_idx}]')              
    
    def _play_episode(self, step_type):
            rewards = np.array([])
            actions = []
            self.step_info = []
            self.env.reset()
            is_done = False
            total_steps = 0

            while not is_done:
                if step_type == "testing": # Always use best action
                    _ , action, reward, _ , end , action_type = self._act(step_type)

                else:
                    raise ValueError(f'Invalid step_type: {step_type}')

                step_data = {f'{self.name} Action': action, 
                            f'{self.name} Action Type': action_type,
                            f'{self.name} Reward': reward}
                self.step_info.append(step_data)
                is_done = end
                
                total_steps += 1
                rewards = np.append(rewards, reward)
                actions = np.append(actions, action)
                
            return np.sum(rewards), np.mean(rewards), np.std(rewards), actions
   
    def _act(self, step_type):
        
        state = self.env.get_observation()
        action = random.choice(self.get_avail_actions())
        action_type = "Random"    
        new_state, reward, is_done = self.env.step(self, action, step_type) #Passing Self to allow enviornment to get Agent connected functions
        return(state, action, reward, new_state, is_done, action_type)
    
    def get_testing_episodic_data(self):
        return pd.DataFrame(self.testing_episodic_data)  # Generate a DataFrame from stored step information
    
    def get_step_data(self):
        return pd.DataFrame(self.step_info)  # Generate a DataFrame from stored step information
    
    def get_metric(self):
        """
        Returns the value of the metric function
        """
        return self.metric_func(self.env, **self.metric_func_arg)      