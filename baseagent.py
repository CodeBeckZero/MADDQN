import pandas as pd
from abc import ABC, abstractmethod

class baseagent(ABC):
    def __init__(self, name, reward_function, environment, sub_agents = None):
        self.name = name
        self.reward_function = reward_function
        self.env = environment
        self.sub_agents = None
       
    @abstractmethod
    def train(self, 
              start_idx:int, 
              end_idx:int, 
              training_epsidoes:int,
              epsilon_decya_func, 
              initial_epsilon, 
              final_epsilon):
        pass
    
    @abstractmethod
    def test(self, 
             start_idx:int, 
             end_idx:int,):
        pass
    
    @abstractmethod
    def _act(self,step_type):
        pass

    def get_name(self):
        return self.name

    def get_reward_function(self):
        return self.reward_function(self.env)

    def get_avail_actions(self):
        return self.env.available_actions

    def get_step_data(self):
        return pd.DataFrame(self.step_info)  # Generate a DataFrame from stored step information            
            
    def get_training_episodic_data(self):
        return pd.DataFrame(self.training_episodic_data)  # Generate a DataFrame from stored step information  
    
    def get_testing_episodic_data(self):
        return pd.DataFrame(self.testing_episodic_data)  # Generate a DataFrame from stored step information 
     
    def set_environment(self, enviornment):
        self.env = enviornment
    
    def set_reward_func(self, reward_function):
        self.reward_function = reward_function

