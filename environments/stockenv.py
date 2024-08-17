import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import csv

class ContinuousOHLCVEnv(gym.Env):
       
    def __init__(self, name: str, ohlcv_data: np.array, stock_price_data: np.array, commission_rate: float = 0.00, initial_cash: float = 100000.0) -> None:
        
        # Env Name
        self.name = name
        self.window_size = self._data_window_check(ohlcv_data,stock_price_data)
                           
        # Define Action Space
        self.actions = ('S','H','B') # Sell, Hold, Buy
        self.action_space = spaces.Discrete(3)

        # Define the observation space for OHLCV data
        num_features = ohlcv_data.shape[1]  # Assuming OHLCV columns are Open, High, Low, Close, Volume
        feature_min = np.array(-np.inf * num_features)  
        feature_max = np.array(np.inf * num_features)  
        self.observation_space = spaces.Box(low=feature_min, high=feature_max, dtype=np.float32)
        self.action_functions = {"B": self._buy,
                                 "H": self._hold,
                                 "S": self._sell}
        
        # Init Input Data
        self.ohlcv_raw_data = ohlcv_data
        self.stock_price_data = stock_price_data
                
        # Init Portfolio
        self.current_state = None
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.position = None
        self.n_idx_position = None
        self.n_idx_no_position = None
        self.purchase_price = None
        self.value = None
        self.previous_action = None

       # Init Agent Tuple
        self.agents = set()
        self.agent_sequence = []
        
        
        # Init Final Agent 
        self.DECISION_AGENT = None

        # Init Indices
        self.start_idx = 0
        self.max_idx = ohlcv_data.shape[0] -1
        self.finish_idx = None
        self.update_idx(self.start_idx, self.max_idx)
        self.reset()

    def reset(self):
        # Reset State 
        self.current_step = self.start_idx
        self.position = 0
        self.n_idx_position = 0
        self.n_idx_no_position = 0  
        self.purchase_price = self.stock_price_data[self.current_step]
        self.value = 1
        self.done = False
        self.previous_action = 1 # Need to address as env_to_agent state is handled in Agent...'H':1
        
        # Update State
        vars_to_state = np.array((self.position, self.n_idx_position, self.purchase_price, self.value, self.previous_action, self.n_idx_no_position))
        self.current_state = np.concatenate((self.ohlcv_raw_data[self.current_step],vars_to_state)) 
        

        # Reset Portfolio
        self.cash_in_hand = self.initial_cash
        self.last_commission_cost = 0
        self.total_commission_cost = 0
        self.stock_holding = int(0)
        self.stock_price = self.stock_price_data[self.current_step]
        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)

        # Reset Logging
        self.step_info = []  # Initialize an empty list to store step information

        # Reset Sequencing
        self.agent_sequence = list(self.agents)
        
        # Reset Action Space
        if self.cash_in_hand > self.stock_price:
            self.available_actions = ('H','B')
        else:
            self.available_actions = ('H')
            print(f"{self.get_name()}: Warning - Not enough cash to buy single stock")
        
    def add_agent(self, agent_name: str):
        if agent_name not in self.agents:
            self.agents.add(agent_name)
            print(f'{self.get_name()} ENV: Agent {agent_name} added')
        else:
            print(f'{self.get_name()} ENV: Agent {agent_name} already exists')

    def remove_agent(self, agent_name: str):
        if agent_name in self.agents:
            self.agents.remove(agent_name)
            print(f'{self.get_name()} ENV: Agent {agent_name} removed')
        else:
            print(f'{self.get_name()} ENV: Agent {agent_name} not found')

    def set_decision_agent(self,agent_name:str):
        assert agent_name in self.agents, f'Invalid Agent Name: "{agent_name}" not in agent list'
        self.DECISION_AGENT = agent_name
        print(f'{self.get_name()} ENV: Agent {agent_name} assigned as decision agent')
        return
  
    def step(self, agent_instance, action, step_type):
        agent_name = agent_instance.get_name()
        # Checking valid action
        
        try:
            if action not in self.available_actions:
                raise ValueError(f'Action {action} not in {self.available_actions}')
            
            if agent_name not in self.agents:
                raise ValueError(f'Invalid agent "{agent_name}"')
            
            if step_type == "testing" or step_type == 'validating':
                if not self.DECISION_AGENT:
                    raise ValueError('Decision Agent not assigned')
                
                if not (((agent_name == self.DECISION_AGENT) and (agent_name in self.agent_sequence) and (len(self.agent_sequence) == 1)) or \
                        ((agent_name != self.DECISION_AGENT) and (agent_name in self.agent_sequence) and (len(self.agent_sequence) >= 1))):
                    raise ValueError('Invalid Sequence: All Sub-Agents need to act before Final agent')
        except ValueError as e:
            print(f'Error: {e}')
        
        # Assigining Hold Action as previous action for first iteration
        if self.current_step == self.start_idx:
            self.previous_action = 0
        else:
            last_action = self.step_info[-1]['Env Action']
            self.previous_action = agent_instance._act_env_to_nn[last_action]
            
        step_data = {
            'Step': self.current_step - self.start_idx + 1,
            'idx': self.current_step,
            'Portfolio Value': self.total_portfolio_value, 
            'Cash': self.cash_in_hand,
            'Stock Value': self.stock_price * self.stock_holding, 
            'Stock Holdings': self.stock_holding,
            'Stock Price': self.stock_price,
            'State': self.current_state,
            "Available Actions": self.available_actions,
            "Env Action": action
        }

        # Perform action based on action type
        if action in self.action_functions:
            self.action_functions[action](agent_instance)
            self.agent_sequence.remove(agent_name)
        else:
            # Handle unexpected actions here
            raise ValueError (f'Action {action} is not a valid action')

        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)

        if (self.total_portfolio_value < self.stock_price) and (int(self.position) == 0):
            self.done = True
        
        # Completed Step
        if  (not self.agent_sequence and (step_type == 'testing' or step_type == 'training' or (step_type == 'validating'))):

            step_data.update({'New Portfolio Value': self.total_portfolio_value,
                            'New Cash': self.cash_in_hand,
                            'New Stock Value': self.stock_price * self.stock_holding, 
                            'New Stock Holdings': self.stock_holding,
                            "New Commission Cost": self.last_commission_cost,
                            'Total Commission Cost': self.total_commission_cost})
            self.step_info.append(step_data)
            reward = self.get_reward(action, step_type, agent_instance, agent_name)
            self.current_step += 1
            self.stock_price = self.stock_price_data[self.current_step]
            self.agent_sequence = list(self.agents)
             # Purchase price is passed to state and potentially nomarlized/scaled, setting to 0 caused large numbers in agents NN
            if self.position == 0:
                self.purchase_price = self.stock_price_data[self.current_step]
        else: # Subagent Step (no change in enviornment but reuqire to give accurate reward - reward function may depend on calculation in if..help)
            reward = self.get_reward(action, step_type, agent_instance, agent_name) #
        
        if self.current_step == (self.finish_idx-1):

            if int(self.position) ==  1:
                self.available_actions = ('S',)
            else:
                self.available_actions = ('H',)



        # Update State
        self.value = self.total_portfolio_value / self.initial_cash
        vars_to_state = np.array((self.position, self.n_idx_position, self.purchase_price, self.value, self.previous_action, self.n_idx_no_position))
        self.current_state = np.concatenate((self.ohlcv_raw_data[self.current_step],vars_to_state)) 

        next_observation = self.get_observation()

        if not self.done:
            self.done = self.current_step == self.finish_idx

        return next_observation, reward, self.done
            
    def _buy(self,agent_instance):
        if agent_instance.get_name() == self.DECISION_AGENT: # could be a problem for multiagent (only want decision agent to change balances)
            self.num_stocks_buy = int(np.floor(self.cash_in_hand/self.stock_price)) # Buy Maximum allowed (Current Method)
            self.last_commission_cost = self.num_stocks_buy * self.stock_price * self.commission_rate
            self.total_commission_cost += self.last_commission_cost
            self.cash_in_hand -= self.num_stocks_buy * self.stock_price + self.last_commission_cost
            self.stock_holding = self.num_stocks_buy
            self.position = 1
            self.n_idx_position = 1
            self.n_idx_no_position = 0 
            self.purchase_price = self.stock_price
            self.available_actions = ('S','H')
  
    def _hold(self,agent_instance):
        if agent_instance.get_name() == self.DECISION_AGENT: # could be a problem for multiagent (only want decision agent to change balances)
            if self.position == 1:
                self.n_idx_position += 1
            elif self.position == 0:
                self.n_idx_no_position += 1
            self.last_commission_cost = 0
                    
    def _sell(self, agent_instance):
        if agent_instance.get_name() == self.DECISION_AGENT: # could be a problem for multiagent (only want decision agent to change balances)
            self.num_stocks_sell = self.stock_holding # Sell all stocks (Current Mehtod)
            self.last_commission_cost = self.num_stocks_sell * self.stock_price * self.commission_rate
            self.total_commission_cost += self.last_commission_cost
            self.cash_in_hand += self.num_stocks_sell * self.stock_price - self.last_commission_cost
            self.stock_holding -= self.num_stocks_sell
            self.position = 0
            self.n_idx_position = 0
            self.purchase_price = self.stock_price
            self.available_actions = ('H','B')

   
    def update_idx(self,start_idx:int,final_idx:int):
        assert start_idx < final_idx, f'invalid: start_idx: {start_idx} < final_idx {final_idx}'
        self.start_idx = start_idx
        self.finish_idx = final_idx
        self.current_step = self.start_idx

    def get_observation(self):
        return(self.current_state)

    def get_step_data(self):
        return pd.DataFrame(self.step_info)  # Generate a DataFrame from stored step information

    def csv_export_step_data(self,csv_filename):
        # Column Names for CSV
        col_names = self.step_info[0].keys()
        
        # Writing CSV
        with open(csv_filename, mode='w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=col_names)
    
            # Write the header row
            writer.writeheader()
            
            # Write the data rows
            for row in self.step_info:
                writer.writerow(row)

        print(f"{self.get_name()}: Step data exported to {csv_filename}")
    
    def get_name(self):
        return self.name
    
    def _data_window_check(self, env_state_input, stock_price_input):
        """
        Function verifies that environmental states and the stock price input have the same length.

        Parameters:
        env_state_input (np.array): Environmental states input data.
        stock_price_input (np.array): Stock price input data.

        Rasies:
        ValueError: If the window lengths do not match.
        """
        num_env_state = env_state_input.shape[0]
        num_stock_price = stock_price_input.shape[0]
        if num_env_state != num_stock_price:
            raise ValueError("Non-matching window lengths for environmental states and stock prices")      
        

    def get_reward(self, action, step_type, agent_instance, agent_name): # Included action for possible expansion of reward type based on action
        if step_type == 'testing':
            reward = agent_instance.get_metric()

        elif step_type == 'training':
            reward = agent_instance.get_reward_function()
        elif step_type == 'validating':
            reward = agent_instance.get_metric()

        else:
            # Handle unexpected step types here
            raise ValueError(f'Improper step type: {step_type}')
        return reward

        