import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import csv

class ContinuousOHLCVEnv(gym.Env):
       
    def __init__(self, name: str, ohlcv_data: np.array, stock_price_data: np.array, commission_rate: float = 0.00, initial_cash: float = 100000.0) -> None:
        
        # Env Name
        self.name = name
                           
        # Define Action Space
        self.actions = ('S','H','B') # Sell, Hold, Buy
        self.action_space = spaces.Discrete(3)

        # Define the observation space for OHLCV data
        num_features = ohlcv_data.shape[1]  # Assuming OHLCV columns are Open, High, Low, Close, Volume
        feature_min = np.array([-20.0] * num_features)  # Assuming Normalization over mean of 0
        feature_max = np.array([20.0] * num_features)  # Assuming Normalization over mean of 0
        self.observation_space = spaces.Box(low=feature_min, high=feature_max, dtype=np.float32)
        
        # Init Input Data
        self.ohlcv_raw_data = ohlcv_data
        self.stock_price_data = stock_price_data
                
        # Init Portfolio
        self.initial_cash = initial_cash
        self.commission_rate = commission_rate
        self.position = None
        self.current_state = None

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
        self.current_state = list(self.ohlcv_raw_data[self.current_step]) + [self.position]

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
            if step_type == "testing":
                if not self.DECISION_AGENT:
                    raise ValueError('Decision Agent not assigned')
                
                if not (((agent_name == self.DECISION_AGENT) and (agent_name in self.agent_sequence) and (len(self.agent_sequence) == 1)) or \
                        ((agent_name != self.DECISION_AGENT) and (agent_name in self.agent_sequence) and (len(self.agent_sequence) >= 1))):
                    raise ValueError('Invalid Sequence: All Sub-Agents need to act before Final agent')
        except ValueError as e:
            print(f'Error: {e}')
     

        step_data = {
            'Step': self.current_step - self.start_idx + 1,
            'idx': self.current_step,
            'Portfolio Value': self.total_portfolio_value, 
            'Cash': self.cash_in_hand,
            'Stock Value': self.stock_price * self.stock_holding, 
            'Stock Holdings': self.stock_holding,
            'Stock Price': self.stock_price,
            "Last Commission Cost": self.last_commission_cost,
            'Total Commision Cost': self.total_commission_cost,
            'State': self.current_state,
            "Available Actions": self.available_actions,
            "Env Action": action
        }


        if action == 'S': # Sell
            reward = self._sell(agent_instance)
            if step_type == 'testing':
                self.agent_sequence.remove(agent_name)

        elif action == 'H': # Hold
            reward = self._hold(agent_instance)
            if step_type == 'testing':
                self.agent_sequence.remove(agent_name)
                
        elif action == "B": # Buy
            reward = self._buy(agent_instance)
            if step_type == 'testing':
                self.agent_sequence.remove(agent_name)


                        
        self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)

        
        if  ((not self.agent_sequence and step_type == 'testing') or (step_type == 'training')):
            self.total_portfolio_value = self.cash_in_hand + (self.stock_holding * self.stock_price)
            self.current_step += 1
            self.stock_price = self.stock_price_data[self.current_step]
            self.step_info.append(step_data)
            self.agent_sequence = list(self.agents)
        
        if self.current_step == (self.finish_idx-1):

            if int(self.stock_holding) > 0:
                self.available_actions = ('S',)
            else:
                self.available_actions = ('H',)       

        self.current_state = list(self.ohlcv_raw_data[self.current_step]) + [self.position]

        next_observation = self.get_observation()
        
        done = self.current_step == self.finish_idx
         
        return next_observation, reward, done
            
    def _buy(self,agent_instance):
        if agent_instance.get_name() == self.DECISION_AGENT:

            self.num_stocks_buy = int(np.floor(self.cash_in_hand/self.stock_price)) # Buy Maximum allowed (Current Method)
            self.last_commission_cost = self.num_stocks_buy * self.stock_price * self.commission_rate
            self.total_commission_cost += self.last_commission_cost
            self.cash_in_hand -= self.num_stocks_buy * self.stock_price - self.last_commission_cost
            self.stock_holding = self.num_stocks_buy
            self.position = 1
            self.available_actions = ('S','H')

        return agent_instance.get_reward_function()
    
    def _hold(self,agent_instance):
        if agent_instance.get_name() == self.DECISION_AGENT:

            self.last_commission_cost = 0
            
        return agent_instance.get_reward_function()
        
        
    def _sell(self, agent_instance):
        if agent_instance.get_name() == self.DECISION_AGENT:

            self.num_stocks_sell = self.stock_holding # Sell all stocks (Current Mehtod)
            self.last_commission_cost = self.num_stocks_sell * self.stock_price * self.commission_rate
            self.total_commission_cost += self.last_commission_cost
            self.cash_in_hand += self.num_stocks_sell * self.stock_price - self.last_commission_cost
            self.stock_holding -= self.num_stocks_sell
            self.position = 0
            self.available_actions = ('H','B')

        return agent_instance.get_reward_function()
    
    
    def update_idx(self,start_idx:int,final_idx:int):
        assert start_idx < final_idx, f'invalid: start_idx: {start_idx} < final_idx {final_idx}'
        self.start_idx = start_idx
        self.finish_idx = final_idx
        self.current_step = self.start_idx

    def get_observation(self):
        return(tuple(self.current_state))

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