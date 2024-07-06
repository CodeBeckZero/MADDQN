import os
from neuralforecast.core import NeuralForecast
from sklearn import preprocessing
import pandas as pd
import numpy as np


class Star_TNsubagents:
    def __init__(self, uni_data, decision_agent, sub_agents, tn_type):
        self.decision_agent = decision_agent
        self.sub_agents = sub_agents
        self._tn_type = tn_type
        self.data = uni_data
        self.scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    def upload_subagent(self, agent, loc_trained_model):
        self.sub_agents.append(agent)
        agent.import_Q_nn(loc_trained_model)

    def upload_timesnet_model(self,loc_trained_model):
        # Ensure the directory and file exist
        if os.path.exists(loc_trained_model):
            self.nf = NeuralForecast.load(path=loc_trained_model)
        else:
            raise FileNotFoundError(f"Model path {loc_trained_model} does not exist.")
        
    def upload_csv(self,csv_loc):
        self.env_csv = pd.read_csv(csv_loc)
        self.env_csv['date'] = pd.to_datetime(self.env_csv['date'])
    
    def _get_environment_state(self, env):
        raw_state, position = env.get_observation()
        cur_state_idx = env.current_step
        columns = ['open', 'high', 'low', 'close', 'volume']

        # Check if the environmental state is in the form of OHLCV data
        if raw_state.shape[1] != len(columns):
            raise ValueError('Environmental State is not in the form of OHLCV data')

        # Get window size (number rows of data) 
        state_window_size = raw_state.shape[0]
        
        # Create a dictionary to store environmental state by column
        env_state_by_col_dic = {col: raw_state[:, idx] for idx, col in enumerate(columns)}
        
            
        return raw_state, position, cur_state_idx, state_window_size,  env_state_by_col_dic

    def _gen_subagent_state(self, env, tn_type):
        """
        Process the environment data.

        Args:
        - env: Environment object.

        Returns:
        - agent_state: Processed agent state.
        """
        # Get observation from the environment
        env_state_by_col_dic, position, cur_state_idx, state_window_size = self._get_environment_state(env)
         
        if tn_type == 'model' and hasattr(self, 'nf'):
            # Predict model output and extend 'close' column
            model_output = self.nf.predict(self.data[env.name]['rw_long_raw_price'][cur_state_idx])['timesnet'].to_numpy()
            env_state_by_col_dic['close'] = np.concatenate([env_state_by_col_dic['close'], model_output])
        elif tn_type == 'csv':
            model_output_date = self.data[env.name]['rw_long_raw_price'][cur_state_idx]['ds'].iloc[-1]
            filtered_date = self.env_csv['date'] == model_output_date
            desired_columns = ['1d', '2d', '3d', '4d', '5d']
            model_output = self.env_csv.loc[filtered_date, desired_columns].values.flatten()
            env_state_by_col_dic['close'] = np.concatenate([env_state_by_col_dic['close'], model_output])
        else:
            raise AttributeError("NeuralForecast Model was not loaded")
        
        
        # Normalize each column separately
        columns = env_state_by_col_dic.keys()
        for col in columns:
            env_state_by_col_dic[col] = self.scaler.fit_transform(env_state_by_col_dic[col].reshape(-1, 1)).flatten()

        # Extract normalized prediction and current state
        norm_predict = env_state_by_col_dic['close'][-5:].tolist()
        norm_current_state = [env_state_by_col_dic[col][state_window_size - 1] for col in columns]

        # Append position to the normalized current state
        norm_current_state.append(position)

        # Concatenate normalized current state with normalized prediction
        subagent_state = norm_current_state + norm_predict

        return subagent_state
 
    def gen_agent_state(self, env):
        # Get observation from the environment
        env_state_by_col_dic, position, cur_state_idx, state_window_size = self._get_environment_state(env)
        # Get sub_agent state
        sub_agent_state = self._gen_subagent_state(env,self.tn_type)

        sub_agent_states = []
        for agent in self.sub_agents:
            sub_agent_state = agent.forward(sub_agent_state).flatten().tolist
            sub_agent_states.append(sub_agent_state)

        # Normalize each column separately
        
        for col in columns:
            env_state_by_col_dic[col] = self.scaler.fit_transform(env_state_by_col_dic[col].reshape(-1, 1)).flatten()

        # Extract normalized prediction and current state
        norm_predict = env_state_by_col_dic['close'][-5:].tolist()
        norm_current_state = [env_state_by_col_dic[col][std_cur_state_idx - 1] for col in columns]

        # Append position to the normalized current state
        norm_current_state.append(position)

        # Concatenate normalized current state with normalized prediction
        agent_state = norm_current_state + norm_predict

        return agent_state

        


        
        
        
        
        