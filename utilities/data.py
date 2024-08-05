import numpy as np
import torch
from torch.utils.data import Dataset
from neuralforecast.core import NeuralForecast
from sklearn import preprocessing
import pandas as pd
import copy
import os
from typing import Callable, Dict
from scipy.stats import rankdata



    
class RunningWindowDataset(Dataset):
    def __init__(self, data, window_size:int,data_type = 'DataFrame'):
        """
        Initialize the RunningWindowDataset.

        Parameters:
            data (numpy.ndarray): The input data.
            window_size (int): The size of the sliding window.
        """
        match data_type:
            case 'DataFrame':
                self.data = self._running_window_setup(pd.DataFrame(data), window_size)
            
            case 'array':
                self.data = self._running_window_setup(np.array(data), window_size)
            case _:
                raise ValueError(f'Invalid data_type input: {data_type}')

        self.window_size = window_size
        self.shape = (len(self.data),self.data[0].shape[0],self.data[0].shape[1])

    def __len__(self):
        """
        Return the length of the dataset after considering the sliding window.
        """
        return len(self.data)

    def __getitem__(self, index):
        """
        Retrieve a window of data from the dataset.

        Parameters:
            index (int, slice): The index or slice to retrieve.

        Returns:
            numpy.ndarray: The window of data.
        """
        if isinstance(index, int):
            # If index is an integer, get the window of data at that index
            window_data = self.data[index]
        elif isinstance(index, slice):
            # If index is a slice, handle slice indexing
            start, stop, step = index.indices(len(self))
            # Stack the window data along a new axis to create a single array
            window_data = np.stack([self[i] for i in range(start, stop, step)], axis=0)
        else:
            raise TypeError("Unsupported index type. Expected int or slice.")

        return window_data

    def _running_window_setup(self, input_data, window_size: int):
        """
        Converts a NumPy array into a NumPy ndarray with specified sliding window size.

        Parameters:
            input_data (np.array): An array with samples as rows and features as columns.
            window_size (int): Number of samples in each window.

        Returns:
            np.ndarray: NumPy ndarray with the shape of [Batch, Sample #, Feature #].
        """

        # Calculate the number of windows
    
        num_windows = input_data.shape[0] - window_size + 1

        # Create an empty array to store the windowed data
        windowed_data = []

        # Iterate over the input data and create windows
        for i in range(num_windows):
            temp = input_data[i:i+window_size].copy()
            windowed_data.append(temp)
        return windowed_data

class UniStockEnvDataStruct():
    
    def __init__(self,clean_ohlcv_df,state_cols_names, env_price_col,window_size):
        
        # Raw OHLCV Data & Price Data for Stockmarket Environment
        raw_env_df = clean_ohlcv_df.drop(columns=['date']).copy()
        raw_array = clean_ohlcv_df[state_cols_names].copy().values
        raw_price = clean_ohlcv_df[env_price_col].copy().values
        
        # Long Form DFs for Neuralforecast 
        timesnet_df = self._gen_long_form_df_from_ohlcv(clean_ohlcv_df)
        focused_timesnet_df = timesnet_df[timesnet_df['unique_id'] == env_price_col].copy().reset_index(drop=True)
        
        self.data = {'raw_df': clean_ohlcv_df,
                    'raw_env': raw_array,
                    'raw_price_env': raw_price,
                    'long_raw': timesnet_df,
                    'long_raw_price': focused_timesnet_df}

        if window_size > 1:
            rw_raw_df = RunningWindowDataset(clean_ohlcv_df, window_size)
            rw_raw_env = self._df_to_env_array(raw_env_df,window_size)
            rw_focused_timesnet_df = RunningWindowDataset(focused_timesnet_df,window_size)
            rw_closing_price = self._df_to_env_array(clean_ohlcv_df[[env_price_col]],window_size)

            self.data.update({'rw_raw_df': rw_raw_df,
                              'rw_raw_env': rw_raw_env,
                              'rw_raw_price_env':rw_closing_price,
                              'rw_long_raw_price': rw_focused_timesnet_df})

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __delitem__(self, key):
        del self.data[key]

    def __repr__(self):
        return repr(self.data)

        
    def _gen_long_form_df_from_ohlcv(self, df_date_ohlcv):
        """
        Generate long-form timeseries dataset suitable for TimesNet model from OHLCV DataFrame.

        Parameters:
        - df_date_ohlcv: DataFrame containing OHLCV data with 'date' column and columns for each value (e.g., open, high, low, close, volume).

        Returns:
        - combined_df: Long-form DataFrame with columns 'ds', 'unique_id', and 'y', suitable for TimesNet model.
        """

        # Extract names of value columns (excluding 'date')
        value_names = [col for col in df_date_ohlcv.columns if col != 'date']
        
        # Initialize dictionary to store DataFrames by value
        df_by_value = {}

        # Iterate through each value column
        for value in value_names:
            # Select 'date' and value column, copy DataFrame
            new_df = df_date_ohlcv[['date', value]].copy()

            # Rename columns to match TimesNet input format
            new_df = new_df.rename(columns={'date': 'ds', value: 'y'})

            # Add 'unique_id' column
            new_df.insert(1, 'unique_id', value)

            # Store DataFrame in dictionary with value as key
            df_by_value[value] = new_df
            
        
        # Combine DataFrames from dictionary into a single DataFrame
        combined_df = pd.concat(df_by_value.values(), axis=0, ignore_index=True)
        
        # Return combined DataFrame
        return combined_df   
        
    def _df_to_env_array(self,input_data: pd.DataFrame, window_size: int) -> np.ndarray:
        """
        Converts a NumPy array into a NumPy ndarray with specified sliding window size.

        Parameters:
            input_data (np.array): An array with samples as rows and features as columns.
            window_size (int): Number of samples in each window.

        Returns:
            np.ndarray: NumPy ndarray with the shape of [Batch, Sample #, Feature #].
        """
        # Convert input data to a NumPy ndarray
        ndarray_copy = input_data.to_numpy().copy()
        
        # Ensure input_data is at least 2D
        if ndarray_copy.ndim == 1:
            ndarray_copy = ndarray_copy.reshape(-1, 1)

        # Calculate the number of windows
        total_samples = ndarray_copy.shape[0]
        num_windows = total_samples - window_size + 1

        # Create an empty array to store the windowed data
        windowed_array = np.zeros((num_windows, window_size, ndarray_copy.shape[1]))

        # Iterate over the input data and create windows
        for i in range(num_windows):
            windowed_array[i] = ndarray_copy[i:i+window_size]

        return windowed_array
    
    def gen_idxs(self,datetime_pair):
        
        # Check if datetime_pair contains exactly two elements
        if len(datetime_pair) != 2:
            raise ValueError("Require beginning and ending date")
        
        # Check if the beginning date is before the ending date
        if datetime_pair[0] > datetime_pair[1]:
            raise ValueError("Require beginning and ending date to be in proper order")
        
        # Extract the 'date' column from the DataFrame
        df_datetime = self.data['raw_df']['date']
        
        # Filter rows within the specified date range
        ranged_df = df_datetime.loc[(df_datetime >= datetime_pair[0]) & (df_datetime <= datetime_pair[1])]
        
        # Find the index of the first and last rows within the filtered date range
        closet_start_idx = ranged_df.index[0]
        closet_end_idx = ranged_df.index[-1]
        
        # Return a tuple representing the range of row indices for the sliding window
        return (int(closet_start_idx), int(closet_end_idx))        
        
    
    def gen_rw_idxs(self,datetime_pair):
        """
        Generate tuple with indices for starting and ending rows within a specified date range.

        Args:
        - datetime_pair (tuple): A tuple containing the beginning and ending dates of the desired range.
        - uni_data_struc (dict): A dictionary containing structured data, including a DataFrame 'raw_df' with a 'date' column.

        Returns:
        - tuple: A tuple representing the range of row indices within the specified date range.
        """

        # Check if datetime_pair contains exactly two elements
        if len(datetime_pair) != 2:
            raise ValueError("Require beginning and ending date")
        
        # Check if the beginning date is before the ending date
        if datetime_pair[0] > datetime_pair[1]:
            raise ValueError("Require beginning and ending date to be in proper order")

        # Extract dimensions of the raw data
        n_rw, rw_len, _ = self.data['rw_raw_env'].shape
        
        # Extract the 'date' column from the DataFrame
        df_datetime = self.data['raw_df']['date']

        # Filter rows within the specified date range
        ranged_df = df_datetime.loc[(df_datetime >= datetime_pair[0]) & (df_datetime <= datetime_pair[1])]
        
        # Find the index of the first and last rows within the filtered date range
        closet_start_idx = ranged_df.index[0]
        closet_end_idx = ranged_df.index[-1]
        
        # Calculate the starting and ending row indices for the sliding window
        start_rw_idx = closet_start_idx
        end_rw_idx = closet_end_idx - rw_len + 1
        
        # Return a tuple representing the range of row indices for the sliding window
        return (int(start_rw_idx), int(end_rw_idx))

    

class TimesNetProcessing:
    def __init__(self, uni_data):
        """
        Initialize the TimesNetProcessing class.

        Args:
        - uni_data: Dictionary containing universal data.
        - loc_trained_model: Path to the trained model file.

        Raises:
        - FileNotFoundError: If the specified model path does not exist.
        """
        self.data = uni_data
        self.window_scaling = True
        self.scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))

    def upload_model(self,loc_trained_model):
                # Ensure the directory and file exist
        if os.path.exists(loc_trained_model):
            self.nf = NeuralForecast.load(path=loc_trained_model)
        else:
            raise FileNotFoundError(f"Model path {loc_trained_model} does not exist.")

    def process(self, env):
        """
        Process the environment data.

        Args:
        - env: Environment object.

        Returns:
        - agent_state: Processed agent state.
        """
        if hasattr(self, 'nf'):

          # Get observation from the environment
          raw_state, position = env.get_observation()
          cur_idx = env.current_step
          columns = ['open', 'high', 'low', 'close', 'volume']

          # Check if the environmental state is in the form of OHLCV data
          if raw_state.shape[1] != len(columns):
              raise ValueError('Environmental State is not in the form of OHLCV data')

          # Get the index for the current state
          std_cur_state_idx = raw_state.shape[0]

          # Create a dictionary to store environmental state by column
          env_state_by_col_dic = {col: raw_state[:, idx] for idx, col in enumerate(columns)}

          # Predict model output and extend 'close' column
          model_output = self.nf.predict(self.data[env.name]['rw_long_raw_price'][cur_idx])['timesnet'].to_numpy()
          env_state_by_col_dic['close'] = np.concatenate([env_state_by_col_dic['close'], model_output])

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
        else:
          raise AttributeError("NeuralForecast Model was not loaded")
    
    def csv_process(self, env):
                # Get observation from the environment
        raw_state, position = env.get_observation()
        cur_idx = env.current_step
        columns = ['open', 'high', 'low', 'close', 'volume']

        # Check if the environmental state is in the form of OHLCV data
        if raw_state.shape[1] != len(columns):
            raise ValueError('Environmental State is not in the form of OHLCV data')

        # Get the index for the current state
        std_cur_state_idx = raw_state.shape[0]

        # Create a dictionary to store environmental state by column
        env_state_by_col_dic = {col: raw_state[:, idx] for idx, col in enumerate(columns)}
        
        # Find prediction
        model_output_date = self.data[env.name]['rw_long_raw_price'][cur_idx]['ds'].iloc[-1]
        filtered_date = self.env_csv['date'] == model_output_date
        desired_columns = ['1d', '2d', '3d', '4d', '5d']
        model_output = self.env_csv.loc[filtered_date, desired_columns].values.flatten()
        env_state_by_col_dic['close'] = np.concatenate([env_state_by_col_dic['close'], model_output])
        
        
        # Normalize each column separately
        if self.window_scaling:
            for col in columns:
                env_state_by_col_dic[col] = self.scaler.fit_transform(env_state_by_col_dic[col].reshape(-1, 1)).flatten()
        else:
            for idx, col in enumerate(columns):
                env_state_by_col_dic[col] = self.scaler[idx].transform(env_state_by_col_dic[col].reshape(-1, 1)).flatten()

        
        # Extract normalized prediction and current state
        norm_predict = env_state_by_col_dic['close'][-5:].tolist()
        norm_current_state = [env_state_by_col_dic[col][std_cur_state_idx - 1] for col in columns]

        # Append position to the normalized current state
        norm_current_state.append(position)

        # Concatenate normalized current state with normalized prediction
        agent_state = norm_current_state + norm_predict

        return agent_state
                       
    def upload_csv(self,csv_loc):
        self.env_csv = pd.read_csv(csv_loc)
        self.env_csv['date'] = pd.to_datetime(self.env_csv['date'])
    
class ModifyDDQNAgentState:
    def __init__(self, uni_data:Dict, columns:list, csv_import=False, csv_path:str = None, scaling_type:str = None, scaler_func = None,
                 start_scale_idx:int = None, finish_scale_idx:int = None, subagent_list:list=None, device=None, add_noise:bool=False,
                 add_noise_to_cols=None,noise_mean=0, noise_std=0.1):
        """
        Initialize the ModifyDDQNAgentState class.

        Args:
        - uni_data (Dict): Dictionary containing universal data.
        - columns (list): List of column names.
        - csv_import (bool, optional): If CSV file will be used to augment enviornmental data
        - scaling_type (str, optional): Type of scaling to apply ('col' or 'rw_col').
        - scaler_func (Callable, optional): A callable that provides scaling functionality.
        
        Raises:
        - ValueError: If the scaling_type is not valid.
        """
        self.columns = columns
        self.data = uni_data
        self.scaling_type = scaling_type
        self.csv_import = csv_import
        self.csv_path = csv_path
        self.scaler = scaler_func
        self.env_csv = 'N/A'
        self.start_scale_idx = start_scale_idx
        self.finish_scale_idx = finish_scale_idx
        self.subagent_list = subagent_list
        self.device = device
        self.add_noise = add_noise
        self.add_noise_to_cols = add_noise_to_cols
        self.noise_mean = noise_mean
        self.noise_std = noise_std
        
        self._validate_agruments()

        if self.scaling_type == 'col':
            self._initiate_total_scaler(scaler_func)
        elif self.scaling_type == 'rw_col':
            self._initiate_rw_scaler(scaler_func)
        elif self.scaling_type == None:
            self.scaler = None
        else:
            raise ValueError(f'{self.scaling_type} is not a valid scaling_type arguement')
        
        if csv_import:
            self.env_csv = pd.read_csv(self.csv_path)
            self.env_csv['date'] = pd.to_datetime(self.env_csv['date'])        


    def process(self, env):
        """
        Process the environment data.

        Args:
        - env: Environment object.

        Returns:
        - agent_state (list): Processed agent state.
        
        Raises:
        - ValueError: If the environmental state does not match expected columns.
        """
        
        # Get observation from the environment
        raw_state, position, pos_idx, purchase_price, portfilo_value, prev_act, no_pos_idx = env.get_observation()
 
        # Check if the environmental state is in the form of OHLCV data
        if raw_state.shape[1] != len(self.columns):
            raise ValueError('Environmental State does not share the same number columns in Universal Data Type')

        # Get window size of a step
        std_cur_state_idx = raw_state.shape[0]

        # Create a dictionary to store environmental state by column
        env_state_by_col_dic = {col: raw_state[:, idx] for idx, col in enumerate(self.columns)}

        # Custome code for a specific CSV type 
        if self.csv_import:
            cur_idx = env.current_step
            model_output_date = self.data['rw_long_raw_price'][cur_idx]['ds'].iloc[-1]
            filtered_date = self.env_csv['date'] == model_output_date
            desired_columns = ['1d', '2d', '3d', '4d', '5d']
            model_output = self.env_csv.loc[filtered_date, desired_columns].values.flatten()
 

        # Scalerize each column separately
        if self.scaling_type == 'rw_col':
            for idx, col in enumerate(self.columns):
                env_state_by_col_dic[col] = self.scaler[idx].fit_transform(env_state_by_col_dic[col].reshape(-1, 1)).flatten()
                if col == 'close' and self.csv_import:
                    scaler_model_output = self.scaler[idx].transform(model_output.reshape(-1, 1)).flatten().tolist()
                    scaler_purchase_price = self.scaler[idx].transform(np.array(purchase_price).reshape(-1, 1)).flatten().tolist()

        elif self.scaling_type =='col':
            for idx, col in enumerate(self.columns):
                env_state_by_col_dic[col] = self.scaler[idx].transform(env_state_by_col_dic[col].reshape(-1, 1)).flatten()
                if col == 'close' and self.csv_import:
                    scaler_model_output = self.scaler[idx].transform(model_output.reshape(-1, 1)).flatten().tolist()
                    scaler_purchase_price = self.scaler[idx].transform(np.array(purchase_price).reshape(-1, 1)).flatten().tolist()
                    
        elif self.scaling_type is None:
            pass
        
        else:
            raise ValueError(f'{self.scaling_type} is not a valid scaling_type arguement')    

        # Extract scalerized current enviornment state and add noise
        if self.add_noise:
            scaler_current_state = []
            for col in self.columns:
                value = env_state_by_col_dic[col][std_cur_state_idx - 1]
                if col in self.add_noise_to_cols:
                    value = self._add_gaussian_noise(value,self.noise_mean,self.noise_std)
                scaler_current_state.append(value)
        else:
            scaler_current_state = [env_state_by_col_dic[col][std_cur_state_idx - 1] for col in self.columns]

        


        # Append position & Value to the scalerized agent state
        scaler_current_state.append(position)
        scaler_current_state.append(pos_idx)
        scaler_current_state.append(no_pos_idx)
        scaler_current_state.append(scaler_purchase_price[0])
        scaler_current_state.append(portfilo_value)
        scaler_current_state.append(prev_act)
        agent_state = scaler_current_state
        
        if self.csv_import and (self.scaling_type == 'rw_col' or self.scaling_type =='col'):
            agent_state += scaler_model_output
            
        if self.csv_import and self.scaling_type is None: ## Scaled purchase price will be here.....
            agent_state += model_output.tolist()
        
        if self.subagent_list is not None:
            subagent_outputs = []
            for subagent in self.subagent_list:
                q_vals_tensor = subagent.Q1_nn.forward(torch.tensor(agent_state,dtype=torch.float32).to(self.device))
                q_vals_list = q_vals_tensor.cpu().detach().numpy().tolist()

                # Get the ranks using the 'max' method
                ranks = rankdata(q_vals_list, method='max')
                ranks = len(q_vals_list) + 1 - ranks
                                
                subagent_outputs.extend(ranks)
            agent_state.extend(subagent_outputs)
        
        return agent_state

    def _validate_agruments(self):       
        if self.scaler is None and self.scaling_type is not None:
            raise ValueError('Argument scaling_type specifies scaling but no function is passed')
        
        if self.scaler is not None and self.scaling_type is None:
            raise ValueError('Scaler function is passed but scaling_type is not')
        
        if self.csv_import and self.csv_path is None:
            raise ValueError('No file path provided for CSV import')
        
        proper_scaling_type_args = ['col', 'rw_col', None]
        if not(self.scaling_type in proper_scaling_type_args):
            raise ValueError(f'Invalid Agrument: {self.scaling_type} for scaling_type')
        
        if self.scaler is not None and not(hasattr(self.scaler, 'fit')):
            raise AttributeError('scaler function does not have fit() method, use sklearn.preprocessing methods or as guide')
        
        if self.scaler is not None and not(hasattr(self.scaler, 'fit_transform')):
            raise AttributeError('scaler function does not have fit_transform() method, use sklearn.preprocessing methods or as guide')
        
        if self.scaling_type == 'rw_col' and (self.start_scale_idx is not None or
                                              self.finish_scale_idx is not None):
            raise ValueError('Rolling window scaling cannot have starting or finishing indices defined')
        
        if self.scaling_type == 'col' and (self.start_scale_idx is None) != (self.finish_scale_idx is None):
            raise ValueError('Improper starting or finishing indices defined')

        if self.scaling_type == 'col' and self.start_scale_idx > self.finish_scale_idx:
            raise ValueError('Starting index cannot be greater than finishing index')
        
        
    def _initiate_total_scaler(self, scaler_func):
        self.scaler = []
        if self.start_scale_idx is None and self.finish_scale_idx is None:
            for col in self.columns:
                new_scaler = scaler_func.fit(self.data['raw_df'][col].to_numpy().reshape(-1,1))             
                self.scaler.append(copy.deepcopy(new_scaler))
        else:
            for col in self.columns:
                new_scaler = scaler_func.fit(self.data['raw_df'][col].iloc[self.start_scale_idx:self.finish_scale_idx].to_numpy().reshape(-1,1))             
                self.scaler.append(copy.deepcopy(new_scaler))
    
    def _initiate_rw_scaler(self, scaler_func):
        self.scaler = []
        for _ in self.columns:
            new_scaler = scaler_func            
            self.scaler.append(copy.deepcopy(new_scaler))
    
    def add_subagents(self, subagent_list:list):
        self.subagent_list = subagent_list
    
    def remove_subagents(self):
        self.subagent_list = None
        
    def _add_gaussian_noise(self, data, mean=0, std=0.1):
        noise = np.random.normal(mean,std, data.shape)
        return data + noise
        
                   
  
  

        