# CSV - Input and Wrangling

import numpy as np
import pandas as pd

def NASDAQ_csv_input(file_name,file_path):
    """
    ---------------------------------------------------------------------------------------------
    Converts NASDAQ stock csv files from https://www.nasdaq.com/market-activity/quotes/historical
    to pd.dataframe[date, open, high, low, close, volume] 
    with dtypes[Datetime,np.float32, np.float32, np.float32, np.float32, np.float32, np.int)
     in ascentding order
    ----------------------------------------------------------------------------------------------
    Parameters:
    -----------------------------------------------------------------------------------------------
    file_name: string name of full file name
    file_path: string name of full path to file
    ----------------------------------------------------------------------------------------------
    Returns:
    ----------------------------------------------------------------------------------------------
    pd.dataframe
    """
    # Import File
    df_ohlcv = pd.read_csv(f'{file_path}/{file_name}').iloc[::-1].reset_index()

    # Updating Column names and order
    column_names_mapping = {'Date':'date',
                            'Close/Last':'close',
                            'Volume':'volume',
                            'Open':'open',
                            'High':'high',
                            'Low':'low'}
    desired_order = ['date','open','high','low','close','volume']
    df_ohlcv = df_ohlcv.rename(columns=column_names_mapping)[desired_order]


    # Converting to Date String to datetime datatype
    df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date'] , format='%m/%d/%Y')

    # Converting currency columns to float32 datatype
    columns_with_dollars = [col for col in column_names_mapping.values() if col not in ['date','volume']]

    for col in columns_with_dollars:
        df_ohlcv[col] = df_ohlcv[col].str.replace('$', '').astype(float)

    return df_ohlcv
                         
                         
def YAHOO_csv_input(file_name,file_path):
    """
    ---------------------------------------------------------------------------------------------
    Converts YAHOO Finance stock csv files from to pd.dataframe[date, open, high, low, close, volume] 
    with dtypes[Datetime,np.float32, np.float32, np.float32, np.float32, np.float32, np.int)
    in ascentding order
    ----------------------------------------------------------------------------------------------
    Parameters:
    -----------------------------------------------------------------------------------------------
    file_name: string name of full file name
    file_path: string name of full path to file
    ----------------------------------------------------------------------------------------------
    Returns:
    ----------------------------------------------------------------------------------------------
    pd.dataframe
    """
    # Import File
    df_ohlcv = pd.read_csv(f'{file_path}/{file_name}')

    # Updating Column names and order
    column_names_mapping = {'Date':'date',
                            'Close':'close',
                            'Volume':'volume',
                            'Open':'open',
                            'High':'high',
                            'Adj Close': 'adj_close',
                            'Low':'low'}
    desired_order = ['date','open','high','low','close','volume']
    df_ohlcv = df_ohlcv.rename(columns=column_names_mapping)[desired_order]

    # List of columns to round
    columns_to_round = ['close', 'open', 'high','low']  

    # Round selected columns to 2 digits
    df_ohlcv[columns_to_round] = df_ohlcv[columns_to_round].round(2)
    
    
    # Converting to Date String to datetime datatype
    df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date'], format='%Y-%m-%d')

    return df_ohlcv

def normalize_df_ohlcv_by_row_range(df, start_row, end_row):
    """
    Normalize each column of a dataframe based on the mean and standard deviation
    calculated from a specific row range.

    Parameters:
        df (DataFrame): The input dataframe.
        start_row (int): The starting row index of the range.
        end_row (int): The ending row index of the range.

    Returns:
        DataFrame: A new dataframe with each column normalized based on the specified row range.
    """
    # Extract the subset of rows from the dataframe
    subset = df[['open','high','low','close','volume']].iloc[start_row :end_row]
    
    # Calculate the mean and standard deviation of each column within the subset
    mean_values = subset.mean()
    std_values = subset.std()
    
    # Normalize each column of the entire dataframe based on these calculated mean and standard deviation values
    normalized_df = (df[['open','high','low','close','volume']] - mean_values) / std_values
    
    return normalized_df               