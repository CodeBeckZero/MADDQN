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

def avg_datasets(df_list: list, avg_col_list: list):
    """
    Blend multiple datasets by merging them based on a common 'date' column and averaging specified columns.

    Parameters:
        df_list (list): A list of DataFrames to be blended.
        avg_col_list (list): A list of column names to be averaged across the merged DataFrames.

    Returns:
        DataFrame: A new DataFrame containing the merged and averaged data.
        
    Raises:
        ValueError: If the DataFrames in the list do not have the same columns, index, or matching dates.
    """
    # Extract information from the first DataFrame for comparison
    first_df_columns = df_list[0].columns  # Set of column names of the first DataFrame
    first_df_dates = set(df_list[0]['date'])    # Set of unique dates in the 'date' column of the first DataFrame
    set_avg_col = set(avg_col_list)        # Set of columns to be averaged
    first_df_index = df_list[0].index           # Index of the first DataFrame

    # Check each DataFrame in the list for consistency
    for df in df_list[1:]:
        # Check if columns are the same as the first DataFrame
        if not df.columns.equals(first_df_columns):
            raise ValueError('DataFrames in list do not have the same columns.')
        # Check if index is the same as the first DataFrame
        if not df.index.equals(first_df_index):
            raise ValueError('DataFrames in list do not have the same index.')
        # Check if all columns to be blended are present in the DataFrame
        if not set_avg_col.issubset(set(df.columns)):
            raise ValueError('A DataFrame does not have the column names to blend.')
        # Check if dates are the same as the first DataFrame
        if set(df['date']) != first_df_dates:
            raise ValueError('A DataFrame does not have matching dates.')
    
    # Initial merge with the first DataFrame
    merged_df = df_list[0]

    # Iterate over the remaining DataFrames and merge on the 'date' column
    for df in df_list[1:]:
        merged_df = pd.merge(merged_df, df, on='date')
    
    # Generate column names for the averaged values based on the number of DataFrames
    avg_col_names = []
    num_of_dfs = len(df_list)
    for col_name in avg_col_list:
        temp_col_list = generate_column_names(col_name, num_of_dfs)
        avg_col_names.append(temp_col_list)
    
    # Calculate the mean for each group of columns to be averaged
    averaged_values = {}
    for cols_to_avg in avg_col_names:
        averaged_values[cols_to_avg[0]] = merged_df[cols_to_avg].mean(axis=1)
    
    # Create a new DataFrame with the averaged values
    averaged_df = pd.DataFrame(averaged_values)
    averaged_df['date'] = df_list[0]['date']

    return averaged_df

def generate_column_names(col_name, length):
    """
    Generate a list of column names based on the initial column name and length.

    Parameters:
        col_name (str): The initial column name.
            The starting name for generating new column names.
        length (int): The desired length of the list.
            The number of column names to generate.

    Returns:
        list: A list of column names.
            The list containing the generated column names.

    Raises:
        ValueError: If the length exceeds 30, as the function is designed for a maximum length of 30.
            Ensures compatibility with DataFrame merge column notation.
    """
    if length > 30:
        raise ValueError('Function only designed for a length of 30 to match df.merge notation')
    
    col_name_list = [col_name]
    for idx in range(1, length):
        if idx <= 3:
            temp_name = f"{col_name}_{chr(119 + idx)}"
            col_name_list.append(temp_name)
        else:
            temp_name = f"{col_name}_{chr(97)}{chr(93+idx)}"
            col_name_list.append(temp_name)        
    
    return col_name_list          