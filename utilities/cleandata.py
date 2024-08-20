# CSV - Input and Wrangling
import torch
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
    normalized_df['date'] = df['date']
    
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

def gen_OHLCV_envs(filename_dic, path, env_keys, norm_idx_range = None):
    """
    Generate OHLCV (Open, High, Low, Close, Volume) environments based on data stored in CSV files.

    Parameters:
    - filename_dic: Dictionary containing environment names as keys and corresponding file names as values.
    - path: Path to the directory containing the CSV files.
    - env_keys: List of environment names to generate environments for.
    - norm_idx_range: Optional normalization index range as a tuple (start_index, end_index).

    Returns:
    - environments: Dictionary containing generated environments with environment names as keys and DataFrames as values.
    
    Raises:
        ValueError: If normalization range isn't a list with 2 int elements, in ascending order.
    """

    # Check if normalization index range is provided and valid
    if norm_idx_range is None:
        normalize = False
    elif len(norm_idx_range) == 2 and all(isinstance(x, int) for x in norm_idx_range) and norm_idx_range[0] < norm_idx_range[1]:
        normalize = True
    else:
        raise ValueError(f'norm_idx_range: Invalid parameter -> {norm_idx_range}')

    # Initialize dictionary to store generated environments
    environments = {}

    # Iterate through each environment name and corresponding file
    for name, file in filename_dic.items():
        # Check if the environment name is in the list of keys
        if name in env_keys:
            print(name)
            # Read the CSV file using YAHOO_csv_input function
            temp_df = YAHOO_csv_input(file, path)
            # Apply normalization if enabled
            if normalize:
                temp_norm_df = normalize_df_ohlcv_by_row_range(temp_df, norm_idx_range[0], norm_idx_range[1])
                temp_norm_df['date'] = temp_df['date']
                environments[name] = temp_norm_df
            else:
                # Add the DataFrame to the environments dictionary
                environments[name] = temp_df

    # Return the dictionary of generated environments
    return environments
        
def create_avg_dataset_update_env_dic(avg_dataset_name, 
                                      df_keys_to_avg,
                                      cols_names_to_avg,
                                      envs_dic):
    """
    Create an average dataset from multiple DataFrames and update the environments dictionary.

    Parameters:
    - avg_dataset_name: Name for the average dataset.
    - df_keys_to_avg: List of keys corresponding to DataFrames in the environments dictionary to be averaged.
    - cols_names_to_avg: List of column names to average across DataFrames.
    - envs_dic: Dictionary containing environment names as keys and DataFrames as values.

    Returns:
    - None
    """

    # Initialize list to store DataFrames to be averaged
    dfs_to_blend = []

    # Iterate through each key corresponding to DataFrames to be averaged
    for key in df_keys_to_avg:
        # Add DataFrame to list
        dfs_to_blend.append(envs_dic[key])

    # Create average dataset from the list of DataFrames
    envs_dic[avg_dataset_name] = avg_datasets(dfs_to_blend, cols_names_to_avg)
    
    # Optional: Use 'pass' if function does not contain any additional logic
    pass

def gen_long_form_timesnet_from_ohlcv(df_date_ohlcv):
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

def OHLC_to_Candle_img(OHLC_window_narray: np.array) -> np.array:
    """
    Convert OHLC (Open-High-Low-Close) window data to a candlestick image.

    Parameters:
    OHLC_window_narray (np.array): OHLC window data in the shape [window_size, 4], where columns represent [open, high, low, close].

    Returns:
    np.array: Candlestick image represented as a 2D array with shape (window_size * 3) x (window_size * 3).
    """
    # Initialize a blank candlestick image
    window_size = OHLC_window_narray.shape[0]
    image_size = int(window_size * 3)
    ctl_stk_img = np.full((image_size, image_size), 255).astype(int)
    
    # Normalize OHLC array
    min_val = OHLC_window_narray.min()
    max_val = OHLC_window_narray.max()
    OHLC_window_norm = (OHLC_window_narray - min_val) / (max_val - min_val)
    
    # Find y-pixel location of data
    idy_img = np.round(OHLC_window_norm * image_size).astype(int)

    # Loop through each sample in window and generate graphic
    for img_idx, data_idx in zip(range(1, image_size, 3), range(len(idy_img))):
        
        # Sample data labels
        open_price = OHLC_window_narray[data_idx, 0]
        high_price = OHLC_window_narray[data_idx, 1]
        low_price = OHLC_window_narray[data_idx, 2]
        close_price = OHLC_window_narray[data_idx, 3]
        
        # Sample data label's y-pixel location
        open_img_loc = idy_img[data_idx, 0]
        high_img_loc = idy_img[data_idx, 1]
        low_img_loc = idy_img[data_idx, 2]
        close_img_loc = idy_img[data_idx, 3]
        
        # Determine Color of candlestick
        if open_price > close_price: # Losing day: Open > Close -> black
            color = 0
        elif open_price <= close_price: # Winning day: Open =< Close -> grey
            color = 128
        
        # Min/Max price reference for candle (fat) part of graph
        max_price = max(open_price, close_price)
        min_price = min(open_price, close_price)
        
        # Min/Max y-pixel location reference for candle (fat) part of graph
        max_loc = max(open_img_loc, close_img_loc)
        min_loc = min(open_img_loc, close_img_loc)
        
        # Generate Candle (fat) part of the graph
        ctl_stk_img[min_loc:max_loc+1, img_idx-1:img_idx+2] = color
        
        # Generate small part of the graph
        if high_price != max_price:  # Generates top narrow part of graphic
            ctl_stk_img[max_loc:high_img_loc+1, img_idx] = color
        
        if low_price < min_price: # Generates bottom narrow part of graphic
            ctl_stk_img[low_img_loc:min_loc+1, img_idx] = color

    return ctl_stk_img

def batch_data_to_tensor(input_data: np.array, window_size: int) -> torch.Tensor:
    """
    Converts a NumPy array into a PyTorch tensor with specified batch size and padding.

    Parameters:
    input_data (np.array): An array with samples as rows and features as columns.
    window_size (int): Number of samples in each batch.

    Returns:
    torch.Tensor: Tensor with the shape of [Batch, Sample #, Feature #].
    """
    # Convert input data to a tensor and perform a deep copy
    tensor_copy = torch.tensor(input_data).clone()

    # Calculate the number of elements needed to pad
    total_elements = tensor_copy.size(0)
    additional_elements = window_size - (total_elements % window_size)

    # Pad the tensor with zeros to make it evenly divisible
    n_features = tensor_copy.size(1)
    padded_tensor = torch.cat((tensor_copy, torch.zeros(additional_elements, n_features)), dim=0)

    # Reshape the padded tensor
    batch_tensor = padded_tensor.view(-1, window_size, n_features)

    return batch_tensor

def std_df_data_by_row_range_rtn_narray(df, start_row, end_row):
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
    narray_data = df.to_numpy()
    
    # Extract the subset of rows from the dataframe
    subset = narray_data[start_row:end_row,:]
    
    # Calculate the mean and standard deviation of each column within the subset
    mean = np.mean(subset, axis=0)
    std = np.std(subset, axis=0)
    
    # Standardize the data
    standardized_data = (narray_data - mean) / std
    
    return standardized_data

def flatten_state(state):
    stock_data_array, financial_data = state
    flatten_state = stock_data_array.flatten().tolist()
    flatten_state.append(financial_data)
    return flatten_state


def macro_x_stock_csv_input(file_name,file_path):
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
    column_names_mapping = {'date':'date',
                            'close':'close',
                            'volume':'volume',
                            'open':'open',
                            'high':'high',
                            'adj close': 'adj close',
                            'low':'low'}
    # desired_order = ['date','open','high','low','close','volume',
    # "1-Year Treasury Constant Maturity Rate_Log_delta",
    # "5-Year Treasury Constant Maturity Rate_Log_delta",
    # "Federal Funds Effective Rate_Log_delta",
    # "ICE BofA US High Yield Index Effective Yield_Log_delta",
    # "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity_Log_delta",
    # "10-Year Treasury Constant Maturity Minus Federal Funds Rate_Log_delta",
    # "10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity_Log_delta",
    # "Crude Oil Prices: West Texas Intermediate (WTI)_Log_delta",
    # "Crude Oil Prices: Brent - Europe_Log_delta",
    # "VIX Volatility Index_Log_delta",
    # "Wilshire 5000 Total Market Index_Log_delta",
    # "Moody's Seasoned Baa Corporate Bond Yield_Log_delta",
    # "30-Year Fixed Rate Mortgage Average_Log_delta",
    # "XAU_Close_YF_Log_delta"]

    desired_order = ['date',
    "1-Month Treasury Constant Maturity Rate_Log_delta",
    "3-Month Treasury Bill: Secondary Market Rate_Log_delta",
    "1-Year Treasury Constant Maturity Rate_Log_delta",
    "close",
    "2-Year Treasury Constant Maturity Rate_Log_delta",
    "5-Year Treasury Constant Maturity Rate_Log_delta",
    "10-Year Treasury Constant Maturity Rate_Log_delta",
    "Federal Funds Effective Rate_Log_delta",
    "Commercial Paper Interest Rate_Log_delta",
    "ICE BofA US High Yield Index Effective Yield_Log_delta",
    "ICE BofA US High Yield Index Option-Adjusted Spread (OAS)_Log_delta",
    "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity_Log_delta",
    "10-Year Treasury Constant Maturity Minus Federal Funds Rate_Log_delta",
    "10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity_Log_delta",
    "U.S. Dollars to Euro Spot Exchange Rate_Log_delta",
    "Crude Oil Prices: West Texas Intermediate (WTI)_Log_delta",
    "Crude Oil Prices: Brent - Europe_Log_delta",
    "CBOE NASDAQ 100 Volatility Index_Log_delta",
    "CBOE DJIA Volatility Index_Log_delta",
    "VIX Volatility Index_Log_delta",
    "Nasdaq Composite Index_Log_delta",
    "NASDAQ 100 Index_Log_delta",
    "Wilshire 5000 Total Market Index_Log_delta",
    "Moody's Seasoned Aaa Corporate Bond Yield_Log_delta",
    "Moody's Seasoned Baa Corporate Bond Yield_Log_delta",
    "Nominal Broad U.S. Dollar Index_Log_delta",
    "Wilshire 5000 to GDP Ratio_Log_delta",
    "SPGSCI_Log_delta",
    "open_Log_delta",
    "high_Log_delta",
    "low_Log_delta",
    "close_Log_delta",
    "adj close_Log_delta",
    "volume_Log_delta",
    "Consumer Price Index (CPI)_Log_delta",
    "Retail Sales_Log_delta",
    "Industrial Production_Log_delta",
    "Nonfarm Payrolls_Log_delta",
    "Personal Income_Log_delta",
    "Personal Consumption Expenditures (PCE)_Log_delta",
    "Business Inventories_Log_delta",
    "Consumer Credit_Log_delta",
    "Construction Spending_Log_delta",
    "Money Supply (M2)_Log_delta",
    "Consumer Price Index: All Items: Total for United States_Log_delta",
    "Equity Market Volatility Tracker: Macroeconomic News and Outlook: Other Financial Indicators_Log_delta",
    "Equity Market Volatility Tracker: Exchange Rates_Log_delta",
    "Equity Market Volatility Tracker: Housing And Land Management_Log_delta",
    "Equity Market Volatility Tracker: Competition Matters_Log_delta",
    "Equity Market Volatility Tracker: Government Sponsored Enterprises_Log_delta",
    "Equity Market Volatility Tracker: Taxes_Log_delta",
    "Equity Market Volatility Tracker: Competition Policy_Log_delta",
    "Equity Market Volatility Tracker: Labor Disputes_Log_delta",
    "Equity Market Volatility Tracker: Intellectual Property Matters_Log_delta",
    "Monetary Base; Total_Log_delta",
    "Monetary Base; Currency in Circulation_Log_delta",
    "Swiss Monetary Base Aggregate_Log_delta",
    "Currency in Circulation_Log_delta",
    "All Employees, Private Service-Providing_Log_delta",
    "Continued Claims (Insured Unemployment)_Log_delta",
    "Real personal consumption expenditures (chain-type quantity index)_Log_delta",
    "Market Value of Marketable Treasury Debt_Log_delta",
    "Coincident Economic Activity Index for the United States_Log_delta",
    "Total Construction Spending: Total Construction in the United States_Log_delta",
    "Industrial Production: Manufacturing: Durable Goods: Semiconductor and Other Electronic Component (NAICS = 3344)_Log_delta",
    "Manufacturers Inventories_Log_delta",
    "Manufacturers Sales_Log_delta",
    "Advance Retail Sales: Retail Trade_Log_delta",
    "Advance Real Retail and Food Services Sales_Log_delta",
    "Advance Retail Sales: Retail Trade and Food Services_Log_delta",
    "Retail Sales: Retail Trade_Log_delta",
    "Retailers Inventories_Log_delta",
    "Real Manufacturing and Trade Industries Sales_Log_delta",
    "Merchant Wholesalers Inventories_Log_delta",
    "S&P CoreLogic Case-Shiller U.S. National Home Price Index_Log_delta",
    "Producer Price Index by Industry: Total Manufacturing Industries_Log_delta",
    "GDP_Log_delta",
    "Real Gross Domestic Product_Log_delta",
    "Real gross domestic product per capita_Log_delta",
    "Real Potential Gross Domestic Product_Log_delta",
    "Federal government current expenditures: Interest payments_Log_delta",
    "Federal government current tax receipts_Log_delta",
    "Federal Government: Current Expenditures_Log_delta",
    "Corporate Profits After Tax (without IVA and CCAdj)_Log_delta",
    "Federal Debt: Total Public Debt_Log_delta"]

    df_ohlcv = df_ohlcv.rename(columns=column_names_mapping)[desired_order]

    # List of columns to round
    columns_to_round = [
    "1-Month Treasury Constant Maturity Rate_Log_delta",
    "3-Month Treasury Bill: Secondary Market Rate_Log_delta",
    "1-Year Treasury Constant Maturity Rate_Log_delta",
    "close",
    "2-Year Treasury Constant Maturity Rate_Log_delta",
    "5-Year Treasury Constant Maturity Rate_Log_delta",
    "10-Year Treasury Constant Maturity Rate_Log_delta",
    "Federal Funds Effective Rate_Log_delta",
    "Commercial Paper Interest Rate_Log_delta",
    "ICE BofA US High Yield Index Effective Yield_Log_delta",
    "ICE BofA US High Yield Index Option-Adjusted Spread (OAS)_Log_delta",
    "10-Year Treasury Constant Maturity Minus 2-Year Treasury Constant Maturity_Log_delta",
    "10-Year Treasury Constant Maturity Minus Federal Funds Rate_Log_delta",
    "10-Year Treasury Constant Maturity Minus 3-Month Treasury Constant Maturity_Log_delta",
    "U.S. Dollars to Euro Spot Exchange Rate_Log_delta",
    "Crude Oil Prices: West Texas Intermediate (WTI)_Log_delta",
    "Crude Oil Prices: Brent - Europe_Log_delta",
    "CBOE NASDAQ 100 Volatility Index_Log_delta",
    "CBOE DJIA Volatility Index_Log_delta",
    "VIX Volatility Index_Log_delta",
    "Nasdaq Composite Index_Log_delta",
    "NASDAQ 100 Index_Log_delta",
    "Wilshire 5000 Total Market Index_Log_delta",
    "Moody's Seasoned Aaa Corporate Bond Yield_Log_delta",
    "Moody's Seasoned Baa Corporate Bond Yield_Log_delta",
    "Nominal Broad U.S. Dollar Index_Log_delta",
    "Wilshire 5000 to GDP Ratio_Log_delta",
    "SPGSCI_Log_delta",
    "open_Log_delta",
    "high_Log_delta",
    "low_Log_delta",
    "close_Log_delta",
    "adj close_Log_delta",
    "volume_Log_delta",
    "Consumer Price Index (CPI)_Log_delta",
    "Retail Sales_Log_delta",
    "Industrial Production_Log_delta",
    "Nonfarm Payrolls_Log_delta",
    "Personal Income_Log_delta",
    "Personal Consumption Expenditures (PCE)_Log_delta",
    "Business Inventories_Log_delta",
    "Consumer Credit_Log_delta",
    "Construction Spending_Log_delta",
    "Money Supply (M2)_Log_delta",
    "Consumer Price Index: All Items: Total for United States_Log_delta",
    "Equity Market Volatility Tracker: Macroeconomic News and Outlook: Other Financial Indicators_Log_delta",
    "Equity Market Volatility Tracker: Exchange Rates_Log_delta",
    "Equity Market Volatility Tracker: Housing And Land Management_Log_delta",
    "Equity Market Volatility Tracker: Competition Matters_Log_delta",
    "Equity Market Volatility Tracker: Government Sponsored Enterprises_Log_delta",
    "Equity Market Volatility Tracker: Taxes_Log_delta",
    "Equity Market Volatility Tracker: Competition Policy_Log_delta",
    "Equity Market Volatility Tracker: Labor Disputes_Log_delta",
    "Equity Market Volatility Tracker: Intellectual Property Matters_Log_delta",
    "Monetary Base; Total_Log_delta",
    "Monetary Base; Currency in Circulation_Log_delta",
    "Swiss Monetary Base Aggregate_Log_delta",
    "Currency in Circulation_Log_delta",
    "All Employees, Private Service-Providing_Log_delta",
    "Continued Claims (Insured Unemployment)_Log_delta",
    "Real personal consumption expenditures (chain-type quantity index)_Log_delta",
    "Market Value of Marketable Treasury Debt_Log_delta",
    "Coincident Economic Activity Index for the United States_Log_delta",
    "Total Construction Spending: Total Construction in the United States_Log_delta",
    "Industrial Production: Manufacturing: Durable Goods: Semiconductor and Other Electronic Component (NAICS = 3344)_Log_delta",
    "Manufacturers Inventories_Log_delta",
    "Manufacturers Sales_Log_delta",
    "Advance Retail Sales: Retail Trade_Log_delta",
    "Advance Real Retail and Food Services Sales_Log_delta",
    "Advance Retail Sales: Retail Trade and Food Services_Log_delta",
    "Retail Sales: Retail Trade_Log_delta",
    "Retailers Inventories_Log_delta",
    "Real Manufacturing and Trade Industries Sales_Log_delta",
    "Merchant Wholesalers Inventories_Log_delta",
    "S&P CoreLogic Case-Shiller U.S. National Home Price Index_Log_delta",
    "Producer Price Index by Industry: Total Manufacturing Industries_Log_delta",
    "GDP_Log_delta",
    "Real Gross Domestic Product_Log_delta",
    "Real gross domestic product per capita_Log_delta",
    "Real Potential Gross Domestic Product_Log_delta",
    "Federal government current expenditures: Interest payments_Log_delta",
    "Federal government current tax receipts_Log_delta",
    "Federal Government: Current Expenditures_Log_delta",
    "Corporate Profits After Tax (without IVA and CCAdj)_Log_delta",
    "Federal Debt: Total Public Debt_Log_delta"
]


    # Round selected columns to 10 digits
    df_ohlcv[columns_to_round] = df_ohlcv[columns_to_round].round(10) # Todo do we really need to round?
    
    
    # Converting to Date String to datetime datatype
    # df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date'], format= '%m/%d/%Y')
    df_ohlcv['date'] = pd.to_datetime(df_ohlcv['date'], format='%m/%d/%Y')
    return df_ohlcv