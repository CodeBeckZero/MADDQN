import numpy as np

def future_profit(env,n):
    """
    Calculate the profit rate based on n futures-in-the-future stock price data and current position in the environment.
    Assumes a negative rate based on risk-free rate

    Args:
    - env: Environment object containing OHLCV raw data and position information.
    - n : Number of days in the fucutre to consider

    Returns:
    - float: Profit Rate
    """
    current_price = env.stock_price
    
    # Check if there are enough elements for the future prices
    if len(env.ohlcv_raw_data) < env.current_step + n:
        raise ValueError("Not enough OHLCV data for the future prices")
    
    # Tomorrow's Price
    tomorrows_price = env.stock_price_data[env.current_step+n][-1,0]       
    position = env.position
    reward = (((tomorrows_price - current_price)/current_price))*position
    opp_cost = 0.0002*(1-position) # Assuming risk-free return of 5% / 252 trading days
    
    return (reward - opp_cost)*100

def risk_reward(env, n):
    """
    Calculate the risk-reward ratio based on historical price data and current position in the environment.

    Args:
    - env: Environment object containing OHLCV raw data and position information.
    - n : Number of days in the fucutre to consider

    Returns:
    - float: Risk-reward ratio.
    """

    current_price = env.stock_price
    
    # Check if there are enough elements for the future prices
    if len(env.ohlcv_raw_data) < env.current_step + n:
        raise ValueError("Not enough OHLCV data for the future prices")
    
    tomorrows_price = env.stock_price_data[env.current_step:env.current_step+n][-1,0]    
    position = env.position
    
    rewards = (tomorrows_price - current_price) / current_price
    
    rewards_mean = np.mean(rewards)  # Calculate mean using NumPy's mean function
    rewards_std = np.std(rewards)  # Calculate standard deviation using NumPy's std function
    
    return (rewards_mean / rewards_std) * position
