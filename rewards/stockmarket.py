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
    position = env.position
    
    
    
    # Check if there are enough elements for the future prices
    if len(env.ohlcv_raw_data) < env.current_step + n:
        raise ValueError("Not enough OHLCV data for the future prices")
    
    # Tomorrow's Price
    if position == 1: #If position, then self.purchase_price != 0 
        tomorrows_price = env.stock_price_data[env.current_step+n][-1,0]       
        current_price = env.purchase_price   
        reward = ((tomorrows_price - current_price)/current_price)
        opp_cost = 0
    else:    
        reward = 0
        # If Tomorrow's price is below today's, avoided loss need to reward
        tomorrows_price = env.stock_price_data[env.current_step+1][-1,0]
        current_price = env.stock_price
        reward = ((tomorrows_price - current_price)/current_price)
        if reward < 0:
            reward = -reward #
        else:
            reward = 0
        
        opp_cost = 0.0002*(1-position) # Assuming risk-free return of 5% / 252 trading days
    
    # Bad Behaviour Punishment Previous action Sell and next action Buy 
    if env.previous_action == 0 and env.step_info[-1]['Env Action'] == 'B': # Little Convoluted with numbers/leters for states depending on where in code
        return - (2 * env.last_commission_cost / env.total_portfolio_value * 10)  
    
    return (reward - opp_cost)*100

def risk_reward(env, n):
    """
    Calculate the risk-reward ratio based on Future price data and current position in the environment.

    Args:
    - env: Environment object containing OHLCV raw data and position information.
    - n : Number of days in the fucutre to consider

    Returns:
    - float: Risk-reward ratio.
    """
    position = env.position

    
    # Check if there are enough elements for the future prices
    if len(env.ohlcv_raw_data) < env.current_step + n:
        raise ValueError("Not enough OHLCV data for the future prices")
    
    if position == 1: #If position, then self.purchase_price != 0 
        current_price = env.purchase_price
        tomorrows_price = env.stock_price_data[env.current_step:env.current_step+n][-1,0]    
        rewards = (tomorrows_price - current_price) / current_price
        rewards_mean = np.mean(rewards)  # Calculate mean using NumPy's mean function
        rewards_std = np.std(rewards)  # Calculate standard deviation using NumPy's std function
        # Risk-reward ratio calculation
        reward = rewards_mean / rewards_std if rewards_std != 0 else 0
    else:
        reward = 0 
    
    return reward

def zero_reward(env):
    """
    Returns 0 reward to agent regardless of state and action

    Args:
    - env: Environment object containing OHLCV raw data and position information.

    Returns:
    - float: Risk-reward ratio.
    """
    return 0