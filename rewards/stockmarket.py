import numpy as np

def future_profit(env,n, lower, upper):
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
    # Don Pushinment only works when portfolio is less than stock price 
    # and not in position due to location of check in stockenv.py, .step()
    # Done due to being finished because of index is after the reaward calc 
    if env.done == True: 
        return -100
        
    # Check if there are enough elements for the future prices
    if len(env.ohlcv_raw_data) < env.current_step + n:
        raise ValueError("Not enough OHLCV data for the future prices")
    
    # Tomorrow's Price
    if position == 1: #If position, then self.purchase_price != 0 
        tomorrows_price = env.stock_price_data[env.current_step+n]    
        current_price = env.purchase_price   
        reward = ((tomorrows_price - current_price)/current_price)
        opp_cost = 0

        No_Act_punshipment = 0
        # Long-Term Punishment for avoiding B&H
        if env.n_idx_position > upper:

            BH_punishment = -2.5*np.log(env.n_idx_position-upper)
        else:
            BH_punishment = 0
    else:    
        reward = 0
        # If Tomorrow's price is below today's, avoided loss need to reward
        tomorrows_price = env.stock_price_data[env.current_step+1]
        current_price = env.stock_price
        reward = ((tomorrows_price - current_price)/current_price)
        if reward < 0:
            reward = -reward #
        else:
            reward = 0
        
        BH_punishment = 0
        # # Long-Term Punishment for continuous hold
        if env.n_idx_no_position > lower:
            No_Act_punshipment = -2.5*np.log(env.n_idx_no_position-lower)
        else:
            No_Act_punshipment = 0
        
        opp_cost = 0.0002*(1-position) # Assuming risk-free return of 5% / 252 trading days
    

    # Bad Behaviour Punishment Previous action Sell and next action Buy 
    if env.previous_action == 0 and env.step_info[-1]['Env Action'] == 'B': # Little Convoluted with numbers/leters for states depending on where in code
        return - (2 * env.last_commission_cost / env.total_portfolio_value * 1000)  
    
    return (reward - opp_cost)*100 + BH_punishment + No_Act_punshipment

def risk_reward(env, n, lower, upper):
    """
    Calculate the risk-reward ratio based on future price data and current position in the environment.

    Args:
    - env: Environment object containing OHLCV raw data and position information.
    - n: Number of days in the future to consider.
    - lower: Lower threshold for no action punishment.
    - upper: Upper threshold for B&H punishment.

    Returns:
    - float: Risk-reward ratio.
    """
    position = env.position

    # Check if there are enough elements for the future prices
    if len(env.ohlcv_raw_data) < env.current_step + n:
        raise ValueError("Not enough OHLCV data for the future prices")
    
    if position == 1:  # If position, then self.purchase_price != 0 
        current_price = env.purchase_price
        future_prices = env.stock_price_data[env.current_step:env.current_step+n]
        rewards = (future_prices - current_price) / current_price
        
        rewards_mean = np.mean(rewards)
        rewards_std = np.std(rewards)
        
        # Risk-reward ratio calculation
        reward = rewards_mean / rewards_std if rewards_std != 0 else 0
        
        # Long-Term Punishment for avoiding B&H
        BH_punishment = -2.5 * np.log(env.n_idx_position - upper) if env.n_idx_position > upper else 0
        No_Act_punishment = 0  # No action punishment not applicable when in position
    
    else:
        reward = 0
        BH_punishment = 0
        
        # Long-Term Punishment for continuous hold
        No_Act_punishment = -2.5 * np.log(env.n_idx_no_position - lower) if env.n_idx_no_position > lower else 0
    
    return reward + BH_punishment + No_Act_punishment

def zero_reward(env):
    """
    Returns 0 reward to agent regardless of state and action

    Args:
    - env: Environment object containing OHLCV raw data and position information.

    Returns:
    - float: Risk-reward ratio.
    """
    return 0

def vanilia_profit(env,n):
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
    tomorrows_price = env.stock_price_data[env.current_step+n]       
    current_price = env.stock_price  
    reward = ((tomorrows_price - current_price)/current_price)

    return reward * 100 *position
