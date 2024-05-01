def running_window_reward_function_1D(env):
    """
    Base reward function with 1 future day, assumes a running window where the final data 
    point of each window is the new entry: Example [[1,2,3]]"""
    n = 1 # Number of rewards considered in the future
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

def running_window_reward_function_3D(env):
    """Base reward function with 3 future day"""
    n = 3 # Number of rewards considered in the future
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