import numpy as np
import pandas as pd
import pytest
import torch
from stockenv import ContinuousOHLCVEnv
from agents.manual import ManualAgent

# Agent Setup
@pytest.fixture
def setup_agent_1D():
    """Generate agent and actions with 1D reward window"""
    agent = ManualAgent('test_agent', None, basic_reward_function_1D)
    agent.input_training_sequence(['B','H','S','H'])
    agent.input_testing_sequence(['H','B','S','H'])
    
    return agent

@pytest.fixture
def setup_agent_3D():
    """Generate agent and actions with 3D reward window"""
    agent = ManualAgent('test_agent', None, basic_reward_function_3D)
    agent.input_training_sequence(['B','H','S','H'])
    agent.input_testing_sequence(['H','B','S','H'])
    
    return agent

def basic_reward_function_1D(env):
    """Base reward function with 1 future day"""
    n = 1 # Number of rewards considered in the future
    current_price = env.stock_price
    
    # Check if there are enough elements for the future prices
    if len(env.ohlcv_raw_data) < env.current_step + n:
        raise ValueError("Not enough OHLCV data for the future prices")
    
    # Tomorrow's Price
    if env.window_size == 1:
        tomorrows_price = env.stock_price_data[env.current_step+n]
    else :
        tomorrows_price = env.stock_price_data[env.current_step+n][-1]
        
    position = env.position
    reward = (((tomorrows_price - current_price)/current_price))*position
    opp_cost = 0.0002*(1-position) # Assuming risk-free return of 5% / 252 trading days
    
    return (reward - opp_cost)*100

def basic_reward_function_3D(env):
    """Base reward function with 1 future day"""
    n = 3 # Number of rewards considered in the future
    current_price = env.stock_price
    
    # Check if there are enough elements for the future prices
    if len(env.ohlcv_raw_data) < env.current_step + n:
        raise ValueError("Not enough OHLCV data for the future prices")
    
    # Tomorrow's Price
    if env.window_size == 1:
        tomorrows_price = env.stock_price_data[env.current_step+n]
    else :
        tomorrows_price = env.stock_price_data[env.current_step+n][-1]
        
    position = env.position
    reward = (((tomorrows_price - current_price)/current_price))*position
    opp_cost = 0.0002*(1-position) # Assuming risk-free return of 5% / 252 trading days
    
    return (reward - opp_cost)*100


@pytest.fixture
def setup_1D_environment():
    """Create Environment with state = batch_size 1 of data"""
    ohlcv_1w_data = np.array([[5,10,3,6], [6,7,5,5], [5,10,5,9], [9,10,2,3], [3,7,5,6]])
    stock_prices_1w_data = ohlcv_1w_data[:, -1]
    env_1w = ContinuousOHLCVEnv(name='env_1w', ohlcv_data=ohlcv_1w_data, stock_price_data=stock_prices_1w_data, commission_rate=0.1, initial_cash=100)
    return env_1w


@pytest.fixture
def setup_3D_environment():
    """Create Environment with state = batch_size 3 of data"""
    env_name = 'env_3w'
    ohlcv_3w_data = np.array([[[8,10,7,7], [7,10,5,5], [5,10,3,6]],
                              [[7,10,5,5], [5,10,3,6], [6,7,5,5]],
                              [[5,10,3,6], [6,7,5,5], [5,10,5,9]],
                              [[6,7,5,5], [5,10,5,9], [9,10,2,3]],
                              [[5,10,5,9], [9,10,2,3], [3,7,5,6]],
                              [[6,9,5,9], [9,9,2,5], [5,6,3,6]]])
    stock_prices_3w_data = ohlcv_3w_data[:,:,-1]
    env_3w = ContinuousOHLCVEnv(name=env_name,
                            ohlcv_data=ohlcv_3w_data,
                            stock_price_data=stock_prices_3w_data,
                            commission_rate=0.1,
                            initial_cash=100)

    return env_3w

def calculate_expected_portfolio_value(env, action_seq):
    
    if len(env.stock_price_data.shape) == 1:
        nd_data = True
    else:
        nd_data = False
    print(nd_data)     
    index_of_b = action_seq.index('B')
    index_of_s = action_seq.index('S')
    if nd_data:
        price_at_b = env.stock_price_data[index_of_b]
        price_at_s = env.stock_price_data[index_of_s]
    else:
        price_at_b = env.stock_price_data[index_of_b][-1]
        price_at_s = env.stock_price_data[index_of_s][-1]
        
    cash = env.initial_cash
    num_stock = int(np.floor((cash / price_at_b)))            
    cash -= num_stock * price_at_b
    b_trade_cost = num_stock * price_at_b * env.commission_rate 
    s_trade_cost = num_stock * price_at_s * env.commission_rate        
    return (num_stock * price_at_s) - b_trade_cost - s_trade_cost + cash

def test_portfolio_values(setup_agent_1D, setup_1D_environment, setup_3D_environment):
    """Test correctness of portfolio value calculation considering 1 day forward."""
    env_training_data = []
    env_testing_data = []
    agent_training_data = []
    agent_testing_data = []
    
    agent = setup_agent_1D 
    
    for n_dim in [1, 3]:
        # Setting enivironment and manual agent
        env = setup_1D_environment if n_dim == 1 else setup_3D_environment
        
        # Setting interactions enivironment/manual agent
        env.add_agent(agent.get_name())
        env.set_decision_agent(agent.get_name())
        agent.set_environment(env)
    
        # Train the agent
        agent.train(0, 4)
        
        # Collect Results
        agent_results = agent.get_step_data()
        agent_training_data.append(agent_results)
        env_results = env.get_step_data()
        env_training_data.append(env_results)

        # Assert portfolio value after training
        expected_portfolio_value = np.round(calculate_expected_portfolio_value(env, agent.training_seq),5)
        assert np.round(env_results['New Portfolio Value'].iloc[-1],5) == expected_portfolio_value

        # Test the agent
        agent.test(0, 4)
        
        # Collect Results
        agent_results = agent.get_step_data()
        agent_testing_data.append(agent_results)
        env_results = env.get_step_data()
        env_testing_data.append(env_results)       
       
        # Assert portfolio value after training
        expected_portfolio_value = calculate_expected_portfolio_value(env, agent.testing_seq)
        assert np.round(env_results['New Portfolio Value'].iloc[-1],5) == np.round(expected_portfolio_value)
        
        # Cleanup
        env.remove_agent(agent.get_name())
    
    # Assert all 1D and 3D portfolio values are the same (sliding window with last price)
    assert (np.round(env_training_data[0]['New Portfolio Value'],5) == np.round(env_training_data[1]['New Portfolio Value'],5)).all()
    assert (np.round(env_testing_data[0]['New Portfolio Value'],5) == np.round(env_testing_data[1]['New Portfolio Value'],5)).all()
    
def test_reward_values(setup_agent_1D, setup_1D_environment, setup_3D_environment):
    """Test correctness of reward values with 1D reward, regardless of ND enviornment."""
    # Create Data holders
    env_training_data = []
    env_testing_data = []
    agent_training_data = []
    agent_testing_data = []
    
    # Create Agent
    agent = setup_agent_1D
    
    for n_dim in [1, 3]:
        # Create Environment      
        env = setup_1D_environment if n_dim == 1 else setup_3D_environment
        
        # Setting enivironment with manual agent
        env.add_agent(agent.get_name())
        env.set_decision_agent(agent.get_name())
        agent.set_environment(env)
    
        # Train the agent
        agent.train(0, 4)
        
        # Collect Results
        agent_results = agent.get_step_data()
        agent_training_data.append(agent_results)
        env_results = env.get_step_data()
        env_training_data.append(env_results)

        # Test the agent
        agent.test(0, 4)
        
        # Collect Results
        agent_results = agent.get_step_data()
        agent_testing_data.append(agent_results)
        env_results = env.get_step_data()
        env_testing_data.append(env_results)
        
        # Clean Up
        env.remove_agent(agent.get_name())
    
        
    # Assert the rewards are the same for both 1D and 3D windows environments
    assert (agent_training_data[0]['test_agent Reward'] == agent_training_data[1]['test_agent Reward']).all()
    assert (agent_testing_data[0]['test_agent Reward'] == agent_testing_data[1]['test_agent Reward']).all()


def test_environment_agent_interaction(setup_agent_1D, setup_1D_environment,setup_agent_3D, setup_3D_environment):
    """Verifies agent and environment actions match and are valid."""
    for n_dim in [1, 3]:
        agent = setup_agent_1D if n_dim == 1 else setup_agent_3D
        env = setup_1D_environment if n_dim == 1 else setup_3D_environment
        env.add_agent(agent.get_name())
        env.set_decision_agent(agent.get_name())
        agent.set_environment(env)

        # Train the agent
        agent.train(0, 4) if n_dim == 1 else agent.train(0, 3)
        agent_results = agent.get_step_data()
        env_results = env.get_step_data()

        # Assert agent actions are registered in the environment
        assert (agent_results['test_agent Action'] == env_results['Env Action']).all()

        # Assert that available actions are correct
        assert all(element in list_values for list_values, element in zip(env_results['Available Actions'], env_results['Env Action']))

        # Test the agent
        agent.test(0, 4) if n_dim == 1 else agent.test(0, 3)
        agent_results = agent.get_step_data()
        env_results = env.get_step_data()

        # Assert agent actions are registered in the environment
        assert (agent_results['test_agent Action'] == env_results['Env Action']).all()

        # Assert that available actions are correct
        assert all(element in list_values for list_values, element in zip(env_results['Available Actions'], env_results['Env Action']))

        # Clean up
        env.remove_agent(agent.get_name())

