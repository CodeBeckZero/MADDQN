import pytest
import random
import pandas as pd
import numpy as np
from agents.random import RandomAgent
from agents.manual import ManualAgent
from environments.stockenv import ContinuousOHLCVEnv
from utilities.data import RunningWindowDataset
from rewards.stockmarket import running_window_reward_function_1D

RANDOM_SEED = 75

def gen_trade_seq(n_actions):
    """ 
    Generates trade action sequence of length n_actions with same trade action conditions as
    ContinousOHLCVEnv. 
    """
    trade_action_seq = []
    position = 0
    action_space = ('B','H','S')
    available_actions = ('H','B')
    
    for idx in range(n_actions):
        action = random.choice(available_actions)
        if action == 'B':
            position = 1
            available_actions = ('S','H')
        elif action == 'S':
            position = 0
            available_actions = ('H','B')
        
        trade_action_seq.append(action)
        
        if idx == n_actions - 2:
            if position == 1:
                available_actions = ('S')
            else:
                available_actions = ('H')
        
    return trade_action_seq

@pytest.fixture
def setup_1D_environment():
    """Create Environment with state = batch_size 1 of data"""
    env_name = 'env_1w'
    raw_data = np.array([[5,10,3,6], [6,7,5,5], [5,10,5,9], [9,10,2,3], [3,7,5,6]])
    ohlcv_1w_data = RunningWindowDataset(raw_data,1)
    stock_prices_1w_data = RunningWindowDataset(raw_data[:,-1],1) # Last value (closing price) is used for stock price
    env_1w = ContinuousOHLCVEnv(name=env_name, 
                                ohlcv_data = ohlcv_1w_data, 
                                stock_price_data = stock_prices_1w_data, 
                                commission_rate = 0.1, 
                                initial_cash = 100)
    return env_1w


# Agent Setup
@pytest.fixture
def setup_manual_agent():
    """Generate agent and actions with 1D reward window"""
    agent = ManualAgent('manual', None, running_window_reward_function_1D)
    random.seed(RANDOM_SEED)
    actions = gen_trade_seq(12)
    agent.input_testing_sequence(actions)

    return agent

@pytest.fixture
def setup_random_agent():
    """Generate agent and actions with 1D reward window"""
    agent = RandomAgent('random', None, running_window_reward_function_1D)
    random.seed(RANDOM_SEED)
    
    return agent

def test_random_agent(setup_1D_environment, setup_manual_agent, setup_random_agent):
    """
    Test the behavior and performance of a random agent compared to a manual agent in a given environment.
    """

    # Setup Environment and Agents
    env = setup_1D_environment
    random_agent = setup_random_agent
    manual_agent = setup_manual_agent

    # Setup Data Collection
    env_testing_data = []
    agent_testing_data = []

    # Test each agent
    for agent in [random_agent, manual_agent]:
        # Add agent to the environment and set as decision agent
        env.add_agent(agent.get_name())
        env.set_decision_agent(agent.get_name())
        
        # Set environment for the agent
        agent.set_environment(env)
        
        # Set random seed for reproducibility (for random agent actions to match manual agent)
        random.seed(RANDOM_SEED)

        # Test the agent
        agent.test(0, 4)

        # Collect results
        agent_results = agent.get_step_data()
        agent_testing_data.append(agent_results)
        env_results = env.get_step_data()
        env_testing_data.append(env_results)

        # Cleanup
        env.remove_agent(agent.get_name())

    # Assertions
    # Assert portfolio values are the same
    assert (np.round(env_testing_data[0]['New Portfolio Value'], 5) == np.round(env_testing_data[1]['New Portfolio Value'], 5)).all()

    # Assert rewards are the same
    assert (agent_testing_data[0]['random Reward'] == agent_testing_data[1]['manual Reward']).all()
    
    