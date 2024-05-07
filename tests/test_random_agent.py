import pytest
import random
import pandas as pd
import numpy as np
from datetime import datetime
from agents.random import RandomAgent
from agents.manual import ManualAgent
from environments.stockenv import ContinuousOHLCVEnv
from utilities.data import UniStockEnvDataStruct
from rewards.stockmarket import future_profit

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
    columns = ['open','high','low','close']
    raw_data = np.array([[5,10,3,6], [6,7,5,5], [5,10,5,9], [9,10,2,3], [3,7,5,6]])
    raw_dates = [datetime(year=2010,month=1,day=1),
                 datetime(year=2010,month=1,day=2),
                 datetime(year=2010,month=1,day=3),
                 datetime(year=2010,month=1,day=4),
                 datetime(year=2010,month=1,day=5)]
    df_ohlc = pd.DataFrame(raw_data,columns=columns)
    df_ohlc['date'] = raw_dates
    env_data = UniStockEnvDataStruct(df_ohlc,'close',1)
    env_1w = ContinuousOHLCVEnv(name=env_name, 
                                ohlcv_data = env_data['raw_env'], 
                                stock_price_data = env_data['raw_price_env'], 
                                commission_rate = 0.1, 
                                initial_cash = 100)
    return env_1w


# Agent Setup
@pytest.fixture
def setup_manual_agent_1F():
    """Generate agent and actions with 1D reward window"""
    reward_params = {'n': 1}
    agent = ManualAgent('manual', None, future_profit,reward_params)
    random.seed(RANDOM_SEED)
    actions = gen_trade_seq(12)
    agent.input_testing_sequence(actions)

    return agent

@pytest.fixture
def setup_random_agent_1F():
    """Generate agent and actions with 1D reward window"""
    reward_params = {'n': 1}
    agent = RandomAgent('random', None, future_profit, reward_params)
    random.seed(RANDOM_SEED)
    
    return agent

def test_random_agent(setup_1D_environment, setup_manual_agent_1F, setup_random_agent_1F):
    """
    Test the behavior and performance of a random agent compared to a manual agent in a given environment.
    """

    # Setup Environment and Agents
    env = setup_1D_environment
    random_agent = setup_random_agent_1F
    manual_agent = setup_manual_agent_1F

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
    
    