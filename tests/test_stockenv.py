import numpy as np
import pandas as pd
import pytest
from datetime import datetime
from environments.stockenv import ContinuousOHLCVEnv
from agents.manual import ManualAgent
from utilities.data import UniStockEnvDataStruct
from rewards.stockmarket import future_profit

# Agent Setup
@pytest.fixture
def setup_manual_agent_1F():
    """Generate agent and actions with 1D reward window"""
    reward_params = {'n': 1}
    agent = ManualAgent('test_agent', None, future_profit, reward_params)
    agent.input_training_sequence(['B','H','S','H'])
    agent.input_testing_sequence(['H','B','S','H'])
    
    return agent

@pytest.fixture
def setup_manual_agent_3F():
    """Generate agent and actions with 3D reward window"""
    reward_params = {'n': 3}
    agent = ManualAgent('test_agent', None, future_profit, reward_params)
    agent.input_training_sequence(['B','H','S','H'])
    agent.input_testing_sequence(['H','B','S','H'])
    
    return agent

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



@pytest.fixture
def setup_3D_environment():
    """Create Environment with state = batch_size 3 of data"""
    env_name = 'env_3w'
    raw_data = np.array([[8,10,7,7], [7,10,5,5], [5,10,3,6], [6,7,5,5], 
                         [5,10,5,9], [9,10,2,3], [3,7,5,6],  [6,9,5,9], 
                         [9,9,2,5], [5,6,3,6]])
    raw_dates = [datetime(year=2010,month=1,day=1),
                 datetime(year=2010,month=1,day=2),
                 datetime(year=2010,month=1,day=3),
                 datetime(year=2010,month=1,day=4),
                 datetime(year=2010,month=1,day=5),
                 datetime(year=2010,month=1,day=6),
                 datetime(year=2010,month=1,day=7),
                 datetime(year=2010,month=1,day=8),
                 datetime(year=2010,month=1,day=9),
                 datetime(year=2010,month=1,day=10)]
    df_ohlc = pd.DataFrame(raw_data,columns=['open','high','low','close'])
    df_ohlc['date'] = raw_dates
    env_data = UniStockEnvDataStruct(df_ohlc,'close',3)
    env_3w = ContinuousOHLCVEnv(name=env_name, 
                                ohlcv_data = env_data['rw_raw_env'], 
                                stock_price_data = env_data['rw_raw_price_env'], 
                                commission_rate = 0.1, 
                                initial_cash = 100)

    return env_3w

def calculate_expected_portfolio_value(env, action_seq):
    
    if len(env.stock_price_data.shape) == 1:
        nd_data = True
    else:
        nd_data = False
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

def test_portfolio_values(setup_manual_agent_1F, setup_1D_environment, setup_3D_environment):
    """Test correctness of portfolio value calculation considering 1 day forward."""
    env_training_data = []
    env_testing_data = []
    agent_training_data = []
    agent_testing_data = []
    
    agent = setup_manual_agent_1F
    
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
    
def test_reward_values(setup_manual_agent_1F, setup_1D_environment, setup_3D_environment):
    """Test correctness of reward values with 1D reward, regardless of ND enviornment."""
    # Create Data holders
    env_training_data = []
    env_testing_data = []
    agent_training_data = []
    agent_testing_data = []
    
    # Create Agent
    agent = setup_manual_agent_1F
    
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


def test_environment_agent_interaction(setup_manual_agent_1F, setup_1D_environment,setup_manual_agent_3F, setup_3D_environment):
    """Verifies agent and environment actions match and are valid."""
    for n_dim in [1, 3]:
        agent = setup_manual_agent_1F if n_dim == 1 else setup_manual_agent_3F
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

